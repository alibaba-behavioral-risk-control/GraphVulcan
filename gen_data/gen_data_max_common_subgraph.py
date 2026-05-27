import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
from networkx.algorithms import isomorphism
import json
import random
import argparse
from tqdm import tqdm
from utils.random_graph import generate_nm_random_graph, generate_erdos_renyi_graph, random_relabel
from graph_vocab.graph_tokenizer import GraphTokenizer
from graph_vocab.graph_vocabulary import GraphVocabulary
from copy import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument("--CoT", type=int, default=1, help="Whether Enable CoT in train data, 0=disable, 1=enable")
parser.add_argument("--num_samples", type=int, default=100, help="Number of training samples to generate")
parser.add_argument("--min_nodes", type=int, default=5, help="Minimum number of nodes in generated graphs")
parser.add_argument("--max_nodes", type=int, default=10, help="Maximum number of nodes in generated graphs")
parser.add_argument("--split", type=str, default="train", help="train/test")
parser.add_argument("--num_splits", type=int, default=1, help="Number of data splits")
parser.add_argument("--num_workers", type=int, default=10, help="Threads to use for sample generation")
args = parser.parse_args()

graph_vocab = GraphVocabulary()
tokenizer = GraphTokenizer()

def find_maximum_common_subgraph(G1, G2):
    ismags = isomorphism.ISMAGS(G1, G2)
    if ismags.is_isomorphic():
        isomorphisms = list(ismags.isomorphisms_iter(symmetry=False))
        mapping = isomorphisms[0]
        # common_nodes_G1 = list(mapping.keys())
        # common_subgraph = G1.subgraph(common_nodes_G1).copy()
        common_subgraph = G1
        return common_subgraph, mapping
    else:
        # largest_mappings = list(ismags.largest_common_subgraph(symmetry=False))
        largest_mapping = next(ismags.largest_common_subgraph(symmetry=False), None)
        if largest_mapping:
            mapping = largest_mapping
            common_nodes_G1 = list(mapping.keys())
            common_subgraph = G1.subgraph(common_nodes_G1).copy()
            is_connected = nx.is_connected(common_subgraph)
            if is_connected:
                return common_subgraph, mapping
            else:
                # print("Warning: MCS is not connected. Skipping this sample.")
                return None, {}
        else:
            return None, {}


def get_raw_reasoning_path(G1_token_list, G2_token_list, MCS_node_mapping_G2_to_G1):
    rev_node_mapping = {v: k for k, v in MCS_node_mapping_G2_to_G1.items()}
    G1_list, G2_list = [], []
    source_node_split = set()
    target_node_split = set()
    mapped_target_node_split = set()
    source_edge_split = set()
    target_edge_split = set()
    mapped_target_edge_split = set()

    for token in G1_token_list:
        if token["token"] not in graph_vocab.GRAPH_STR_TOKENS:
            continue
        Graph = graph_vocab.instantiate_graph_from_token(token["token"], token["node_ids"])
        nodes_tuple = tuple(token["node_ids"])
        source_node_split.add(nodes_tuple)
        edges = frozenset(tuple(sorted(edge)) for edge in Graph.edges())
        source_edge_split.add(edges)
        G1_list.append(Graph)

    for token in G2_token_list:
        if token["token"] not in graph_vocab.GRAPH_STR_TOKENS:
            continue
        Graph = graph_vocab.instantiate_graph_from_token(token["token"], token["node_ids"])
        nodes_tuple = tuple(token["node_ids"])
        target_node_split.add(nodes_tuple)
        mapped_tuple = tuple(MCS_node_mapping_G2_to_G1[n] for n in token["node_ids"] if n in MCS_node_mapping_G2_to_G1)
        mapped_target_node_split.add(mapped_tuple)
        edges = frozenset(tuple(sorted(edge)) for edge in Graph.edges())
        mapped_edges = frozenset(tuple(sorted((MCS_node_mapping_G2_to_G1[edge[0]], MCS_node_mapping_G2_to_G1[edge[1]]))) for edge in Graph.edges() if edge[0] in MCS_node_mapping_G2_to_G1 and edge[1] in MCS_node_mapping_G2_to_G1)
        target_edge_split.add(edges)
        if len(mapped_edges) > 0:
            mapped_target_edge_split.add(mapped_edges)
        G2_list.append(Graph)

    operations = []
    sorted_mapped_targets = sorted(mapped_target_edge_split, key=lambda edges: len(edges), reverse=True)
    remaining_fragments = sorted([frozenset(edges) for edges in source_edge_split], key=lambda edges: len(edges), reverse=True)

    for mapped_target_edges in sorted_mapped_targets:
        target_edges = frozenset(tuple(sorted((rev_node_mapping[edge[0]], rev_node_mapping[edge[1]]))) for edge in mapped_target_edges)
        fragments_to_merge = []
        next_fragments = copy(remaining_fragments)
        for fragment in remaining_fragments:
            common_subset = fragment & mapped_target_edges
            if common_subset:
                residual_subset = fragment - common_subset
                if residual_subset:
                    operations.append({
                        "operation": "decompose",
                        "input": list(fragment),
                        "common": list(common_subset),
                        "residual": list(residual_subset)
                    })
                    next_fragments.append(frozenset(residual_subset))
                    next_fragments.remove(fragment)
                    fragments_to_merge.append(common_subset)
                elif common_subset == mapped_target_edges:
                    operations.append({
                        "operation": "match",
                        "input": list(fragment),
                        "match": list(target_edges),
                    })
                    next_fragments.remove(fragment)
                    break
                elif common_subset == fragment:
                    fragments_to_merge.append(common_subset)
                    next_fragments.remove(fragment)
        if len(fragments_to_merge) > 0:
            if len(fragments_to_merge) > 1:
                operations.append({
                    "operation": "merge",
                    "inputs": [list(frag) for frag in fragments_to_merge],
                    "output": mapped_target_edges
                })
            operations.append({
                "operation": "match",
                "input": list(mapped_target_edges),
                "match": list(target_edges),
            })
        remaining_fragments = next_fragments

    if len(remaining_fragments) > 0:
        operations.append({
            "operation": "leftover",
            "input": list(remaining_fragments)
        })
    return operations


def _compute_node_counts(count, min_nodes, max_nodes):
    if count == 0:
        return []
    if min_nodes == max_nodes:
        return [min_nodes] * count
    step = (max_nodes - min_nodes + 1) / max(1, count)
    return [int(min_nodes + i * step) for i in range(count)]


def _worker_build_single_sample(args_tuple):
    """Build one MCS sample. Args: (num_nodes_1, num_nodes_2, build_reasoning)"""
    num_nodes_1, num_nodes_2, build_reasoning = args_tuple
    local_vocab = GraphVocabulary()
    local_tokenizer = GraphTokenizer()

    def _local_build(max_attempts=64):
        for _ in range(max_attempts):
            # edge sampling style same as isomorphism
            # max_edges_1 = num_nodes_1 * (num_nodes_1 - 1) // 3
            # min_edges_1 = max(1, num_nodes_1 - 1)
            # max_edges_2 = num_nodes_2 * (num_nodes_2 - 1) // 3
            # min_edges_2 = max(1, num_nodes_2 - 1)

            # num_edges_1 = random.randint(min_edges_1, max_edges_1)
            # if build_reasoning:
            #     # allow different density to ease MCS computation
            #     num_edges_2 = random.randint(min_edges_2, max_edges_2)
            # else:
            #     num_edges_2 = num_edges_1 if num_nodes_1 == num_nodes_2 else random.randint(min_edges_2, max_edges_2)

            # G1 = generate_nm_random_graph(num_nodes_1, num_edges=num_edges_1)
            # G2 = generate_nm_random_graph(num_nodes_2, num_edges=num_edges_2)
            G1 = generate_erdos_renyi_graph(num_nodes_1, p=0.3)
            G2 = generate_erdos_renyi_graph(num_nodes_2, p=0.3)
            if G1.number_of_nodes() < 2 or G2.number_of_nodes() < 2:
                continue
            if not nx.is_connected(G1):
                # G1 = max((G1.subgraph(c) for c in nx.connected_components(G1)), key=len).copy()
                continue
            if not nx.is_connected(G2):
                G2 = max((G2.subgraph(c) for c in nx.connected_components(G2)), key=len).copy()
                # continue
            G1 = random_relabel(G1)
            G2 = random_relabel(G2)

            MCS, MCS_mapping_G2_to_G1 = find_maximum_common_subgraph(G2, G1)
            if MCS is None or not MCS_mapping_G2_to_G1:
                continue

            token_list_1 = local_tokenizer.tokenize(G1, strategy="greedy+wl")
            token_list_2 = local_tokenizer.tokenize(G2, strategy="greedy+wl")
            mcs_token_list = local_tokenizer.tokenize(MCS, strategy="greedy+wl")
            graph_text_vocab_1 = local_tokenizer.token_list_to_text(token_list_1)
            graph_text_vocab_2 = local_tokenizer.token_list_to_text(token_list_2)
            graph_text_edge_1 = local_tokenizer.encode_edge_list(G1)
            graph_text_edge_2 = local_tokenizer.encode_edge_list(G2)

            raw_reasoning_path = get_raw_reasoning_path(token_list_1, mcs_token_list, MCS_mapping_G2_to_G1) if build_reasoning else None
            nodes_b = sorted(list(MCS_mapping_G2_to_G1.keys()))
            nodes_a = sorted(MCS_mapping_G2_to_G1[n] for n in nodes_b)

            graph1_text_incident = local_tokenizer.encode_incident(G1)
            graph2_text_incident = local_tokenizer.encode_incident(G2)
            
            return {
                "task": "MCS",
                "num_nodes_1": num_nodes_1,
                "num_nodes_2": num_nodes_2,
                "graph1_text_vocab": graph_text_vocab_1,
                "graph2_text_vocab": graph_text_vocab_2,
                "graph1_text_edge": graph_text_edge_1,
                "graph2_text_edge": graph_text_edge_2,
                "graph1_text_incident": graph1_text_incident,
                "graph2_text_incident": graph2_text_incident,
                "token_list_1": token_list_1,
                "token_list_2": token_list_2,
                "raw_reasoning_path": raw_reasoning_path,
                "answer_nodes_1": nodes_a,
                "answer_nodes_2": nodes_b,
            }
        raise RuntimeError("Failed to build MCS sample after multiple attempts")

    return _local_build()


def generate_mcs_samples(num_samples, min_nodes, max_nodes, num_splits=1, build_reasoning=True, num_workers=10):
    samples = []
    node_list_1 = _compute_node_counts(num_samples, min_nodes, max_nodes)
    node_list_2 = _compute_node_counts(num_samples, min_nodes, max_nodes)
    jobs = list(zip(node_list_1, node_list_2))
    for split_id in range(num_splits):
        print(f"\nGenerating samples for split {split_id + 1}/{num_splits}\n")
        split_samples = []
        if num_workers <= 1:
            for n1, n2 in tqdm(jobs, desc="Generating MCS samples"):
                sample = _worker_build_single_sample((n1, n2, build_reasoning))
                split_samples.append(sample)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_job = {
                    executor.submit(_worker_build_single_sample, (n1, n2, build_reasoning)): (n1, n2)
                    for n1, n2 in jobs
                }
                for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Generating MCS samples"):
                    sample = future.result()
                    split_samples.append(sample)
        split_samples.sort(key=lambda s: (s["num_nodes_1"], s["num_nodes_2"]))
        samples.extend(split_samples)
    return samples


def generate_cot_reasoning(sample):
    reasoning = "<think>\n"
    reasoning += "To find the Maximum Common Subgraph (MCS) between the two graphs, we can decompose and match graph tokens as follows:\n"
    raw_reasoning_path = sample['raw_reasoning_path']
    for step in raw_reasoning_path:
        if step["operation"] == "decompose":
            input_edges = step["input"]
            input_graph = nx.Graph()
            input_graph.add_edges_from(input_edges)
            input_graph_token_text = tokenizer.encode_graph_vocab(input_graph, mark_connected_components=True, mark_last_disconnect=False)
            common_edges = step["common"]
            common_graph = nx.Graph()
            common_graph.add_edges_from(common_edges)
            common_graph_token_text = tokenizer.encode_graph_vocab(common_graph, mark_connected_components=True, mark_last_disconnect=False)
            residual_edges = step["residual"]
            residual_graph = nx.Graph()
            residual_graph.add_edges_from(residual_edges)
            residual_graph_token_text = tokenizer.encode_graph_vocab(residual_graph, mark_connected_components=True, mark_last_disconnect=False)
            reasoning += f"{input_graph_token_text} {graph_vocab.GRAPH_OP_EQ_TOKEN} {common_graph_token_text} {graph_vocab.GRAPH_CONNECT_TOKEN} {residual_graph_token_text}.\n"
        elif step["operation"] == "merge":
            input_edges_list = step["inputs"]
            input_graph_token_text_list = []
            for input_edges in input_edges_list:
                input_graph = nx.Graph()
                input_graph.add_edges_from(input_edges)
                input_graph_token_text = tokenizer.encode_graph_vocab(input_graph, mark_connected_components=False)
                input_graph_token_text_list.append(input_graph_token_text)
            input_graph_token_text = f" {graph_vocab.GRAPH_CONNECT_TOKEN} ".join(input_graph_token_text_list)
            output_edges = step["output"]
            output_graph = nx.Graph()
            output_graph.add_edges_from(output_edges)
            output_graph_token_text = tokenizer.encode_graph_vocab(output_graph, mark_connected_components=False)
            # output_graph_token_text = output_graph_token_text.replace(graph_vocab.GRAPH_DISCONNECT_TOKEN, "")
            reasoning += f"{input_graph_token_text} {graph_vocab.GRAPH_OP_EQ_TOKEN} {output_graph_token_text}.\n"
        elif step["operation"] == "match":
            input_edges = step["input"]
            input_graph = nx.Graph()
            input_graph.add_edges_from(input_edges)
            input_graph_token_text = tokenizer.encode_graph_vocab(input_graph, mark_connected_components=False)
            match_edges = step["match"]
            match_graph = nx.Graph()
            match_graph.add_edges_from(match_edges)
            match_graph_token_text = tokenizer.encode_graph_vocab(match_graph, mark_connected_components=False)
            reasoning += f"{input_graph_token_text} matches with {match_graph_token_text}.\n"
        elif step["operation"] == "leftover":
            input_edges_list = step["input"]
            input_graph_token_text_list = []
            for input_edges in input_edges_list:
                input_graph = nx.Graph()
                input_graph.add_edges_from(input_edges)
                input_graph_token_text = tokenizer.encode_graph_vocab(input_graph, mark_connected_components=False)
                input_graph_token_text_list.append(input_graph_token_text)
            input_graph_token_text = ", ".join(input_graph_token_text_list)
            reasoning += f"The remaining graph {input_graph_token_text} could not be matched.\n"
    nodes_a = sample['answer_nodes_1']
    nodes_b = sample['answer_nodes_2']
    reasoning += f"Thus the maximum common subgraph maps nodes {nodes_a} in Graph A to nodes {nodes_b} in Graph B.\n"
    reasoning += "</think>\n"
    reasoning += f"Graph A: [{', '.join(str(x) for x in nodes_a)}], Graph B: [{', '.join(str(x) for x in nodes_b)}]"
    return reasoning


def convert_to_openai_format(samples, encoding_mode="GraphVocab", system_prompt=None):
    print(f"Converting to OpenAI format ({encoding_mode})...")

    # Import system_prompts module
    try:
        from gen_data.system_prompts import get_system_prompt
    except ImportError:
        from system_prompts import get_system_prompt

    if system_prompt is None:
        system_prompt = get_system_prompt(encoding_mode)

    openai_data = []
    for sample in tqdm(samples):
        if encoding_mode == "GraphVocab":
            graph1_text = sample["graph1_text_vocab"]
            graph2_text = sample["graph2_text_vocab"]
        elif encoding_mode == "Incident":
            graph1_text = sample["graph1_text_incident"]
            graph2_text = sample["graph2_text_incident"]
        else:
            graph1_text = sample["graph1_text_edge"]
            graph2_text = sample["graph2_text_edge"]

        nodes_a = sample['answer_nodes_1']
        nodes_b = sample['answer_nodes_2']

        user_msg = []
        user_msg.append('You are required to solve the Maximum Common Subgraph problem. Your goal is to identify the common subgraph with the maximum number of nodes shared between the two graphs.')
        user_msg.append('You are given the following two graphs:')
        user_msg.append("Graph A:")
        user_msg.append(graph1_text)
        user_msg.append("Graph B:")
        user_msg.append(graph2_text)
        user_msg.append('Provide the indices of the nodes in the common subgraph for each graph in the following format: Graph A: [Node indices in graph A], Graph B: [Node indices in graph B].')
        user_msg.append('For example, if the common subgraph is the subgraph of node 1, 2, 3 in graph A and the subgraph of node 2, 3, 4 in graph B, you should answer: Graph A: [1, 2, 3], Graph B: [2, 3, 4].')
        user_msg = '\n'.join(user_msg)

        if args.CoT:
            if encoding_mode == "GraphVocab" and sample.get('raw_reasoning_path'):
                assistant_msg = generate_cot_reasoning(sample)
            else:
                # EdgeList / Incident CoT
                mcs_size = len(nodes_a)
                assistant_msg = (
                    f"\n"
                    f"Graph A: [{', '.join(str(x) for x in nodes_a)}], Graph B: [{', '.join(str(x) for x in nodes_b)}]"
                )
        else:
            assistant_msg = f"Graph A: [{', '.join(str(x) for x in nodes_a)}], Graph B: [{', '.join(str(x) for x in nodes_b)}]"

        openai_sample = {
            "task": "max_common_subgraph",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        openai_data.append(openai_sample)
    return openai_data


def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {filename}")


if __name__ == '__main__':


    samples = generate_mcs_samples(
        num_samples=args.num_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        num_splits=args.num_splits,
        build_reasoning=bool(args.CoT),
        num_workers=args.num_workers,
    )
    for i, sample in enumerate(samples[:3]):
        print(sample)
    openai_data_vocab = convert_to_openai_format(samples, encoding_mode="GraphVocab")

    print("[OpenAI format data]")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", "s2_max_common_subgraph") + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_suffix = ""
    base_suffix += "_CoT" if args.CoT else "_None"
    base_suffix += f"_Nodes-{args.min_nodes}-{args.max_nodes}"
    base_suffix += f"_Samples-{args.num_samples}"
    base_suffix += f"_Splits-{args.num_splits}"
    base_suffix += f"_{args.split.capitalize()}"
    file_name_vocab = f"GraphVocab_Stage2_MCS{base_suffix}.jsonl"
    file_name_edge = f"EdgeList_Stage2_MCS{base_suffix}.jsonl"
    file_name_incident = f"Incident_Stage2_MCS{base_suffix}.jsonl"

    save_to_jsonl(openai_data_vocab, save_path + file_name_vocab)
    openai_data_edge = convert_to_openai_format(samples, encoding_mode="EdgeList")
    save_to_jsonl(openai_data_edge, save_path + file_name_edge)
    openai_data_incident = convert_to_openai_format(samples, encoding_mode="Incident", system_prompt=None)
    save_to_jsonl(openai_data_incident, save_path + file_name_incident)

    # Print a few example samples
    print("\n--- Example samples ---")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i + 1}:")
        print(f"  Graph1 (GraphVocab): {sample['graph1_text_vocab']}")
        print(f"  Graph2 (GraphVocab): {sample['graph2_text_vocab']}")
        print(f"  Graph1 (EdgeList): {sample['graph1_text_edge']}")
        print(f"  Graph2 (EdgeList): {sample['graph2_text_edge']}")
        print(f"  Answer Nodes in Graph1: {sample['answer_nodes_1']}")
        print(f"  Answer Nodes in Graph2: {sample['answer_nodes_2']}")
        if args.CoT:
            print(f"  CoT Reasoning Path: {sample['raw_reasoning_path']}")
