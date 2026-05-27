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
import time
from concurrent.futures import as_completed, ProcessPoolExecutor
import os

parser = argparse.ArgumentParser()
parser.add_argument("--CoT", type=int, default=1, help="Whether Enable CoT in train data, 0=disable, 1=enable")
parser.add_argument("--num_samples", type=int, default=10000, help="Number of training samples to generate")
parser.add_argument("--min_nodes", type=int, default=6, help="Minimum number of nodes in generated graphs")
parser.add_argument("--max_nodes", type=int, default=12, help="Maximum number of nodes in generated graphs")
parser.add_argument("--split", type=str, default="train", help="train/test")
parser.add_argument("--num_splits", type=int, default=1, help="Number of data splits")
parser.add_argument("--num_workers", type=int, default=10, help="Threads to use for sample generation, accelerates MCS computation")
args = parser.parse_args()

graph_vocab = GraphVocabulary()
tokenizer = GraphTokenizer()


def check_isomorphic(G1: nx.Graph, G2: nx.Graph) -> bool:
    """
    Check whether G1 and G2 are isomorphic.
    """
    # Quick rejects: different number of nodes or edges
    if G1.number_of_nodes() != G2.number_of_nodes():
        return False
    if G1.number_of_edges() != G2.number_of_edges():
        return False
    return nx.is_isomorphic(G1, G2)

def find_maximum_common_subgraph(G1, G2):
    # gm = isomorphism.GraphMatcher(G1, G2)
    ismags = isomorphism.ISMAGS(G1, G2)
    is_isomorphic = ismags.is_isomorphic()
    if is_isomorphic:
        isomorphisms = list(ismags.isomorphisms_iter(symmetry=False))
        len(isomorphisms)
        mapping = isomorphisms[0]
        common_nodes_G1 = list(mapping.keys())
        common_subgraph = G1.subgraph(common_nodes_G1).copy()
        return common_subgraph, mapping
    else:
        # ismags = isomorphism.ISMAGS(G1, G2)
        # start = time.perf_counter()
        largest_mappings = list(ismags.largest_common_subgraph(symmetry=False))
        if largest_mappings:
            mapping = largest_mappings[0]
            common_nodes_G1 = list(mapping.keys())
            common_subgraph = G1.subgraph(common_nodes_G1).copy()
            connected_common_subgraph = nx.connected_components(common_subgraph)
            largest_cc = max(connected_common_subgraph, key=len)
            common_subgraph = common_subgraph.subgraph(largest_cc).copy()
            return common_subgraph, mapping
        else:
            return None, {}

def get_raw_reasoning_path(G1_token_list, G2_token_list, node_mapping):
    rev_node_mapping = {v: k for k, v in node_mapping.items()}
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
        mapped_tuple = tuple(node_mapping[n] for n in token["node_ids"])
        mapped_target_node_split.add(mapped_tuple)
        edges = frozenset(tuple(sorted(edge)) for edge in Graph.edges())
        mapped_edges = frozenset(tuple(sorted((node_mapping[edge[0]], node_mapping[edge[1]]))) for edge in Graph.edges())
        target_edge_split.add(edges)
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
                # decomposed_source_edge_split.append(frozenset(common_subset))
                residual_subset = fragment - common_subset
                if residual_subset:
                    operations.append({
                        "operation": "decompose",
                        "input": list(fragment),
                        "common": list(common_subset),
                        "residual": list(residual_subset)
                    })
                    # next_fragments.append(frozenset(common_subset))
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
        # only non-isomorphic case will have leftovers
        operations.append({
            "operation": "leftover",
            "input": list(remaining_fragments)
        })
    return operations


def _worker_build_single_sample(args_tuple):
    """
    args_tuple: (num_nodes, make_iso)
    """
    num_nodes, make_iso, build_reasoning = args_tuple

    local_vocab = GraphVocabulary()
    local_tokenizer = GraphTokenizer()

    def _local_build_single_sample(num_nodes, make_iso, max_attempts=32):
        for _ in range(max_attempts):

            max_edges = num_nodes * (num_nodes - 1) // 3
            min_edges = num_nodes - 1
            # num_edges = random.randint(min_edges, max_edges)
            num_edges = max_edges

            G1 = generate_nm_random_graph(num_nodes, num_edges=num_edges)

            if G1.number_of_nodes() < 1 or not nx.is_connected(G1):
                continue
            G1 = random_relabel(G1)

            if make_iso:
                G2 = random_relabel(G1)
                is_iso = True
            else:
                while True:
                    if build_reasoning:
                        # To reduce the complexity of MCS computation, generate G2 with different edge_num
                        G2 = generate_nm_random_graph(num_nodes)
                    else:
                        G2 = generate_nm_random_graph(num_nodes, num_edges=num_edges)
                    if G2.number_of_nodes() < 1 or not nx.is_connected(G2):
                        continue
                    if check_isomorphic(G1, G2):
                        continue
                    G2 = random_relabel(G2)
                    break
                is_iso = False

            token_list_1 = local_tokenizer.tokenize(G1, strategy="greedy")
            token_list_wl_1 = local_tokenizer.tokenize(G1, strategy="greedy+wl")
            token_list_2 = local_tokenizer.tokenize(G2, strategy="greedy")
            token_list_wl_2 = local_tokenizer.tokenize(G2, strategy="greedy+wl")

            # just to find out the difference between two tokenization strategies

            graph_text_vocab_wl_1 = local_tokenizer.token_list_to_text(token_list_wl_1)
            graph_text_vocab_wl_2 = local_tokenizer.token_list_to_text(token_list_wl_2)



            if build_reasoning:
                MCS, MCS_mapping = find_maximum_common_subgraph(G2, G1)
                if is_iso:
                    raw_reasoning_path = get_raw_reasoning_path(token_list_1, token_list_2, MCS_mapping)
                else:
                    if MCS is not None:
                        token_list_mcs = local_tokenizer.tokenize(MCS)
                        raw_reasoning_path = get_raw_reasoning_path(token_list_1, token_list_mcs, MCS_mapping)
                    else:
                        raw_reasoning_path = []
            else:
                raw_reasoning_path = None
            # print("Sample Status: ")
            # print(f"Is isomorphic: {is_iso}")
            # print(f"Data status: G1 nodes={G1.number_of_nodes()}, edges={G1.number_of_edges()}; ")
            # print(f"Data status: G2 nodes={G2.number_of_nodes()}, edges={G2.number_of_edges()}; ")

            graph1_text_incident = local_tokenizer.encode_incident(G1)
            graph2_text_incident = local_tokenizer.encode_incident(G2)
            
            sample = {
                "task": "Isomorphism",
                "num_nodes": num_nodes,
                "is_isomorphic": is_iso,
                "graph1_text_vocab": graph_text_vocab_wl_1,
                "graph2_text_vocab": graph_text_vocab_wl_2,
                "graph1_text_edge": local_tokenizer.encode_edge_list(G1),
                "graph2_text_edge": local_tokenizer.encode_edge_list(G2),
                "graph1_text_incident": graph1_text_incident,
                "graph2_text_incident": graph2_text_incident,
                "token_list_1": token_list_1,
                "token_list_2": token_list_2,
                "raw_reasoning_path": raw_reasoning_path,
            }
            return sample
        raise RuntimeError("Failed to build sample after multiple attempts")
    return _local_build_single_sample(num_nodes, make_iso)


def _compute_node_counts(count, min_nodes, max_nodes):
    if count == 0:
        return []
    if min_nodes == max_nodes:
        return [min_nodes] * count
    step = (max_nodes - min_nodes + 1) / max(1, count)
    return [int(min_nodes + i * step) for i in range(count)]

def generate_isomorphism_samples(num_samples, min_nodes, max_nodes, num_workers=10, num_splits=1, build_reasoning=True):
    target_iso = num_samples // 2
    target_noniso = num_samples - target_iso

    iso_jobs = [(True, n) for n in _compute_node_counts(target_iso, min_nodes, max_nodes)]
    noniso_jobs = [(False, n) for n in _compute_node_counts(target_noniso, min_nodes, max_nodes)]
    jobs = iso_jobs + noniso_jobs
    samples = []
    print(f"Generating isomorphism samples, num_workers = {num_workers}")
    for split_id in range(num_splits):
        print(f"\nGenerating samples for split {split_id + 1}/{num_splits}\n")
        iso_samples, noniso_samples = [], []
        if num_workers <= 1:
            # single-thread
            for make_iso, num_nodes in tqdm(jobs, desc="Generating samples"):
                sample = _worker_build_single_sample((num_nodes, make_iso, build_reasoning))
                (iso_samples if sample["is_isomorphic"] else noniso_samples).append(sample)
        else:
            # multi-thread
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_job = {
                    executor.submit(_worker_build_single_sample, (num_nodes, make_iso, build_reasoning)): (make_iso, num_nodes, build_reasoning)
                    for make_iso, num_nodes in jobs
                }
                for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Generating samples"):
                    sample = future.result()
                    (iso_samples if sample["is_isomorphic"] else noniso_samples).append(sample)

        iso_samples.sort(key=lambda iso_sample: iso_sample["num_nodes"])
        noniso_samples.sort(key=lambda noniso_sample: noniso_sample["num_nodes"])
        split_samples = iso_samples + noniso_samples
        samples.extend(split_samples)

    return samples


def generate_cot_reasoning(sample):
    reasoning = "<think>\n"
    reasoning += "To determine if the two graphs are isomorphic, we can follow these steps:\n"
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
                input_graph_token_text = tokenizer.encode_graph_vocab(input_graph, mark_connected_components=True, mark_last_disconnect=False)
                input_graph_token_text_list.append(input_graph_token_text)
            input_graph_token_text = f" {graph_vocab.GRAPH_CONNECT_TOKEN} ".join(input_graph_token_text_list)
            output_edges = step["output"]
            output_graph = nx.Graph()
            output_graph.add_edges_from(output_edges)
            output_graph_token_text = tokenizer.encode_graph_vocab(output_graph, mark_connected_components=True, mark_last_disconnect=False)
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
    is_iso = sample['is_isomorphic']
    if is_iso:
        answer = "All components successfully matched, therefore they are isomorphic."
    else:
        answer = "There are unmatched components, therefore they are not isomorphic."
    reasoning += answer
    reasoning += "\n</think>\n"
    return reasoning



def _format_edges_as_edgelist(edges):
    """Format a list of edges as EdgeList string: {(u,v), (x,y), ...}"""
    sorted_edges = sorted([tuple(sorted(e)) for e in edges])
    return "{" + ", ".join(f"({u},{v})" for u, v in sorted_edges) + "}"


def _format_edges_as_incident(edges):
    """Format a list of edges as Incident string describing the subgraph."""
    G = nx.Graph()
    G.add_edges_from(edges)
    nodes = sorted(G.nodes())
    parts = []
    for node in nodes:
        neighbors = sorted(G.neighbors(node))
        if len(neighbors) == 1:
            parts.append(f"Node {node} is connected to node {neighbors[0]}")
        else:
            neighbors_str = ", ".join(str(n) for n in neighbors)
            parts.append(f"Node {node} is connected to nodes {neighbors_str}")
    return "; ".join(parts)


def generate_cot_reasoning_edgelist(sample):
    """Generate CoT reasoning using EdgeList encoding format.

    Uses the same raw_reasoning_path as GraphVocab but renders each
    sub-graph as an edge-set string  {(u,v), …}  instead of graph-vocab
    tokens.
    """
    reasoning = "\n"
    return reasoning


def generate_cot_reasoning_incident(sample):
    """Generate CoT reasoning using Incident encoding format.

    Uses the same raw_reasoning_path as GraphVocab but renders each
    sub-graph in incident-list style (Node X is connected to …).
    """
    reasoning = "\n"
    return reasoning


def convert_to_openai_format(samples, encoding_mode="GraphVocab"):
    """
    Convert samples to OpenAI fine-tuning format.

    For SFT-style formatting: returns list of {"messages": [...]}
    """
    print(f"Converting to OpenAI format ({encoding_mode})...")

    # Import system_prompts module
    try:
        from gen_data.system_prompts import get_system_prompt
    except ImportError:
        from system_prompts import get_system_prompt

    system_prompt = get_system_prompt(encoding_mode)
    openai_data = []
    for sample in tqdm(samples):
        if encoding_mode == "GraphVocab":
            graph_text_1 = sample['graph1_text_vocab']
            graph_text_2 = sample['graph2_text_vocab']
        elif encoding_mode == "Incident":
            graph_text_1 = sample['graph1_text_incident']
            graph_text_2 = sample['graph2_text_incident']
        else:
            graph_text_1 = sample['graph1_text_edge']
            graph_text_2 = sample['graph2_text_edge']

        is_iso = sample['is_isomorphic']
        answer = "Yes" if is_iso else "No"
        assistant_msg = ""
        if args.CoT:
            if encoding_mode == "GraphVocab":
                assistant_msg += generate_cot_reasoning(sample)
            else:
                assistant_msg += "<think>\n"
                assistant_msg += "\n</think>\n"
        assistant_msg += f"The answer is {answer}."

        user_msg = (
            f"Given the following two graphs: G1: {graph_text_1} and G2: {graph_text_2}. "
            f"Are the two graphs isomorphic? Use Yes or No to answer."
        )

        openai_sample = {
            "task": "isomorphism",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        openai_data.append(openai_sample)

    return openai_data


def save_to_jsonl(data, filename):
    """Save OpenAI format data to a .jsonl file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {filename}")


if __name__ == '__main__':
    samples = generate_isomorphism_samples(
        num_samples=args.num_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        num_workers=args.num_workers,
        num_splits=args.num_splits,
        build_reasoning=(args.CoT == 1)
    )

    print(f"\nGenerated {len(samples)} isomorphism test samples")
    print(f"Isomorphic: {sum(1 for s in samples if s['is_isomorphic'])}")
    print(f"Non-isomorphic: {sum(1 for s in samples if not s['is_isomorphic'])}")

    # Convert to OpenAI format for both encodings
    openai_data_vocab = convert_to_openai_format(samples, encoding_mode="GraphVocab")
    

    # Save to files
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", "s2_isomorphism") + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_suffix = "_CoT" if args.CoT == 1 else "_None"
    base_suffix += f"_Nodes-{args.min_nodes}-{args.max_nodes}"
    base_suffix += f"_Samples-{args.num_samples}"
    base_suffix += f"_Splits-{args.num_splits}"
    base_suffix += "_" + args.split.capitalize()

    file_name_vocab = f"GraphVocab_Stage2_Isomorphism{base_suffix}.jsonl"
    file_name_edge = f"EdgeList_Stage2_Isomorphism{base_suffix}.jsonl"
    file_name_incident = f"Incident_Stage2_Isomorphism{base_suffix}.jsonl"

    save_to_jsonl(openai_data_vocab, save_path + file_name_vocab)
    openai_data_edge = convert_to_openai_format(samples, encoding_mode="EdgeList")
    save_to_jsonl(openai_data_edge, save_path + file_name_edge)
    openai_data_incident = convert_to_openai_format(samples, encoding_mode="Incident")
    save_to_jsonl(openai_data_incident, save_path + file_name_incident)

    # Print a few example samples
    print("\n--- Example samples (GraphVocab) ---")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Graph1 (GraphVocab): {sample['graph1_text_vocab']}")
        print(f"  Graph2 (GraphVocab): {sample['graph2_text_vocab']}")
        print(f"  Graph1 (EdgeList): {sample['graph1_text_edge']}")
        print(f"  Graph2 (EdgeList): {sample['graph2_text_edge']}")
        print(f"  Isomorphic: {sample['is_isomorphic']}")
