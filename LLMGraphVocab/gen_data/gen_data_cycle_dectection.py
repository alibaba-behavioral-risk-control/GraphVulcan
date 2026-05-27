import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import random
from typing import List, Tuple

import networkx as nx
from tqdm import tqdm

from graph_vocab.graph_tokenizer import GraphTokenizer
from graph_vocab.graph_vocabulary import GraphVocabulary
from utils.random_graph import generate_nm_random_graph, random_relabel, generate_erdos_renyi_graph

parser = argparse.ArgumentParser()
parser.add_argument("--CoT", type=int, default=1, help="Whether Enable CoT in train data, 0=disable, 1=enable")
parser.add_argument("--num_samples", type=int, default=10000, help="Number of training samples to generate")
parser.add_argument("--min_nodes", type=int, default=11, help="Minimum number of nodes in generated graphs")
parser.add_argument("--max_nodes", type=int, default=50, help="Maximum number of nodes in generated graphs")
parser.add_argument("--split", type=str, default="train", help="train/test")
parser.add_argument("--num_splits", type=int, default=2, help="Number of data splits")
args = parser.parse_args()


graph_vocab = GraphVocabulary()
tokenizer = GraphTokenizer()


def _components_from_token_list(token_list):
    comps = []
    cur = []
    for item in token_list:
        if item["token"] == graph_vocab.GRAPH_DISCONNECT_TOKEN:
            if cur:
                comps.append(cur)
            cur = []
        else:
            cur.append(item)
    if cur:
        comps.append(cur)
    return comps


def _token_covers_nodes(token_item, nodes: Tuple[int, int]) -> bool:
    return token_item is not None and set(nodes).issubset(set(token_item.get("node_ids", [])))


def _token_for_edge(token_list, u, v):
    for item in token_list:
        tok = item["token"]
        if tok in [graph_vocab.GRAPH_CONNECT_TOKEN, graph_vocab.GRAPH_DISCONNECT_TOKEN]:
            continue
        if _token_covers_nodes(item, (u, v)):
            G_sub = graph_vocab.instantiate_graph_from_token(tok, item["node_ids"])
            if G_sub.has_edge(u, v):
                return item
    return None


def _edge_tokens_for_cycle(token_list, cycle_nodes: List[int]):
    tokens = []
    seen = set()
    edges = list(zip(cycle_nodes, cycle_nodes[1:] + [cycle_nodes[0]]))
    for u, v in edges:
        tok = _token_for_edge(token_list, u, v)
        if tok is not None:
            key = (tok["token"], tuple(sorted(tok.get("node_ids", []))))
            if key in seen:
                continue
            seen.add(key)
            tokens.append(tok)
    return tokens


def _has_cycle(G: nx.Graph):
    basis = nx.cycle_basis(G)
    return (len(basis) > 0), (basis[0] if basis else [])


def _generate_cyclic_graph(num_nodes: int):
    while True:
        # max_edges = num_nodes * (num_nodes - 1) // 2
        # min_edges = max(num_nodes, num_nodes - 1)
        # num_edges = random.randint(min_edges, max_edges)
        G = generate_nm_random_graph(num_nodes, num_nodes - 1)
        # G = generate_erdos_renyi_graph(num_nodes, p=0.3)
        G = random_relabel(G)
        has_cycle, cycle_nodes = _has_cycle(G)
        if has_cycle:
            return G, cycle_nodes


def _generate_acyclic_graph(num_nodes: int):
    G = nx.random_labeled_tree(num_nodes)
    G = random_relabel(G)
    return G


def generate_stage2_cycle_detection_samples(num_samples: int, min_nodes: int, max_nodes: int, num_splits: int = 1):
    target_cyclic = num_samples // 2
    target_acyclic = num_samples - target_cyclic
    step = (max_nodes - min_nodes + 1) / max(1, target_cyclic)
    num_node_list = [int(min_nodes + i * step) for i in range(target_cyclic)]

    samples = []
    for split_id in range(num_splits):
        cyclic_samples = []
        acyclic_samples = []
        print(f"\nGenerating samples for split {split_id + 1}/{num_splits}")
        while len(cyclic_samples) < target_cyclic or len(acyclic_samples) < target_acyclic:
            need_cyclic = len(cyclic_samples) <= len(acyclic_samples)
            idx = len(cyclic_samples) if need_cyclic else len(acyclic_samples)
            num_nodes = num_node_list[min(idx, len(num_node_list) - 1)]

            if need_cyclic:
                G, cycle_nodes = _generate_cyclic_graph(num_nodes)
                has_cycle = True
            else:
                G = _generate_acyclic_graph(num_nodes)
                has_cycle = False
                cycle_nodes = []

            token_list = tokenizer.tokenize(G, strategy="greedy+wl")
            graph_text_vocab = tokenizer.token_list_to_text(token_list)
            graph_text_edge = tokenizer.encode_edge_list(G)
            graph_text_incident = tokenizer.encode_incident(G)

            sample = {
                "task": "Graph_CycleDetection",
                "graph_text_vocab": graph_text_vocab,
                "graph_text_edge": graph_text_edge,
                "graph_text_incident": graph_text_incident,
                "token_list": token_list,
                "has_cycle": has_cycle,
                "cycle_nodes": cycle_nodes,
                "num_nodes": num_nodes,
                "split": split_id,
            }

            if has_cycle and len(cyclic_samples) < target_cyclic:
                cyclic_samples.append(sample)
                if len(cyclic_samples) % 10 == 0:
                    print(f"Split {split_id}: cyclic {len(cyclic_samples)}/{target_cyclic}")
            elif not has_cycle and len(acyclic_samples) < target_acyclic:
                acyclic_samples.append(sample)
                if len(acyclic_samples) % 10 == 0:
                    print(f"Split {split_id}: acyclic {len(acyclic_samples)}/{target_acyclic}")

        samples.extend(cyclic_samples + acyclic_samples)
    return samples


def generate_cot_reasoning(graph_text: str, has_cycle: bool, cycle_nodes: List[int], encoding_mode="GraphVocab", token_list=None):
    reasoning = ""
    if encoding_mode == "EdgeList":
        if not has_cycle:
            reasoning += (
                f"<think>\n"
                # f"The graph is: {graph_text}.\n"
                f"I search for any cycle but all edges form a tree/forest, so there is no cycle.\n"
                f"</think>\n"
                f"The answer is No."
            )
        else:
            cycle_str = " -> ".join(str(n) for n in cycle_nodes + [cycle_nodes[0]]) if cycle_nodes else ""
            reasoning += (
                f"<think>\n"
                # f"The graph is: {graph_text}.\n"
                f"I find a cycle visiting nodes {cycle_str}. Since a closed walk exists, the graph has a cycle.\n"
                f"</think>\n"
                f"The answer is Yes."
            )

    elif encoding_mode == "Incident":
        if not has_cycle:
            reasoning += (
                f"<think>\n"
                f"I examine the neighbor lists and search for any cycle, but all edges form a tree/forest, so there is no cycle.\n"
                f"</think>\n"
                f"The answer is No."
            )
        else:
            cycle_str = " -> ".join(str(n) for n in cycle_nodes + [cycle_nodes[0]]) if cycle_nodes else ""
            reasoning += (
                f"<think>\n"
                f"By examining the neighbor lists, I find a cycle visiting nodes {cycle_str}. Since a closed walk exists, the graph has a cycle.\n"
                f"</think>\n"
                f"The answer is Yes."
            )

    elif encoding_mode == "GraphVocab":
        if not has_cycle:
            reasoning += (
                f"<think>\n"
                f"I look for a set of tokens forming a closed walk, but no such combination exists, so no cycle is present.\n"
                f"</think>\n"
                f"The answer is No."
            )
        else:
            edge_tokens = _edge_tokens_for_cycle(token_list, cycle_nodes)
            # Build a token path that explicitly connects the edge tokens
            path_tokens = []
            for i, t in enumerate(edge_tokens):
                path_tokens.append(t)
                if i != len(edge_tokens) - 1:
                    path_tokens.append({"token": graph_vocab.GRAPH_CONNECT_TOKEN, "node_ids": []})
            path_tokens.append({"token": graph_vocab.GRAPH_DISCONNECT_TOKEN, "node_ids": []})

            target_graph_text = tokenizer.token_list_to_text(path_tokens, mark_connected_components=True, mark_last_disconnect=False)
            target_graph = tokenizer.decode_graph_vocab(target_graph_text)

            cycle_graph = nx.Graph()
            cycle_graph.add_nodes_from(cycle_nodes)
            cycle_graph.add_edges_from(zip(cycle_nodes, cycle_nodes[1:] + [cycle_nodes[0]]))

            cycle_graph_text = tokenizer.encode_graph_vocab(cycle_graph, mark_connected_components=True, mark_last_disconnect=False)
            # cycle_graph_text = cycle_graph_text.replace(f" {graph_vocab.GRAPH_DISCONNECT_TOKEN}", "").replace(graph_vocab.GRAPH_DISCONNECT_TOKEN, "").strip()

            residual_graph = target_graph.copy()
            residual_graph.remove_edges_from(cycle_graph.edges())
            isolated_to_remove = [n for n in cycle_graph.nodes() if residual_graph.degree(n) == 0]
            residual_graph.remove_nodes_from(isolated_to_remove)
            residual_graph_text = tokenizer.encode_graph_vocab(residual_graph, mark_connected_components=True, mark_last_disconnect=False)
            residual_graph_text = residual_graph_text.replace(graph_vocab.GRAPH_DISCONNECT_TOKEN, graph_vocab.GRAPH_CONNECT_TOKEN)
            # residual_graph_text = residual_graph_text.replace(graph_vocab.GRAPH_DISCONNECT_TOKEN, "")
            # Replace all but the final <G_Disconnect> with <G_Connect>
            output_graph_text = f"{cycle_graph_text} {graph_vocab.GRAPH_CONNECT_TOKEN} {residual_graph_text}"
            parts = output_graph_text.split()
            disconnect_idxs = [i for i, p in enumerate(parts) if p == graph_vocab.GRAPH_DISCONNECT_TOKEN]
            if len(disconnect_idxs) > 1:
                for idx in disconnect_idxs[:-1]:
                    parts[idx] = graph_vocab.GRAPH_CONNECT_TOKEN
                output_graph_text = " ".join(parts)

            cycle_str = " -> ".join(str(n) for n in cycle_nodes + [cycle_nodes[0]])
            reasoning += (
                f"<think>\n"
                # f"The graph tokens are: {graph_text}.\n"
                f"{target_graph_text} {graph_vocab.GRAPH_OP_EQ_TOKEN} {output_graph_text}. \n"
                f"{cycle_graph_text} is a cycle visiting nodes {cycle_str}. "
                f"So this graph has a cycle.\n"
                f"</think>\n"
                f"The answer is Yes."
            )
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
            graph_text = sample["graph_text_vocab"]
        elif encoding_mode == "Incident":
            graph_text = sample["graph_text_incident"]
        else:
            graph_text = sample["graph_text_edge"]
        has_cycle = sample["has_cycle"]
        cycle_nodes = sample.get("cycle_nodes", [])

        if args.CoT:
            assistant_msg = generate_cot_reasoning(
                graph_text,
                has_cycle,
                cycle_nodes,
                encoding_mode=encoding_mode,
                token_list=sample.get("token_list"),
            )
        else:
            assistant_msg = "The answer is No." if not has_cycle else "The answer is Yes."

        user_msg = (
            f"Given the following graph: {graph_text}. "
            f"Does the graph contain a cycle? Use Yes or No as the final answer."
        )

        openai_sample = {
            "task": "cycle_detection",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
        }
        openai_data.append(openai_sample)

    return openai_data


def save_to_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {filename}")


if __name__ == "__main__":
    samples = generate_stage2_cycle_detection_samples(
        num_samples=args.num_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        num_splits=args.num_splits,
    )

    print(f"\nGenerated {len(samples)} cycle-detection samples")

    openai_data_vocab = convert_to_openai_format(samples, encoding_mode="GraphVocab", system_prompt=None)


    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", "s2_cycle_detection") + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_suffix = "_CoT" if args.CoT == 1 else "_None"
    base_suffix += f"_Nodes-{args.min_nodes}-{args.max_nodes}"
    base_suffix += f"_Samples-{args.num_samples}"
    base_suffix += f"_Splits-{args.num_splits}"
    base_suffix += "_" + args.split.capitalize()

    file_name_vocab = f"GraphVocab_Stage2_CycleDetection{base_suffix}.jsonl"
    file_name_edge = f"EdgeList_Stage2_CycleDetection{base_suffix}.jsonl"
    file_name_incident = f"Incident_Stage2_CycleDetection{base_suffix}.jsonl"

    save_to_jsonl(openai_data_vocab, save_path + file_name_vocab)
    openai_data_edge = convert_to_openai_format(samples, encoding_mode="EdgeList", system_prompt=None)
    save_to_jsonl(openai_data_edge, save_path + file_name_edge)
    openai_data_incident = convert_to_openai_format(samples, encoding_mode="Incident", system_prompt=None)
    save_to_jsonl(openai_data_incident, save_path + file_name_incident)

    print("\n--- Example samples (CycleDetection) ---")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i + 1}:")
        print(f"  Graph (GraphVocab): {sample['graph_text_vocab']}")
        print(f"  Graph (EdgeList): {sample['graph_text_edge']}")
        print(f"  Has cycle: {sample['has_cycle']}")
        if sample.get('cycle_nodes'):
            cycle_str = " -> ".join(str(n) for n in sample['cycle_nodes'] + [sample['cycle_nodes'][0]])
            print(f"  Cycle nodes: {cycle_str}")
