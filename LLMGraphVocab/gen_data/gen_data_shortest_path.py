import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import argparse
from tqdm import tqdm
import networkx as nx

from utils.random_graph import generate_nm_random_graph, generate_erdos_renyi_graph, random_relabel
from graph_vocab.graph_tokenizer import GraphTokenizer
from graph_vocab.graph_vocabulary import GraphVocabulary
import os


parser = argparse.ArgumentParser()
parser.add_argument("--CoT", type=int, default=1, help="Whether Enable CoT in train data, 0=disable, 1=enable")
parser.add_argument("--num_samples", type=int, default=100, help="Number of training samples to generate")
parser.add_argument("--min_nodes", type=int, default=11, help="Minimum number of nodes in generated graphs")
parser.add_argument("--max_nodes", type=int, default=50, help="Maximum number of nodes in generated graphs")
parser.add_argument("--split", type=str, default="train", help="train/test")
parser.add_argument("--num_splits", type=int, default=1, help="Number of data splits")
args = parser.parse_args()


graph_vocab = GraphVocabulary()
tokenizer = GraphTokenizer()


def generate_stage2_shortest_path_samples(num_samples: int, min_nodes: int, max_nodes: int, num_splits: int = 1):
    """Generate stage 2 shortest-path samples across splits.

    Each sample: random graph G (with relabel), random node pair (a,b), shortest path length dist(a,b)
    or -1 if disconnected.
    """
    samples = []
    for split_id in range(num_splits):
        print(f"\nGenerating samples for split {split_id + 1}/{num_splits}\n")

        step = (max_nodes - min_nodes + 1) / max(1, num_samples)
        num_node_list = [int(min_nodes + i * step) for i in range(num_samples)]

        print(f"Generating {num_samples} shortest-path samples...")
        for i in tqdm(range(num_samples)):
            num_nodes = num_node_list[i]
            while True:
                # G = generate_nm_random_graph(num_nodes)
                G = generate_erdos_renyi_graph(num_nodes, p=0.3)
                G = random_relabel(G)

                if G.number_of_nodes() < 2:
                    continue
                min_dist = random.randint(1, 4)
                token_list = tokenizer.tokenize(G, strategy="greedy+wl")
                graph_text_vocab = tokenizer.token_list_to_text(token_list)
                graph_text_edge = tokenizer.encode_edge_list(G)

                # pick node pair with shortest path >= 2
                max_tries_ab = 50
                dist = -1
                all_path_nodes = []  # Store multiple shortest paths
                node_a = node_b = None
                for _ in range(max_tries_ab):
                    node_a, node_b = random.sample(list(G.nodes()), 2)
                    try:
                        dist = nx.shortest_path_length(G, source=node_a, target=node_b)
                        # Get all shortest paths (up to K paths)
                        all_paths = list(nx.all_shortest_paths(G, source=node_a, target=node_b))
                        # Limit to K paths (e.g., 3) to avoid too many
                        K = min(3, len(all_paths))
                        all_path_nodes = all_paths[:K]
                    except nx.NetworkXNoPath:
                        dist = -1
                        all_path_nodes = []
                    if dist >= min_dist:
                        break
                if dist < min_dist:
                    # could not find a valid pair; skip this graph
                    continue
                else:
                    break  # found valid graph and node pair

            # Generate incident encoding
            graph_text_incident = tokenizer.encode_incident(G)
            
            sample = {
                "task": "Graph_ShortestPath",
                "num_nodes": num_nodes,
                "graph_text_vocab": graph_text_vocab,
                "graph_text_edge": graph_text_edge,
                "graph_text_incident": graph_text_incident,
                "token_list": token_list,
                "node_a": node_a,
                "node_b": node_b,
                "distance": dist,
                "paths": all_path_nodes,  # Changed from "path" to "paths" to store multiple paths
            }
            samples.append(sample)
    return samples


def _components_from_token_list(token_list):
    """Split token_list into components separated by DISCONNECT tokens."""
    components = []
    current = []
    for item in token_list:
        tok = item["token"]
        if tok == graph_vocab.GRAPH_DISCONNECT_TOKEN:
            if current:
                components.append(current)
            current = []
        else:
            current.append(item)
    if current:
        components.append(current)
    return components


def _token_with_nodes(token_list, nodes):
    """Return first token covering all nodes in `nodes` (set/list)."""
    node_set = set(nodes)
    for item in token_list:
        if item["token"] in [graph_vocab.GRAPH_DISCONNECT_TOKEN, graph_vocab.GRAPH_CONNECT_TOKEN]:
            continue
        if node_set.issubset(set(item["node_ids"])):
            return item
    return None


def _shortest_in_token(token_item, u, v):
    """Compute shortest path length and path between u,v inside a graph token instance."""
    if token_item is None:
        return None, None
    G_sub = graph_vocab.instantiate_graph_from_token(token_item["token"], token_item["node_ids"])
    if u not in G_sub or v not in G_sub:
        return None, None
    try:
        seg_path = nx.shortest_path(G_sub, source=u, target=v)
        seg_len = len(seg_path) - 1
        return seg_len, seg_path
    except nx.NetworkXNoPath:
        return None, None


def generate_cot_reasoning_shortest_path(graph_text: str, node_a: int, node_b: int, distance: int, all_path_nodes, encoding_mode="EdgeList", token_list=None):
    """CoT reasoning: EdgeList/Incident keeps old style; GraphVocab uses tokens to infer connectivity and per-token distances.
    
    Args:
        all_path_nodes: List of paths, where each path is a list of nodes
    """
    # EdgeList and Incident: original behavior
    if encoding_mode in ["EdgeList", "Incident"]:
        if distance == -1:
            return (
                f"<think>\n"
                f"The graph is: {graph_text}.\n"
                f"I need the shortest path between nodes {node_a} and {node_b}.\n"
                f"After checking connectivity, there is no path connecting them, so the distance is -1 (disconnected).\n"
                f"</think>\n"
                f"The shortest path length between node {node_a} and node {node_b} is -1."
            )
        # Handle multiple paths
        if not all_path_nodes or len(all_path_nodes) == 0:
            path_str = ""
        elif len(all_path_nodes) == 1:
            path_str = " -> ".join(str(n) for n in all_path_nodes[0])
        else:
            path_strs = [" -> ".join(str(n) for n in path) for path in all_path_nodes]
            path_str = "; ".join(path_strs)
        return (
            f"<think>\n"
            f"The graph is: {graph_text}.\n"
            f"I will compute the shortest path between node {node_a} and node {node_b}.\n"
            f"The shortest path found is: {path_str}. Its length is {distance}.\n"
            f"</think>\n"
            f"The shortest path length between node {node_a} and node {node_b} is {distance}."
        )

    elif encoding_mode == "GraphVocab":
        components = _components_from_token_list(token_list)
        tok_a = _token_with_nodes(token_list, [node_a])
        tok_b = _token_with_nodes(token_list, [node_b])
        comp_a = comp_b = None
        for comp in components:
            if tok_a in comp:
                comp_a = comp
            if tok_b in comp:
                comp_b = comp
        same_component = comp_a is not None and comp_b is not None and comp_a is comp_b

        if not same_component:
            return (
                f"<think>\n"
                f"The graph tokens are: {graph_text}. `{graph_vocab.GRAPH_DISCONNECT_TOKEN}` splits components. "
                f"Node {node_a} lies in one component and node {node_b} lies in another, so they are disconnected and the shortest path is -1.\n"
                f"</think>\n"
                f"The shortest path length between node {node_a} and node {node_b} is -1."
            )

        # 2) If path exists, decompose distance by tokens
        if distance == -1 or not all_path_nodes or len(all_path_nodes) == 0:
            return (
                f"<think>\n"
                f"The token sequence shows both nodes in the same component, but no path was found, so distance is -1.\n"
                f"</think>\n"
                f"The shortest path length between node {node_a} and node {node_b} is -1."
            )

        # Process multiple possible shortest paths
        explain_lines = [
            f"Both nodes are in the same component because `{graph_vocab.GRAPH_DISCONNECT_TOKEN}` does not separate them.",
            f"I will compute the shortest path between node {node_a} and node {node_b} by checking all possible shortest paths."
        ]
        
        all_path_results = []
        
        for path_idx, path_nodes in enumerate(all_path_nodes, 1):
            token_to_edges = {}  # token_item -> list of edges (u, v)
            path_edges = list(zip(path_nodes, path_nodes[1:]))
            
            for u, v in path_edges:
                tok_uv = _token_with_nodes(token_list, [u, v])
                if tok_uv is not None:
                    tok_key = id(tok_uv)  # use object id as key
                    if tok_key not in token_to_edges:
                        token_to_edges[tok_key] = {'token_item': tok_uv, 'edges': []}
                    token_to_edges[tok_key]['edges'].append((u, v))
                else:
                    # Edge not in any token, treat as standalone
                    standalone_key = f"standalone_{u}_{v}"
                    token_to_edges[standalone_key] = {'token_item': None, 'edges': [(u, v)]}
            
            # Now decompose once per token
            per_token_segments = []
            total = 0
            
            for tok_key, tok_data in token_to_edges.items():
                tok_item = tok_data['token_item']
                edges_in_token = tok_data['edges']
                
                # Build G_tok
                if tok_item is not None:
                    G_tok = graph_vocab.instantiate_graph_from_token(tok_item["token"], tok_item["node_ids"])
                else:
                    # Standalone edge
                    G_tok = nx.Graph()
                    G_tok.add_edges_from(edges_in_token)
                
                # Build G_path: subgraph of shortest path within this token
                path_nodes_in_token = set()
                for u, v in edges_in_token:
                    path_nodes_in_token.add(u)
                    path_nodes_in_token.add(v)
                
                G_path = nx.Graph()
                G_path.add_nodes_from(path_nodes_in_token)
                G_path.add_edges_from(edges_in_token)
                
                # Compute distance within this token's path segment
                # Find the segment of path_nodes that belongs to this token
                seg_path_nodes = [n for n in path_nodes if n in path_nodes_in_token]
                seg_len = len(edges_in_token)
                
                # Build G_tok_minus
                G_residual = G_tok.copy()
                G_residual.remove_edges_from(edges_in_token)
                # Remove isolated nodes that were only in the path
                isolated_nodes = [n for n in path_nodes_in_token if G_residual.degree(n) == 0]
                G_residual.remove_nodes_from(isolated_nodes)
                
                # Encode graphs
                tok_text = tokenizer.encode_graph_vocab(G_tok, mark_connected_components=True, mark_last_disconnect=False)
                path_tok_text = tokenizer.encode_graph_vocab(G_path, mark_connected_components=False)
                if G_residual.number_of_edges() > 0:
                    residual_tok_list = tokenizer.tokenize(G_residual)
                    # Replace GRAPH_DISCONNECT_TOKEN with GRAPH_CONNECT_TOKEN in all tokens except the last one
                    for i in range(len(residual_tok_list) - 1):
                        if residual_tok_list[i]["token"] == graph_vocab.GRAPH_DISCONNECT_TOKEN:
                            residual_tok_list[i]["token"] = graph_vocab.GRAPH_CONNECT_TOKEN
                    residual_tok_text = tokenizer.token_list_to_text(residual_tok_list, mark_last_disconnect=False)
                else:
                    residual_tok_text = None
                total += seg_len
                # Store first and last node of this segment for explanation
                first_node = seg_path_nodes[0] if seg_path_nodes else edges_in_token[0][0]
                last_node = seg_path_nodes[-1] if seg_path_nodes else edges_in_token[-1][1]
                per_token_segments.append((tok_text, path_tok_text, residual_tok_text, first_node, last_node, seg_len))
            
            # Add decomposition details for this path
            for tok_text, path_tok_text, residual_tok_text, u, v, seg_len in per_token_segments:
                if residual_tok_text is not None:
                    explain_lines.append(
                        f"{tok_text} {graph_vocab.GRAPH_OP_EQ_TOKEN} {path_tok_text} {graph_vocab.GRAPH_CONNECT_TOKEN} {residual_tok_text}, "
                        f"and {path_tok_text} is a path between nodes {u} and {v}, the distance is {seg_len}."
                    )
                else:
                    explain_lines.append(
                        f"{tok_text} is a path between nodes {u} and {v}, the distance is {seg_len}."
                    )
            explain_lines.append(f"All these segments form the path {path_idx}: {' -> '.join(str(n) for n in path_nodes)}. The total distance for this path is {total}.")
            # explain_lines.append(f"  Total distance for Path {path_idx} is {total}")
            all_path_results.append(total)
            if path_idx < len(all_path_nodes):
                explain_lines.append(f"But wait, I found another possible path, let's check that next.")
            else:
                explain_lines.append(f"I think there's no more paths to check.")
        
        # Final verification

        explain_lines.append(f"After checking all {len(all_path_nodes)} path(s), the shortest path length is: {min(all_path_results)}.")
        explain_lines.append(f"Therefore, the shortest path length between node {node_a} and node {node_b} is {min(all_path_results)}.")
        reasoning = (
            "<think>\n" + "\n".join(explain_lines) + "\n</think>\n"
            + f"The shortest path length between node {node_a} and node {node_b} is {distance}."
        )
        return reasoning
    else:
        return None



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
        if sample["task"] == "Graph_ShortestPath":
            if encoding_mode == "GraphVocab":
                graph_text = sample["graph_text_vocab"]
            elif encoding_mode == "Incident":
                graph_text = sample["graph_text_incident"]
            else:
                graph_text = sample["graph_text_edge"]

            node_a = sample["node_a"]
            node_b = sample["node_b"]
            distance = sample["distance"]
            all_path_nodes = sample.get("paths", [])  # Changed from "path" to "paths"

            if args.CoT:
                assistant_msg = generate_cot_reasoning_shortest_path(
                    graph_text, node_a, node_b, distance, all_path_nodes, encoding_mode=encoding_mode,
                    token_list=sample.get("token_list")
                )
            else:
                assistant_msg = f"The shortest path length between node {node_a} and node {node_b} is {distance}."

            user_msg = (
                f"Given the following graph: {graph_text}. "
                f"What is the shortest path length between node {node_a} and node {node_b}? "
                f"Your answer should be in the format: 'The shortest path length between node {node_a} and node {node_b} is X.'"
            )
        else:
            # unsupported task
            continue

        openai_sample = {
            "task": "shortest_path",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }
        openai_data.append(openai_sample)

    return openai_data


def save_to_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {filename}")


if __name__ == "__main__":
    samples = generate_stage2_shortest_path_samples(
        num_samples=args.num_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        num_splits=args.num_splits,
    )

    print(f"\nGenerated {len(samples)} shortest-path samples")

    openai_data_vocab = convert_to_openai_format(
        samples, encoding_mode="GraphVocab", system_prompt=None
    )


    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", "s2_shortest_path") + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_suffix = "_CoT" if args.CoT == 1 else "_None"
    base_suffix += f"_Nodes-{args.min_nodes}-{args.max_nodes}"
    base_suffix += f"_Samples-{args.num_samples}"
    base_suffix += f"_Splits-{args.num_splits}"
    base_suffix += "_" + args.split.capitalize()

    file_name_vocab = f"GraphVocab_Stage2_ShortestPath{base_suffix}.jsonl"
    file_name_edge = f"EdgeList_Stage2_ShortestPath{base_suffix}.jsonl"
    file_name_incident = f"Incident_Stage2_ShortestPath{base_suffix}.jsonl"

    save_to_jsonl(openai_data_vocab, save_path + file_name_vocab)
    openai_data_edge = convert_to_openai_format(
        samples, encoding_mode="EdgeList", system_prompt=None
    )
    save_to_jsonl(openai_data_edge, save_path + file_name_edge)
    openai_data_incident = convert_to_openai_format(
        samples, encoding_mode="Incident", system_prompt=None
    )
    save_to_jsonl(openai_data_incident, save_path + file_name_incident)

    print("\n--- Example samples (ShortestPath) ---")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i + 1}:")
        print(f"  Graph (GraphVocab): {sample['graph_text_vocab']}")
        print(f"  Graph (EdgeList): {sample['graph_text_edge']}")
        print(f"  Node A: {sample['node_a']}")
        print(f"  Node B: {sample['node_b']}")
        print(f"  Distance: {sample['distance']}")
        if sample.get('paths'):
            print(f"  Paths ({len(sample['paths'])} total):")
            for idx, path in enumerate(sample['paths'], 1):
                print(f"    Path {idx}: {path}")

