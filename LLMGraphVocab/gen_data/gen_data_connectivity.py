import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
import json
import random
import argparse
from tqdm import tqdm
from utils.random_graph import generate_random_graph, generate_NLGraph_random_graph, generate_erdos_renyi_graph, \
    generate_nm_random_graph, random_relabel
from graph_vocab.graph_tokenizer import GraphTokenizer
from graph_vocab.graph_vocabulary import GraphVocabulary
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--CoT", type=int, default=1, help="Whether Enable CoT in train data, 0=disable, 1=enable")
parser.add_argument("--num_samples", type=int, default=100, help="Number of training samples to generate")
parser.add_argument("--min_nodes", type=int, default=11, help="Minimum number of nodes in generated graphs")
parser.add_argument("--max_nodes", type=int, default=50, help="Maximum number of nodes in generated graphs")
parser.add_argument("--split", type=str, default="train", help="train/test")
parser.add_argument("--num_splits", type=int, default=2, help="Number of data splits")
args = parser.parse_args()

graph_vocab = GraphVocabulary()
tokenizer = GraphTokenizer()


def check_connectivity(G, node_a, node_b):
    """
    Check if there's a path between node_a and node_b in graph G.
    Returns True if connected, False otherwise.
    """
    if node_a not in G.nodes() or node_b not in G.nodes():
        return False
    return nx.has_path(G, node_a, node_b)


def generate_stage2_connectivity_samples(num_samples, min_nodes, max_nodes, num_splits=1):
    """
    Generate training samples for connectivity test task.
    For each graph, produce BOTH GraphVocab and EdgeList encodings to keep them aligned.
    """
    # target counts (if odd, connected gets the extra by floor, disconnected gets remainder)
    target_connected = num_samples // 2
    target_disconnected = num_samples - target_connected

    step = (max_nodes - min_nodes + 1) / (target_connected)
    num_node_list = [int(min_nodes + i * step) for i in range(target_connected)]
    samples = []
    for split_id in range(num_splits):
        print(f"\nGenerating samples for split {split_id + 1}/{num_splits}")
        print(f"Generating {num_samples} connectivity test samples ({target_connected}/{target_disconnected} connected vs disconnected)...")
        connected_samples = []
        disconnected_samples = []
        while (len(connected_samples) < target_connected or len(disconnected_samples) < target_disconnected):
            need_connected = len(connected_samples) <= len(disconnected_samples)
            # Generate random graph
            idx = len(connected_samples) if need_connected else len(disconnected_samples)
            # for connectivity task, we need to manually control num_edges to reduce the probability of single connectivity component
            num_nodes = num_node_list[idx]
            p = 0.3
            if need_connected:
                # For connected samples: use standard ER random graph
                # p = np.log(num_nodes) / num_nodes
                G = generate_erdos_renyi_graph(num_nodes, p = p)
            else:

                # For disconnected samples: generate two separate connected subgraphs
                # Split nodes into two groups (roughly equal, both >= 3 nodes)
                min_component_size = max(3, num_nodes // 4)
                max_component_size = num_nodes - min_component_size

                # Randomly decide the size of first component
                size_a = random.randint(min_component_size, max_component_size)
                size_b = num_nodes - size_a

                # Ensure both components have at least min_component_size nodes
                if size_b < min_component_size:
                    continue

                # Generate two separate connected subgraphs with higher edge probability
                # Use higher p to ensure connectivity within each component


                # Generate first component
                G_a = generate_erdos_renyi_graph(size_a, p=p)
                # Ensure it's connected, if not, add edges to make it connected
                if not nx.is_connected(G_a):
                    components = list(nx.connected_components(G_a))
                    # Connect all components by adding edges between them
                    for i in range(len(components) - 1):
                        u = random.choice(list(components[i]))
                        v = random.choice(list(components[i + 1]))
                        G_a.add_edge(u, v)

                # Generate second component with offset node IDs
                G_b = generate_erdos_renyi_graph(size_b, p=p)
                # Relabel nodes in G_b to avoid conflicts
                mapping = {node: node + size_a for node in G_b.nodes()}
                G_b = nx.relabel_nodes(G_b, mapping)

                # Ensure it's connected
                if not nx.is_connected(G_b):
                    components = list(nx.connected_components(G_b))
                    for i in range(len(components) - 1):
                        u = random.choice(list(components[i]))
                        v = random.choice(list(components[i + 1]))
                        G_b.add_edge(u, v)

                # Combine the two disconnected components
                G = nx.compose(G_a, G_b)

            # Ensure graph has at least 2 nodes
            if G.number_of_nodes() < 2:
                continue

            G = random_relabel(G)
            # GraphVocab encoding
            token_list = tokenizer.tokenize(G)
            graph_text_vocab = tokenizer.token_list_to_text(token_list)
            # EdgeList encoding
            graph_text_edge = tokenizer.encode_edge_list(G)

            connected_components = [list(comp) for comp in nx.connected_components(G)]


            if need_connected:
                candidate_components = [comp for comp in connected_components if len(comp) >= 3]
                if not candidate_components:
                    continue
                comp = random.choice(candidate_components)
                node_a, node_b = random.sample(comp, 2)
            else:
                # For disconnected samples generated by our method, we should have exactly 2 components
                # Filter components with at least 3 nodes
                large_components = [comp for comp in connected_components if len(comp) >= 3]
                if len(large_components) < 2:
                    continue
                # Sort components by size to ensure we pick the two largest ones
                large_components.sort(key=len, reverse=True)
                comp_a, comp_b = large_components[0], large_components[1]
                node_a = random.choice(comp_a)
                node_b = random.choice(comp_b)


            # Check connectivity
            is_connected = check_connectivity(G, node_a, node_b)
            if is_connected != need_connected:
                print("Logic error in connectivity sampling.")
                raise ValueError("Logic error in connectivity sampling.")
                continue

            # Generate incident encoding
            graph_text_incident = tokenizer.encode_incident(G)
            
            # Create sample with all encodings
            sample = {
                'task': 'Connectivity',
                'graph_text_vocab': graph_text_vocab,
                'graph_text_edge': graph_text_edge,
                'graph_text_incident': graph_text_incident,
                'token_list': token_list,  # Save token_list for CoT generation (GraphVocab)
                'node_a': node_a,
                'node_b': node_b,
                'is_connected': is_connected,
                'num_nodes': num_nodes,
                'num_edges': G.number_of_edges(),
                'split': split_id
            }

            if is_connected:
                connected_samples.append(sample)
            else:
                disconnected_samples.append(sample)
            if len(connected_samples) % 10 == 0:
                print(f"Split {split_id}: Generated {len(connected_samples)} /{target_connected} connected samples and {len(disconnected_samples)}/ {target_disconnected} disconnected samples")
    
        # combine and, if short, allow slight imbalance
        split_samples = connected_samples + disconnected_samples
        samples.extend(split_samples)
        # If we failed to reach exact targets, fill remaining slots with whatever we have (keeps behavior deterministic)
        # if len(samples) < num_samples:
        #     extra_needed = num_samples - len(samples)
        #     # reuse whichever class has remaining attempts
        #     leftovers = connected_samples if len(connected_samples) < len(disconnected_samples) else disconnected_samples
        #     samples.extend(leftovers[:extra_needed])

    return samples


def find_connected_component_tokens(token_list, node_a):
    """
    Find all graph tokens in the same connected component as node_a.
    Token sequence structure: token1 <G_Connect> token2 <G_Connect> ... <G_Disconnect> token3 ...
    
    Returns:
        tuple: (graph_token_a_index, connected_tokens_indices, connected_tokens_info)
        - graph_token_a_index: index of the token containing node_a
        - connected_tokens_indices: list of indices of tokens in the same component
        - connected_tokens_info: list of dicts with token info (token, nodes_in_G, index)
    """
    # Find the token containing node_a
    graph_token_a_index = None
    for i, item in enumerate(token_list):
        if item["token"] not in [graph_vocab.GRAPH_CONNECT_TOKEN, graph_vocab.GRAPH_DISCONNECT_TOKEN]:
            if node_a in item["node_ids"]:
                graph_token_a_index = i
                break
    
    if graph_token_a_index is None:
        return None, [], []
    
    # Find all tokens in the same connected component
    # Group tokens separated by <G_Connect> but separated by <G_Disconnect>
    current_component = []
    components = []
    
    for i, item in enumerate(token_list):
        token = item["token"]
        if token == graph_vocab.GRAPH_DISCONNECT_TOKEN:
            # End of current component
            if current_component:
                components.append(current_component)
            current_component = []
        elif token == graph_vocab.GRAPH_CONNECT_TOKEN:
            # Continue current component and add the connect token
            current_component.append(i)
        else:
            # It's a graph token, add to current component
            current_component.append(i)
    
    # Add the last component (if any)
    if current_component:
        components.append(current_component)
    
    # Find which component contains graph_token_a_index
    target_component = None
    for comp in components:
        if graph_token_a_index in comp:
            target_component = comp
            break
    
    if target_component:
        connected_indices = set(target_component)
        connected_tokens_info = [
            {
                'index': idx,
                'token': token_list[idx]['token'],
                'node_ids': token_list[idx]['node_ids']
            }
            for idx in target_component
        ]
        return graph_token_a_index, list(connected_indices), connected_tokens_info
    else:
        # Fallback: just return the token containing node_a
        return graph_token_a_index, [graph_token_a_index], [
            {
                'index': graph_token_a_index,
                'token': token_list[graph_token_a_index]['token'],
                'node_ids': token_list[graph_token_a_index]['node_ids']
            }
        ]



def generate_cot_reasoning(token_list, node_a, node_b, is_connected):
    graph_token_a_idx, connected_indices, connected_token_list = find_connected_component_tokens(token_list, node_a)
    if graph_token_a_idx is None:
        return (
            f"<think>\n"
            f"Node {node_a} is not found in the token sequence, so I fallback to the provided label.\n"
            f"</think>\nThe answer is No."
        )

    # graph_token_a = token_list[graph_token_a_idx]
    # graph_token_a_str = tokenizer.token_list_to_text([graph_token_a])

    # Find the shortest contiguous subsequence in the component that starts with a token containing
    # one node and ends with a token containing the other (node_a -> node_b or node_b -> node_a).
    positions_a = [i for i, t in enumerate(connected_token_list) if node_a in t["node_ids"]]
    positions_b = [i for i, t in enumerate(connected_token_list) if node_b in t["node_ids"]]

    best_span = None# (start_idx, end_idx)
    a_start = True
    if positions_a and positions_b:
        for start in positions_a:
            end_candidates = [p for p in positions_b if p >= start]
            if end_candidates:
                end = min(end_candidates)
                span_len = end - start
                if best_span is None or span_len < (best_span[1] - best_span[0]):
                    best_span = (start, end)
                    a_start = True
        for start in positions_b:
            end_candidates = [p for p in positions_a if p >= start]
            if end_candidates:
                end = min(end_candidates)
                span_len = end - start
                if best_span is None or span_len < (best_span[1] - best_span[0]):
                    best_span = (start, end)
                    a_start = False

    if best_span is not None:
        display_tokens = connected_token_list[best_span[0] : best_span[1] + 1]
        if not a_start:
            display_tokens = display_tokens[::-1]
    else:
        display_tokens = connected_token_list

    graph_token_a_str = tokenizer.token_list_to_text([display_tokens[0]])


    # connected_tokens_strs = []
    graph_token_b_str = None
    for token_info in display_tokens:
        # nodes_str = "<NidS>".join(str(n) for n in token_info['node_ids'])
        # token_str = f"<NidB>{nodes_str}<NidE>{token_info['token']}"
        # connected_tokens_strs.append(token_str)
        if node_b in token_info['node_ids']:
            graph_token_b_str = tokenizer.token_list_to_text([token_info])

    has_disconnect_token = any(item["token"] == graph_vocab.GRAPH_DISCONNECT_TOKEN for item in token_list[:-1]) # ignore last one token
    node_b_found = graph_token_b_str is not None
    predicted_connected = not has_disconnect_token or node_b_found


    if not is_connected and len(display_tokens) > 50:
        # truncate for disconnected cases
        display_tokens = display_tokens[:30]
    connected_tokens_strs = tokenizer.token_list_to_text(display_tokens)


    final_answer = "Yes" if predicted_connected else "No"
    labeled_answer = "Yes" if is_connected else "No"
    if final_answer != labeled_answer:
        # should be the same, but in case of mismatch, raise error
        raise ValueError(f"Predicted connected: {predicted_connected}, labeled connected: {is_connected}")

    if not has_disconnect_token:
        reasoning = (
            f"<think>\n"
            f"Node {node_a} is in the graph token: {graph_token_a_str}. "
            f"The graph token sequence only has one `{graph_vocab.GRAPH_DISCONNECT_TOKEN}` at the end, "
            f"so every token belongs to the same connected component. "
            f"Therefore all nodes, including {node_a} and {node_b}, share the same component. So the answer is Yes.\n"
            f"</think>\nThe answer is Yes."
        )
    elif predicted_connected:
        reasoning = (
            f"<think>\n"
            f"Node {node_a} is in the graph token: {graph_token_a_str}. "
            f"Since `{graph_vocab.GRAPH_DISCONNECT_TOKEN}` splits components, I will trace through all graph tokens in the same connected component as node {node_a}. "
            f"The tokens in this component are: "
            f"{connected_tokens_strs}. "
            f"Node {node_b} is found in the token {graph_token_b_str}, so both nodes lie in the same component and are connected. So the answer is Yes.\n"
            f"</think>\nThe answer is Yes."
        )
    else:
        reasoning = (
            f"<think>\n"
            f"Node {node_a} is in the graph token: {graph_token_a_str}. "
            f"Since `{graph_vocab.GRAPH_DISCONNECT_TOKEN}` splits components, I will trace through all graph tokens in the same connected component as node {node_a}. "
            f"The tokens in this component are: "
            f"{connected_tokens_strs}. "
            f"Node {node_b} seems not appear in this component, so they are disconnected. The answer is No.\n"
            f"</think>\nThe answer is No."
        )

    return reasoning


def convert_to_openai_format(samples, encoding_mode="GraphVocab", system_prompt=None):
    """
    Convert list of sample dicts to OpenAI fine-tuning format (list of dict with "messages").
    
    Args:
        samples: list of sample dicts
        encoding_mode: "GraphVocab", "EdgeList", or "Incident"
        system_prompt: optional system prompt
    """
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
        # Select graph text based on encoding mode
        if encoding_mode == "GraphVocab":
            graph_text = sample['graph_text_vocab']
            token_list = sample.get('token_list', None)
        elif encoding_mode == "Incident":
            graph_text = sample['graph_text_incident']
            token_list = None
        else:  # EdgeList
            graph_text = sample['graph_text_edge']
            token_list = None
        
        node_a = sample['node_a']
        node_b = sample['node_b']
        is_connected = sample['is_connected']
        
        answer = "Yes" if is_connected else "No"
        
        if args.CoT:
            if encoding_mode == "GraphVocab" and token_list:
                # Chain of thought reasoning using token analysis (GraphVocab only)
                assistant_msg = generate_cot_reasoning(token_list, node_a, node_b, is_connected)
            else:
                # For EdgeList and Incident, use simpler CoT reasoning
                if is_connected:
                    assistant_msg = (
                        f"<think>\n"
                        f"To check connectivity between node {node_a} and node {node_b}, "
                        f"I analyze the graph structure: {graph_text}. "
                        f"By examining the edges, I can trace a path from node {node_a} to node {node_b}. "
                        f"Therefore, there is a path between these nodes. "
                        f"So the answer is Yes.\n"
                        f"</think>\nThe answer is {answer}."
                    )
                else:
                    assistant_msg = (
                        f"<think>\n"
                        f"To check connectivity between node {node_a} and node {node_b}, "
                        f"I analyze the graph structure: {graph_text}. "
                        f"By examining the edges, I find that node {node_a} and node {node_b} belong to different connected components. "
                        f"Therefore, there is no path between these nodes. "
                        f"So the answer is No.\n"
                        f"</think>\nThe answer is {answer}."
                    )
        else:
            # Direct answer
            assistant_msg = f"The answer is {answer}."
        



        user_msg = (
            f"Given the following graph: {graph_text}. "
            f"Is node {node_a} and node {node_b} in a same connected component? Use yes or no to answer."
        )
        
        openai_sample = {
            "task": "connectivity",
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
    # Generate connectivity test samples (with both encodings)
    samples = generate_stage2_connectivity_samples(
        num_samples=args.num_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        num_splits=args.num_splits
    )
    
    print(f"\nGenerated {len(samples)} connectivity test samples")
    print(f"Connected: {sum(1 for s in samples if s['is_connected'])}")
    print(f"Disconnected: {sum(1 for s in samples if not s['is_connected'])}")
    
    # Convert to OpenAI format for both encodings
    openai_data_vocab = convert_to_openai_format(samples, encoding_mode="GraphVocab", system_prompt=None)

    
    # Save to files
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", "s2_connectivity") + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_suffix = "_CoT" if args.CoT == 1 else "_None"
    base_suffix += f"_Nodes-{args.min_nodes}-{args.max_nodes}"
    base_suffix += f"_Samples-{args.num_samples}"
    base_suffix += f"_Splits-{args.num_splits}"
    base_suffix += "_" + args.split.capitalize()
    
    file_name_vocab = f"GraphVocab_Stage2_Connectivity{base_suffix}.jsonl"
    file_name_edge = f"EdgeList_Stage2_Connectivity{base_suffix}.jsonl"
    file_name_incident = f"Incident_Stage2_Connectivity{base_suffix}.jsonl"
    
    save_to_jsonl(openai_data_vocab, save_path + file_name_vocab)
    openai_data_edge = convert_to_openai_format(samples, encoding_mode="EdgeList", system_prompt=None)
    save_to_jsonl(openai_data_edge, save_path + file_name_edge)
    openai_data_incident = convert_to_openai_format(samples, encoding_mode="Incident", system_prompt=None)
    save_to_jsonl(openai_data_incident, save_path + file_name_incident)
    
    # Print a few example samples
    print("\n--- Example samples (GraphVocab) ---")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Graph (GraphVocab): {sample['graph_text_vocab']}")
        print(f"  Graph (EdgeList): {sample['graph_text_edge']}")
        print(f"  Nodes: {sample['node_a']} and {sample['node_b']}")
        print(f"  Connected: {sample['is_connected']}")



