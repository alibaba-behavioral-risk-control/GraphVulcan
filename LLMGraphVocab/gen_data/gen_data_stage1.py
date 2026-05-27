import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import itertools
import networkx as nx
import json
import random
import argparse
from itertools import product
from tqdm import tqdm
from graph_vocab.graph_vocabulary import GraphVocabulary
graph_vocab = GraphVocabulary()


parser = argparse.ArgumentParser()
parser.add_argument("--CoT", type=int, default=1, help="Whether Enable CoT in train data, 0=disable, 1=enable")
parser.add_argument("--num_relabel", type=int, default=1, help="Number Relabeling")
parser.add_argument("--max_nodes", type=int, default=5, help="Maximum number of nodes per graph token")
parser.add_argument("--num_random_graphs", type=int, default=0, help="Number of random graphs")
parser.add_argument("--split", type=str, default="test", help="train/test")
args = parser.parse_args()


def random_relabel_graph_token(root_nodes, leaf_nodes_1, leaf_nodes_2, root_token, leaf_token_1, leaf_token_2):
    n_nodes = len(root_nodes)
    random_labels = []
    for i in range(n_nodes):
        while True:
            digit = random.randint(1, 3)
            max_label = 10 ** digit - 1
            min_label = 0 if digit == 1 else 10 ** (digit - 1)
            random_label = random.sample(range(min_label, max_label), 1)
            if random_label[0] not in random_labels:
                # loop until find a unique label
                random_labels.append(random_label[0])
                break
    node_to_label = {old: str(new) for old, new in zip(root_nodes, random_labels)}
    big_node_labels = [node_to_label[n] for n in root_nodes]
    root_str = '<NidB>' + '<NidS>'.join(big_node_labels) + '<NidE>' + root_token
    labels1 = [node_to_label[n] for n in leaf_nodes_1]
    labels2 = [node_to_label[n] for n in leaf_nodes_2]
    leaf_str_1 = '<NidB>' + '<NidS>'.join(labels1) + '<NidE>' + leaf_token_1
    leaf_str_2 = '<NidB>' + '<NidS>'.join(labels2) + '<NidE>' + leaf_token_2
    return root_str, leaf_str_1, leaf_str_2


def random_relabel_graph_token_multi(root_nodes, leaf_nodes_list, root_token, leaf_tokens_list):
    """Randomly relabel a root graph and multiple leaf graphs using a shared node-id mapping.

    Returns:
        root_str, leaf_str_list (same shapes as inputs, but string-encoded).
    """
    n_nodes = len(root_nodes)
    random_labels = []
    for _ in range(n_nodes):
        digit = random.randint(1, 3)
        max_label = 10 ** digit - 1
        min_label = 0 if digit == 1 else 10 ** (digit - 1)
        while True:
            random_label = random.sample(range(min_label, max_label), 1)[0]
            if random_label not in random_labels:
                random_labels.append(random_label)
                break
    node_to_label = {old: str(new) for old, new in zip(root_nodes, random_labels)}
    big_node_labels = [node_to_label[n] for n in root_nodes]
    root_str = graph_vocab.NODE_ID_BEGIN_TOKEN + graph_vocab.NODE_ID_SPLIT_TOKEN.join(big_node_labels) + graph_vocab.NODE_ID_END_TOKEN + root_token

    leaf_str_list = []
    for leaf_nodes, leaf_token in zip(leaf_nodes_list, leaf_tokens_list):
        labels = [node_to_label[n] for n in leaf_nodes if n in node_to_label]
        leaf_str = graph_vocab.NODE_ID_BEGIN_TOKEN + graph_vocab.NODE_ID_SPLIT_TOKEN.join(labels) + graph_vocab.NODE_ID_END_TOKEN + leaf_token
        leaf_str_list.append(leaf_str)
    return root_str, leaf_str_list


def _build_root_key(root_dict):
    """Build a hashable key for a root/leaf instance using token, node_ids and wl_hash."""
    return (
        root_dict["token"],
        tuple(root_dict["node_ids"]),
        root_dict["wl_hash"],
    )


def _bfs_decompose_full(start_root, root_index):
    """BFS from a root instance over decomposition tree.

    Instead of only returning final leaves, we record each decomposition step.

    Returns:
        steps: list of dicts, each with keys:
            - "root": a root instance dict
            - "leaf": list of 2 leaf instance dicts
        final_leaves: list of leaf instance dicts that cannot be further decomposed.
    """
    from collections import deque

    queue = deque([start_root])
    steps = []
    final_leaves = []
    visited = set()

    while queue:
        cur = queue.popleft()
        cur_key = _build_root_key(cur)
        if cur_key in visited:
            continue
        visited.add(cur_key)

        children = root_index.get(cur_key)
        if not children:
            # terminal graph instance
            final_leaves.append(cur)
            continue

        for tree_item in children:
            # record this decomposition step
            steps.append({"root": tree_item["root"], "leaf": tree_item["leaf"]})
            for child in tree_item["leaf"]:
                queue.append(child)

    return steps, final_leaves


def _bfs_levels_decompose_full(start_root, root_index):
    """BFS from a root instance over decomposition tree, recording one chosen decomposition per node per level.

    Returns:
        levels: list of list, where each inner list holds the chosen decomposition steps at that BFS depth.
                Each step is a dict: {"root": root_instance_dict, "leaf": [leaf0_instance, leaf1_instance]}.
    """
    from collections import deque

    queue = deque([start_root])
    visited = set()
    levels = []

    while queue:
        level_size = len(queue)
        level_steps = []
        for _ in range(level_size):
            cur = queue.popleft()
            cur_key = _build_root_key(cur)
            if cur_key in visited:
                continue
            visited.add(cur_key)

            children = root_index.get(cur_key)
            if not children:
                # terminal node: no further decomposition; nothing to enqueue
                continue

            # pick a single decomposition option for this node (instead of enumerating all)
            tree_item = random.choice(children)
            step = {"root": tree_item["root"], "leaf": tree_item["leaf"]}
            level_steps.append(step)
            for child in tree_item["leaf"]:
                queue.append(child)
        if level_steps:
            levels.append(level_steps)
    return levels


def _make_node_label_map(root_nodes):
    """Create a deterministic random mapping for node relabeling based on root_nodes."""
    n_nodes = len(root_nodes)
    random_labels = []
    for _ in range(n_nodes):
        digit = random.randint(1, 3)
        max_label = 10 ** digit - 1
        min_label = 0 if digit == 1 else 10 ** (digit - 1)
        while True:
            random_label = random.sample(range(min_label, max_label), 1)[0]
            if random_label not in random_labels:
                random_labels.append(random_label)
                break
    return {old: str(new) for old, new in zip(root_nodes, random_labels)}


def _encode_graph_token(node_ids, token, node_to_label):
    labels = [node_to_label[n] for n in node_ids if n in node_to_label]
    return graph_vocab.NODE_ID_BEGIN_TOKEN + graph_vocab.NODE_ID_SPLIT_TOKEN.join(labels) + graph_vocab.NODE_ID_END_TOKEN + token


def enumerate_all_token_partitions(num_relabels=1, list_isomorphic_mappings=False, max_k=5):
    decompose_samples = []
    merge_samples = []
    decompose_full_samples = []
    g0_candidates = [name for name in graph_vocab.GRAPH_STR_TOKENS if len(graph_vocab.GRAPH_VOCAB[name].edges()) >= 2]
    tree = []
    root_index = {}
    # seen_tree_items = set()
    for g0_name in tqdm(g0_candidates, desc="Building semantic relation tree"):
        g_0 = graph_vocab.GRAPH_VOCAB[g0_name]
        if len(g_0.nodes()) > max_k:
            continue
        edges = list(g_0.edges())
        m = len(edges)
        t0, m0 = graph_vocab.wl_hash_match_with_mapping(g_0, list_all_mappings=list_isomorphic_mappings)
        for r in range(1, m):
            for edge_subset in itertools.combinations(edges, r):
                E1 = set(edge_subset)
                E2 = set(edges) - E1
                if not E1 or not E2:
                    continue

                G1 = nx.Graph()
                G1.add_edges_from(E1)
                G2 = nx.Graph()
                G2.add_edges_from(E2)
                if not (nx.is_connected(G1) and nx.is_connected(G2)):
                    # Verify Graphlet
                    continue
                leaf_token_0, leaf_node_mappings_0 = graph_vocab.wl_hash_match_with_mapping(G1, list_all_mappings=list_isomorphic_mappings)
                leaf_token_1, leaf_node_mappings_1 = graph_vocab.wl_hash_match_with_mapping(G2, list_all_mappings=list_isomorphic_mappings)
                if leaf_token_0 not in graph_vocab.GRAPH_STR_TOKENS or leaf_token_1 not in graph_vocab.GRAPH_STR_TOKENS:
                    continue

                for item in product(m0, leaf_node_mappings_0, leaf_node_mappings_1):
                    root_node_ids, leaf_node_ids_0, leaf_node_ids_1 = item
                    # calculate WL hash
                    root_graph = graph_vocab.instantiate_graph_from_token(t0, root_node_ids)
                    root_wl_hash = graph_vocab.calculate_weisfeiler_lehman_graph_hash(root_graph)
                    leaf_graph_0 = graph_vocab.instantiate_graph_from_token(leaf_token_0, leaf_node_ids_0)
                    leaf_wl_hash_0 = graph_vocab.calculate_weisfeiler_lehman_graph_hash(leaf_graph_0)
                    leaf_graph_1 = graph_vocab.instantiate_graph_from_token(leaf_token_1, leaf_node_ids_1)
                    leaf_wl_hash_1 = graph_vocab.calculate_weisfeiler_lehman_graph_hash(leaf_graph_1)
                    # Verify the graphs match
                    root = {"node_ids": root_node_ids, "token": t0, "wl_hash": root_wl_hash}
                    leaf = [
                        {"node_ids": leaf_node_ids_0, "token": leaf_token_0, "wl_hash": leaf_wl_hash_0},
                        {"node_ids": leaf_node_ids_1, "token": leaf_token_1, "wl_hash": leaf_wl_hash_1}
                    ]
                    # tree_hash = (root_wl_hash, leaf_wl_hash_0, leaf_wl_hash_1)
                    # if tree_hash in seen_tree_items:
                    #     continue
                    # else:
                    #     seen_tree_items.add(tree_hash)
                    tree_item = {"root": root, "leaf": leaf, "hashes": (root_wl_hash, leaf_wl_hash_0, leaf_wl_hash_1)}
                    tree.append(tree_item)

                    root_key = _build_root_key(root)
                    root_index.setdefault(root_key, []).append(tree_item)

    for _ in tqdm(range(num_relabels), desc="Generating Decompose and Merge samples"):
        for item in tree:
            root = item['root']
            leaf_0 = item['leaf'][0]
            leaf_1 = item['leaf'][1]
            root_str, leaf_str_0, leaf_str_1 = random_relabel_graph_token(
                root['node_ids'],
                leaf_0['node_ids'],
                leaf_1['node_ids'],
                root['token'],
                leaf_0['token'],
                leaf_1['token']
            )
            decompose_samples.append(
                {
                    "task": "Decompose",
                    "input": root_str,
                    "output": [leaf_str_0, leaf_str_1]
                }
            )

            root_str, leaf_str_0, leaf_str_1 = random_relabel_graph_token(
                root['node_ids'],
                leaf_0['node_ids'],
                leaf_1['node_ids'],
                root['token'],
                leaf_0['token'],
                leaf_1['token']
            )
            merge_samples.append(
                {
                    "task": "Merge",
                    "input": [leaf_str_0, leaf_str_1],
                    "output": root_str
                }
            )
    decompose_full_samples = []
    return decompose_samples, merge_samples, decompose_full_samples



def generate_stage1_decomp_merge_samples(num_relabels=1, max_k=5):
    decompose_samples, merge_samples, decompose_full_samples = enumerate_all_token_partitions(num_relabels=num_relabels, max_k=max_k)

    all_samples = decompose_samples + merge_samples + decompose_full_samples
    print(f"Decompose: {len(decompose_samples)}")
    print(f"Merge: {len(merge_samples)}")
    print(f"DecomposeFull: {len(decompose_full_samples)}")
    return all_samples


def convert_to_openai_format(samples):
    """
    Convert list of sample dicts to OpenAI fine-tuning format (list of dict with "messages").

    Args:
        samples: list of dicts from decompose_task(), merge_task(), or difference_task()
        system_prompt: optional system message

    Returns:
        list of dicts in OpenAI format
    """
    print("Converting to OpenAI format...")
    system_prompt = (
            "You are a graph reasoning assistant. " +
            "The following are graph tokens: {" + ", ".join(f"{t}" for t in graph_vocab.GRAPH_STR_TOKENS) + "}. "
            f"Each graph token represents a connected subgraph. Node IDs precede the token in this format: "
            f"{graph_vocab.NODE_ID_BEGIN_TOKEN}1{graph_vocab.NODE_ID_SPLIT_TOKEN}2{graph_vocab.NODE_ID_END_TOKEN}<graph_token>. "
            f"where {graph_vocab.NODE_ID_SPLIT_TOKEN} separates node IDs, and {graph_vocab.NODE_ID_BEGIN_TOKEN}/{graph_vocab.NODE_ID_END_TOKEN} mark the start/end of node IDs. "
            "The order of node IDs reflects their relative positions within the graph. "
            "Operators: "
            f"{graph_vocab.GRAPH_OP_EQ_TOKEN}: Indicates the graphs on both sides are identical in structure and node IDs. "
            f"{graph_vocab.GRAPH_CONNECT_TOKEN}: The graphs on both sides belong to the same connected component. "
            f"{graph_vocab.GRAPH_DISCONNECT_TOKEN}: Marks that the left and right graphs belong to separate connected components."
            "Given graph token sequence, perform the requested reasoning task. Think step by step."
    )
    openai_data = []
    for sample in tqdm(samples):
        task = sample['task']
        input = sample['input']
        output = sample['output']

        if task == 'Decompose':
            if args.CoT:
                user_msg = f"Find an equivalent graph token sequence of {input}."
                assistant_msg = (
                        f'<think>\n{input} {graph_vocab.GRAPH_OP_EQ_TOKEN} '
                        + f' {graph_vocab.GRAPH_CONNECT_TOKEN} '.join(f"{t}" for t in output) + '.\n'
                        'So the equivalent graph token sequence is ' + f' {graph_vocab.GRAPH_CONNECT_TOKEN} '.join(f"{t}" for t in output) + '.'
                        '\n</think>\nThe equivalent graph token sequence is ' + f' {graph_vocab.GRAPH_CONNECT_TOKEN} '.join(f"{t}" for t in output) + '.'
                )
            else:
                user_msg = f"Decompose the graph {input} into two smaller graphs."
                assistant_msg = 'The decomposed graphs are ' + f' {graph_vocab.GRAPH_OP_EQ_TOKEN} ' + f' {graph_vocab.GRAPH_CONNECT_TOKEN} '.join(
                    f"{t}" for t in output)

        elif task == 'Merge':
            # input is a list of tokens
            if args.CoT:
                tokens_str = f' {graph_vocab.GRAPH_CONNECT_TOKEN} '.join(f"{t}" for t in input)
                user_msg = f"Find an equivalent graph token sequence of {tokens_str}."
                assistant_msg = (
                        f'<think>\n' + f' {graph_vocab.GRAPH_CONNECT_TOKEN} '.join(f"{t}" for t in input) + f' {graph_vocab.GRAPH_OP_EQ_TOKEN} '
                        + f'{output}.\n' + f'So the equivalent graph token sequence is {output}.'
                        f'\n</think>\nThe equivalent graph token sequence is {output}.'
                )
            else:
                tokens_str = f' {graph_vocab.GRAPH_CONNECT_TOKEN} '.join(f"{t}" for t in input)
                user_msg = f"Find an equivalent graph token sequence of {tokens_str}."
                assistant_msg = f'The equivalent graph token sequence is {output}.'

        else:
            continue


        openai_sample = {
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
    data = generate_stage1_decomp_merge_samples(num_relabels=args.num_relabel, max_k=args.max_nodes)
    for i, sample in enumerate(data):
        print(sample)
    openai_data = convert_to_openai_format(data)
    print("[OpenAI format data]")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", "s1_decomp_merge_convert") + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = "GraphVocab_Stage1_DMC"
    file_name += "_Relabels-" + str(args.num_relabel)
    file_name += f"_MaxNodes-{args.max_nodes}"
    # file_name += f"_RandomGraphs-{args.num_random_graphs}"
    file_name += f"_{args.split.capitalize()}"
    file_name += ".jsonl"

    save_to_jsonl(openai_data, save_path + file_name)