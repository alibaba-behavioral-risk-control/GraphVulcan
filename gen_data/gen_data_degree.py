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
parser.add_argument("--num_samples", type=int, default=10000, help="Number of training samples to generate")
parser.add_argument("--num_relabels", type=int, default=0, help="Number of relabels for each graph token in the vocabulary")
parser.add_argument("--min_nodes", type=int, default=11, help="Minimum number of nodes in generated graphs")
parser.add_argument("--max_nodes", type=int, default=50, help="Maximum number of nodes in generated graphs")
parser.add_argument("--split", type=str, default="train", help="train/test")
parser.add_argument("--num_splits", type=int, default=2, help="Number of data splits")
args = parser.parse_args()

graph_vocab = GraphVocabulary()
tokenizer = GraphTokenizer()


def generate_stage2_degree_samples(num_samples: int, min_nodes: int, max_nodes: int, num_relabels: int = 1, num_splits: int = 1):
    """
    Generate stage 2 degree samples, including token-wise degree samples from graph vocabulary
    and random graph degree samples.
    Args:
        num_samples:
        min_nodes:
        max_nodes:
        num_relabels:

    Returns:
        List of samples.
    """
    samples = []
    for split_id in range(num_splits):
        print(f"\nGenerating samples for split {split_id + 1}/{num_splits}\n")
        print(f"Generating token-wise degree samples with relabels: {num_relabels}")
        for G in tqdm(graph_vocab.GRAPH_VOCAB.values()):
            for _ in range(num_relabels):
                G_relabel = random_relabel(G)
                edge_list = list(G_relabel.edges())
                degrees = dict(G_relabel.degree())
                graph_text_vocab = tokenizer.encode_graph_vocab(G_relabel, mark_connected_components=False)
                graph_text_incident = tokenizer.encode_incident(G_relabel)
                sample = {
                    "task": "Token_Degree",
                    "graph_text_vocab": graph_text_vocab,
                    "graph_text_edge": tokenizer.encode_edge_list(G_relabel),
                    "graph_text_incident": graph_text_incident,
                    "edge_list": edge_list,
                    "degrees": degrees,
                }
                samples.append(sample)

        step = (max_nodes - min_nodes + 1) / max(1, (num_samples))
        num_node_list = [int(min_nodes + i * step) for i in range(num_samples)]

        print(f"Generating {num_samples} degree calculation samples...")

        for i in tqdm(range(num_samples)):
            num_nodes = num_node_list[i]
            # G = generate_nm_random_graph(num_nodes)
            G = generate_erdos_renyi_graph(num_nodes, p=0.3)
            G = random_relabel(G)

            if G.number_of_nodes() == 0:
                continue

            token_list = tokenizer.tokenize(G, strategy="greedy+wl")
            # calculate tokenization time
            graph_text_vocab = tokenizer.token_list_to_text(token_list)
            graph_text_edge = tokenizer.encode_edge_list(G)
            graph_text_incident = tokenizer.encode_incident(G)
            node_a = random.choice(list(G.nodes()))
            degree_a = int(G.degree[node_a])

            sample = {
                "task": "Graph_Degree",
                "num_nodes": num_nodes,
                "graph_text_vocab": graph_text_vocab,
                "graph_text_edge": graph_text_edge,
                "graph_text_incident": graph_text_incident,
                "token_list": token_list,
                "node_a": node_a,
                "degree": degree_a,
            }
            samples.append(sample)

    return samples


def find_all_tokens_with_node(token_list, node_a):
    """
    find all graph tokens in token_list that contain node_a
    return indices and token info list
    """
    indices = []
    tokens_info = []

    for i, token in enumerate(token_list):
        token_text = token["token"]
        if token_text in [
            graph_vocab.GRAPH_CONNECT_TOKEN,
            graph_vocab.GRAPH_DISCONNECT_TOKEN,
        ]:
            continue
        elif node_a in token["node_ids"]:
            G = graph_vocab.instantiate_graph_from_token(token["token"], token["node_ids"])
            indices.append(i)
            degree_a = nx.degree(G, node_a)
            tokens_info.append(
                {
                    "index": i,
                    "token": token_text,
                    "node_ids": list(G.nodes()),
                    "degree_a": degree_a
                }
            )

    return indices, tokens_info


def generate_cot_reasoning_degree(token_list, node_a, true_degree):
    idx_list, tokens_info = find_all_tokens_with_node(token_list, node_a)

    if not idx_list:
        return (
            f"<think>\n"
            f"I did not find any graph tokens containing node {node_a} in the graph's token sequence, "
            f"so the degree of {node_a}.\n"
            f"</think>\nThe degree of node {node_a} is 0."
        )

    token_str_list = []
    relevant_token_str = tokenizer.token_list_to_text(tokens_info)
    reasoning = ""
    reasoning += (
        f"<think>\n"
        f"I need to calculate the degree of node {node_a} in the entire graph.\n"
        f"First, I will search through the graph token sequence for all graph tokens that contain node {node_a}.\n"
        f"These graph tokens are: {relevant_token_str}.\n"
    )
    for info in tokens_info:
        token_str = tokenizer.token_list_to_text([info])
        reasoning += (
            f"In subgraph {token_str}, "
            f"the degree of node {node_a} is {info['degree_a']}.\n"
        )
    reasoning += (
        f"obtaining several local degrees.\n"
        f"Then, by summing up all these local degrees, I obtain the total degree of node {node_a} in the entire graph.\n"
        f"After calculation, the sum is {true_degree}.\n"
        f"The degree of node {node_a} is {true_degree}.\n" 
        f"</think>\n"
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
        if sample["task"] == "Graph_Degree":
            if encoding_mode == "GraphVocab":
                graph_text = sample["graph_text_vocab"]
                token_list = sample.get("token_list", None)
            elif encoding_mode == "Incident":
                graph_text = sample["graph_text_incident"]
                token_list = None
            else:
                graph_text = sample["graph_text_edge"]
                token_list = None

            node_a = sample["node_a"]
            degree_a = sample["degree"]

            # assistant message
            if args.CoT:
                if encoding_mode == "GraphVocab" and token_list:
                    assistant_msg = generate_cot_reasoning_degree(token_list, node_a, degree_a)
                    assistant_msg += f"The degree of node {node_a} is {degree_a}."
                else:
                    assistant_msg = (
                        f"<think>\n"
                        f"I observe the given graph: {graph_text}.\n"
                        f"By traversing all edges connected to node {node_a} and counting the number of its neighboring nodes, "
                        f"I can determine its degree in the graph.\n"
                        f"After counting, the degree of node {node_a} is {degree_a}.\n"
                        f"</think>\n"
                        f"The degree of node {node_a} is {degree_a}."
                    )
            else:
                assistant_msg = f"The degree of node {node_a} is {degree_a}."

            # user message
            user_msg = (
                f"Given the following graph: {graph_text}. "
                f"What is the degree of node {node_a} in this graph? Your answer should be in the format: 'The degree of node {node_a} is X.'"
            )
        elif sample["task"] == "Token_Degree":
            graph_text = sample["graph_text_vocab"]
            user_msg = (
                f"Given the following graph: {graph_text}, "
                f"list the degree of each node in this graph."
            )
            assistant_msg = (
                "<think>\n"
                "To find the degree of each node in the graph, I will examine the edge list and count the number of connections for each node.\n"
            )
            if not sample["edge_list"]:
                assistant_msg += "The graph has no edges, so all nodes have degree 0.\n"
                assistant_msg += "</think>\n"
                answer = ""
                for node in range(len(sample["degrees"])):
                    answer += f"Node {node} has degree 0.\n"
            elif len(sample["edge_list"]) == 1:
                assistant_msg += f"{graph_text} is an edge, so each of the two nodes has degree 1.\n"
                assistant_msg += f"So the answer is:\n"
                answer = ""
                degrees = sample["degrees"]
                for node, degree in degrees.items():
                    answer += f"Node {node} has degree {degree}.\n"
                assistant_msg += answer
                assistant_msg += "</think>\n"
            else:
                assistant_msg += f"{graph_text} has following edges: "
                for edge in sample["edge_list"]:
                    u, v = edge
                    assistant_msg += f"{graph_vocab.NODE_ID_BEGIN_TOKEN}{u}{graph_vocab.NODE_ID_SPLIT_TOKEN}{v}{graph_vocab.NODE_ID_END_TOKEN}<G2_edge> "
                assistant_msg += "\n"
                assistant_msg += "Now, I will count the degree for each node:\n"
                degrees = sample["degrees"]
                answer = ""
                for node, degree in degrees.items():
                    answer += f"The degree of node {node} is {degree}.\n"
                assistant_msg += answer
                assistant_msg += "</think>\n"
            assistant_msg += answer

        openai_sample = {
            "task": "degree",
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
    if args.split =="train" and args.num_samples >= 10000 and args.num_splits > 1:
        num_relabels = args.num_relabels
    else:
        num_relabels = 0

    samples = generate_stage2_degree_samples(
        num_relabels=num_relabels,
        num_samples=args.num_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        num_splits=args.num_splits
    )

    print(f"\nGenerated {len(samples)} degree samples")

    openai_data_vocab = convert_to_openai_format(
        samples, encoding_mode="GraphVocab", system_prompt=None
    )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", "s2_degree") + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_suffix = "_CoT" if args.CoT == 1 else "_None"
    base_suffix += f"_Nodes-{args.min_nodes}-{args.max_nodes}"
    base_suffix += f"_Samples-{args.num_samples}"
    base_suffix += f"_Splits-{args.num_splits}"
    base_suffix += "_" + args.split.capitalize()

    file_name_vocab = f"GraphVocab_Stage2_Degree{base_suffix}.jsonl"
    file_name_edge = f"EdgeList_Stage2_Degree{base_suffix}.jsonl"
    file_name_incident = f"Incident_Stage2_Degree{base_suffix}.jsonl"

    save_to_jsonl(openai_data_vocab, save_path + file_name_vocab)
    openai_data_edge = convert_to_openai_format(
        samples, encoding_mode="EdgeList", system_prompt=None
    )
    save_to_jsonl(openai_data_edge, save_path + file_name_edge)
    openai_data_incident = convert_to_openai_format(samples, encoding_mode="Incident", system_prompt=None)
    save_to_jsonl(openai_data_incident, save_path + file_name_incident)

    print("\n--- Example samples (Degree) ---")
    for i, sample in enumerate(samples[::len(samples)//5]):
        print(f"\nSample {i + 1}:")
        print(f"  Task: {sample['task']}")
        print(f"  Graph (GraphVocab): {sample['graph_text_vocab']}")
        print(f"  Graph (EdgeList): {sample['graph_text_edge']}")
        if sample['task'] == 'Graph_Degree':
            print(f"  Node: {sample['node_a']}")
            print(f"  Degree: {sample['degree']}")
        elif sample['task'] == 'Token_Degree':
            print(f"  Degrees: {sample['degrees']}")
