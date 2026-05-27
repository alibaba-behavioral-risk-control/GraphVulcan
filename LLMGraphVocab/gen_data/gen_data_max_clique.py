import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
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
parser.add_argument("--num_workers", type=int, default=15, help="Threads to use for sample generation")
args = parser.parse_args()

graph_vocab = GraphVocabulary()
tokenizer = GraphTokenizer()

def find_maximum_clique(G):
    """
    Find the maximum clique in graph G.
    Returns the set of nodes forming the maximum clique.
    """
    # Use networkx's max_clique to find maximum clique
    max_clique = max(nx.find_cliques(G), key=len)
    max_clique = set(max_clique)
    return max_clique

def get_raw_reasoning_path(G_token_list, max_clique_nodes):
    """
    Generate reasoning path for maximum clique problem.
    
    Logic:
    1. Determine which nodes form the maximum clique
    2. Identify which tokens cover these nodes
    3. For each covered token, convert to graph instance G_tok
    4. Separate G_clique and G_residual from G_tok
    5. Convert to reasoning expression: G_tok = G_clique + G_residual
    """
    operations = []
    max_clique_nodes = set(max_clique_nodes)
    
    # Find all tokens that contain at least one node from the max clique
    covered_tokens = []
    for token in G_token_list:
        if token["token"] not in graph_vocab.GRAPH_STR_TOKENS:
            continue
        token_nodes = set(token["node_ids"])
        # Check if this token contains any clique nodes
        len_overlap = len(token_nodes & max_clique_nodes)
        if len_overlap > 1:
            covered_tokens.append(token)
    
    # Process each covered token
    for token_item in covered_tokens:
        token_nodes = token_item["node_ids"]
        G_tok = graph_vocab.instantiate_graph_from_token(token_item["token"], token_nodes)
        
        # Find clique nodes within this token
        clique_nodes_in_token = [n for n in token_nodes if n in max_clique_nodes]
        
        if not clique_nodes_in_token:
            continue
        
        # Build G_clique: subgraph of clique nodes in this token
        G_clique = G_tok.subgraph(clique_nodes_in_token).copy()
        
        # Build G_residual: G_tok minus clique edges
        G_residual = G_tok.copy()
        G_residual.remove_edges_from(G_clique.edges())
        # Remove isolated nodes that were only in the clique
        isolated_nodes = [n for n in clique_nodes_in_token if G_residual.degree(n) == 0]
        G_residual.remove_nodes_from(isolated_nodes)
        
        operations.append({
            "operation": "decompose",
            "token": token_item["token"],
            "token_nodes": token_nodes,
            "clique_nodes": clique_nodes_in_token,
            "G_tok": G_tok,
            "G_clique": G_clique,
            "G_residual": G_residual
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
    """Build one max clique sample. Args: (num_nodes, build_reasoning)"""
    num_nodes, build_reasoning = args_tuple
    local_vocab = GraphVocabulary()
    local_tokenizer = GraphTokenizer()

    def _local_build(max_attempts=64):
        for _ in range(max_attempts):
            # Generate random graph with reasonable edge density
            G = generate_erdos_renyi_graph(num_nodes, p=0.3)
            
            if G.number_of_nodes() < 3:
                continue
            
            # Ensure graph is connected
            if not nx.is_connected(G):
                continue
            
            G = random_relabel(G)
            
            # Find maximum clique
            max_clique_nodes = find_maximum_clique(G)
            
            # Ensure clique has at least 2 nodes
            if len(max_clique_nodes) < 2:
                continue
            
            # Tokenize the graph
            token_list = local_tokenizer.tokenize(G, strategy="greedy+wl")
            graph_text_vocab = local_tokenizer.token_list_to_text(token_list)
            graph_text_edge = local_tokenizer.encode_edge_list(G)
            graph_text_incident = local_tokenizer.encode_incident(G)
            
            # Generate reasoning path if needed
            raw_reasoning_path = get_raw_reasoning_path(token_list, max_clique_nodes) if build_reasoning else None
            
            # Sort nodes for consistent output
            answer_nodes = sorted(list(max_clique_nodes))
            
            return {
                "task": "MaxClique",
                "num_nodes": num_nodes,
                "graph_text_vocab": graph_text_vocab,
                "graph_text_edge": graph_text_edge,
                "graph_text_incident": graph_text_incident,
                "token_list": token_list,
                "raw_reasoning_path": raw_reasoning_path,
                "answer_nodes": answer_nodes,
                "clique_size": len(answer_nodes)
            }
        raise RuntimeError("Failed to build max clique sample after multiple attempts")

    return _local_build()

def generate_max_clique_samples(num_samples, min_nodes, max_nodes, num_splits=1, build_reasoning=True, num_workers=10):
    samples = []
    node_list = _compute_node_counts(num_samples, min_nodes, max_nodes)
    jobs = [(n, build_reasoning) for n in node_list]
    
    for split_id in range(num_splits):
        print(f"\nGenerating samples for split {split_id + 1}/{num_splits}\n")
        split_samples = []
        
        if num_workers <= 1:
            for job in tqdm(jobs, desc="Generating max clique samples"):
                sample = _worker_build_single_sample(job)
                split_samples.append(sample)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_job = {
                    executor.submit(_worker_build_single_sample, job): job
                    for job in jobs
                }
                for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Generating max clique samples"):
                    sample = future.result()
                    split_samples.append(sample)
        
        split_samples.sort(key=lambda s: s["num_nodes"])
        samples.extend(split_samples)
    
    return samples

def generate_cot_reasoning(sample):
    """
    Generate CoT reasoning for maximum clique problem.
    
    Logic:
    1. State the maximum clique nodes
    2. For each covered token, show decomposition:
       G_tok = G_clique + G_residual
    """
    answer_nodes = sample['answer_nodes']
    raw_reasoning_path = sample['raw_reasoning_path']
    
    reasoning = "<think>\n"
    reasoning += "To find the Maximum Clique, we can decompose graph tokens as follows:\n"
    
    # Process each operation in the reasoning path
    clique_list = []
    for step in raw_reasoning_path:
        if step["operation"] == "decompose":
            # Get the graphs
            G_tok = step["G_tok"]
            G_clique = step["G_clique"]
            G_residual = step["G_residual"]
            
            # Encode graphs to token text
            tok_text = tokenizer.encode_graph_vocab(G_tok, mark_connected_components=False)
            clique_text = tokenizer.encode_graph_vocab(G_clique, mark_connected_components=False) if G_clique.number_of_edges() > 0 else tokenizer.encode_graph_vocab(nx.Graph(G_clique), mark_connected_components=False)
            residual_text = tokenizer.encode_graph_vocab(G_residual, mark_connected_components=True, mark_last_disconnect=False) if G_residual.number_of_edges() > 0 else tokenizer.encode_graph_vocab(nx.Graph(G_residual), mark_connected_components=False)
            residual_text = residual_text.replace(graph_vocab.GRAPH_DISCONNECT_TOKEN, graph_vocab.GRAPH_CONNECT_TOKEN)
            clique_list.append(clique_text)
            # Generate reasoning expression
            reasoning += f"{tok_text} {graph_vocab.GRAPH_OP_EQ_TOKEN} {clique_text} {graph_vocab.GRAPH_CONNECT_TOKEN} {residual_text}.\n"
    reasoning += "No more tokens can be further decomposed to form a maximum clique. "
    reasoning += "We can see that the maximum clique is formed by the following token sequence:\n"
    reasoning += f"{f' {graph_vocab.GRAPH_CONNECT_TOKEN} '.join(clique_list)}\n"
    reasoning += f"Thus, the maximum clique consists of nodes: [{', '.join(str(x) for x in answer_nodes)}].\n"
    reasoning += "</think>\n"
    reasoning += f"The maximum clique is: [{', '.join(str(x) for x in answer_nodes)}]"
    
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

        answer_nodes = sample['answer_nodes']

        user_msg = []
        user_msg.append('You are required to solve the Maximum Clique problem. Your goal is to identify the largest complete subgraph (clique) in the given graph.')
        user_msg.append('You are given the following graph:')
        user_msg.append(graph_text)
        user_msg.append('Provide the indices of the nodes in the maximum clique in the following format: The maximum clique is: [Node indices].')
        user_msg.append('For example, if the maximum clique consists of nodes 1, 3, 5, you should answer: The maximum clique is: [1, 3, 5].')
        user_msg = '\n'.join(user_msg)

        if args.CoT:
            if encoding_mode == "GraphVocab" and sample.get('raw_reasoning_path'):
                assistant_msg = generate_cot_reasoning(sample)
            else:
                # EdgeList / Incident CoT
                clique_size = sample['clique_size']
                assistant_msg = (
                    f"\n"
                    f"The maximum clique is: [{', '.join(str(x) for x in answer_nodes)}]"
                )
        else:
            assistant_msg = f"The maximum clique is: [{', '.join(str(x) for x in answer_nodes)}]"

        openai_sample = {
            "task": "max_clique",
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
    samples = generate_max_clique_samples(
        num_samples=args.num_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        num_splits=args.num_splits,
        build_reasoning=bool(args.CoT),
        num_workers=args.num_workers,
    )
    
    print("\n--- Sample examples ---")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i + 1}:")
        print(f"  Num nodes: {sample['num_nodes']}")
        print(f"  Clique size: {sample['clique_size']}")
        print(f"  Answer nodes: {sample['answer_nodes']}")
    
    openai_data_vocab = convert_to_openai_format(samples, encoding_mode="GraphVocab")
    
    
    print("\n[OpenAI format data generated]")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", "s2_max_clique") + "/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_suffix = ""
    base_suffix += "_CoT" if args.CoT else "_None"
    base_suffix += f"_Nodes-{args.min_nodes}-{args.max_nodes}"
    base_suffix += f"_Samples-{args.num_samples}"
    base_suffix += f"_Splits-{args.num_splits}"
    base_suffix += f"_{args.split.capitalize()}"
    
    file_name_vocab = f"GraphVocab_Stage2_MaxClique{base_suffix}.jsonl"
    file_name_edge = f"EdgeList_Stage2_MaxClique{base_suffix}.jsonl"
    file_name_incident = f"Incident_Stage2_MaxClique{base_suffix}.jsonl"

    save_to_jsonl(openai_data_vocab, save_path + file_name_vocab)
    openai_data_edge = convert_to_openai_format(samples, encoding_mode="EdgeList")
    save_to_jsonl(openai_data_edge, save_path + file_name_edge)
    openai_data_incident = convert_to_openai_format(samples, encoding_mode="Incident", system_prompt=None)
    save_to_jsonl(openai_data_incident, save_path + file_name_incident)

    # Print detailed example samples
    print("\n--- Detailed example samples ---")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i + 1}:")
        print(f"  Graph (GraphVocab): {sample['graph_text_vocab']}")
        print(f"  Graph (EdgeList): {sample['graph_text_edge']}")
        print(f"  Maximum Clique Nodes: {sample['answer_nodes']}")
        print(f"  Clique Size: {sample['clique_size']}")
        if args.CoT and sample.get('raw_reasoning_path'):
            print(f"  Number of reasoning steps: {len(sample['raw_reasoning_path'])}")
