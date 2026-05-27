import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import networkx as nx
from tqdm import tqdm
from graph_vocab.graph_tokenizer import GraphTokenizer
from utils.tools import mean_and_var

graph_tokenizer = GraphTokenizer()


def parse_graph_degree_user(user_msg: str):
    edge_match = re.search(
        r"(Nodes:\s*.*?Edges:\s*.*?)(?:\.\s*What is the degree of node|\?|$)",
        user_msg,
        re.IGNORECASE | re.DOTALL,
    )
    if edge_match:
        graph_text = edge_match.group(1).strip()
        encoding = "EdgeList"
    else:
        gv_match = re.search(
            r"Given the following graph:\s*(.*?)\.\s*What is the degree of node",
            user_msg,
            re.IGNORECASE | re.DOTALL,
        )
        if not gv_match:
            raise ValueError(f"Cannot parse graph text from user message: {user_msg}")
        graph_text = gv_match.group(1).strip()
        # Determine encoding type
        if graph_text.startswith("This graph has nodes"):
            encoding = "Incident"
        else:
            encoding = "GraphVocab"

    node_match = re.search(
        r"What\s+is\s+the\s+degree\s+of\s+node\s+(\d+)",
        user_msg,
        re.IGNORECASE,
    )
    if not node_match:
        raise ValueError(f"Cannot parse node id from user message: {user_msg}")
    node_a = int(node_match.group(1))

    return graph_text, node_a, encoding


# def parse_token_degree_user(user_msg: str):
#     gv_match = re.search(
#         r"Given the following graph:\s*(.*?),\s*list the degree of each node",
#         user_msg,
#         re.IGNORECASE | re.DOTALL,
#     )
#     if not gv_match:
#         raise ValueError(f"Cannot parse graph text from token-degree user message: {user_msg}")
#     graph_text = gv_match.group(1).strip()
#     encoding = "GraphVocab"
#     return graph_text, encoding


def parse_graph_degree_answer(assistant_msg: str, node_a: int):
    pattern = rf"The\s+degree\s+of\s+node\s+{node_a}\s+is\s+(-?\d+)"
    # match = re.search(pattern, assistant_msg, re.IGNORECASE)
    # if not match:
    #     return None
    # matched = int(match.group(1))
    matches = re.findall(pattern, assistant_msg)
    last_match = int(matches[-1]) if matches else None
    return last_match


def reconstruct_graph(graph_text: str, encoding: str) -> nx.Graph:
    if encoding == "GraphVocab":
        return graph_tokenizer.decode_graph_vocab(graph_text)
    elif encoding == "EdgeList":
        return graph_tokenizer.decode_edge_list(graph_text)
    elif encoding == "Incident":
        return graph_tokenizer.decode_incident(graph_text)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def verify_graph_degree(graph_text: str, node_a: int, assistant_msg: str, encoding: str) -> bool:
    try:
        G = reconstruct_graph(graph_text, encoding)
    except Exception as e:
        print(f"Failed to reconstruct graph ({encoding}): {e}")
        return False

    pred = parse_graph_degree_answer(assistant_msg, node_a)
    if pred is None:
        return False

    if node_a not in G.nodes():
        truth = 0
    else:
        truth = int(G.degree[node_a])

    return pred == truth


# def detect_task_type(user_msg: str):
#     if "What is the degree of node" in user_msg:
#         return "Graph_Degree"
#     if "list the degree of each node" in user_msg:
#         return "Token_Degree"
#     return "Unknown"


def find_all_tokens_with_node(token_list, node_a):
    """
    Find all graph tokens in token_list that contain node_a.
    Returns indices and token info list.
    """
    from graph_vocab.graph_vocabulary import GraphVocabulary
    graph_vocab = GraphVocabulary()
    
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
            degree_a = int(G.degree[node_a])
            tokens_info.append(
                {
                    "index": i,
                    "token": token_text,
                    "node_ids": list(G.nodes()),
                    "degree_a": degree_a
                }
            )

    return indices, tokens_info

def parse_local_degree_from_assistant(assistant_msg: str, node_a: int, expected_tokens_info: list) -> float:
    """
    Parse assistant message to check if local degrees are correctly calculated.
    Returns the success ratio of correctly calculated local degrees.
    
    Matches patterns like:
    - "In subgraph <NidB>1<NidS>2<NidS>3<NidE><G3_triangle>, the degree of node X is Y"
    
    This function matches both the node IDs and the token type to ensure the exact
    subgraph is being referenced.
    """
    if not expected_tokens_info:
        return 0.0  # No tokens to check, consider as success
    
    success_count = 0
    total_count = len(expected_tokens_info)
    
    # Import graph vocabulary to get node ID tokens
    from graph_vocab.graph_vocabulary import GraphVocabulary
    graph_vocab = GraphVocabulary()
    
    # Pattern to match: "In subgraph <NidB>...<NidE><token> ... the degree of node X is Y"
    for token_info in expected_tokens_info:
        expected_degree = token_info["degree_a"]
        token_type = token_info["token"]  # e.g., "<G4_star>", "<G3_triangle>"
        node_ids = token_info["node_ids"]  # e.g., [1, 2, 3]
        
        # Construct the full token string with node IDs
        # Format: <NidB>1<NidS>2<NidS>3<NidE><G3_triangle>
        node_id_str = graph_vocab.NODE_ID_BEGIN_TOKEN
        for i, node_id in enumerate(node_ids):
            if i > 0:
                node_id_str += graph_vocab.NODE_ID_SPLIT_TOKEN
            node_id_str += str(node_id)
        node_id_str += graph_vocab.NODE_ID_END_TOKEN
        full_token_str = node_id_str + token_type
        
        # Escape special regex characters in the full token string
        escaped_full_token = re.escape(full_token_str)
        
        # Pattern 1: "In subgraph <NidB>...<NidE><token>, the degree of node {node_a} is {expected_degree}"
        # This matches the complete token with node IDs and the degree calculation
        pattern1 = rf"In\s+subgraph\s+{escaped_full_token}.*?degree\s+of\s+node\s+{node_a}\s+is\s+{expected_degree}"
        
        # Pattern 2: More flexible pattern that looks for the full token and degree in proximity
        # Matches: "<NidB>...<NidE><token> ... node {node_a} ... degree ... {expected_degree}"
        pattern2 = rf"{escaped_full_token}.*?node\s+{node_a}.*?degree.*?{expected_degree}"
        
        # Pattern 3: Reverse order - degree mentioned before the full token
        # Matches: "node {node_a} ... degree ... {expected_degree} ... <NidB>...<NidE><token>"
        pattern3 = rf"node\s+{node_a}.*?degree.*?{expected_degree}.*?{escaped_full_token}"
        
        if re.search(pattern1, assistant_msg, re.IGNORECASE | re.DOTALL) or \
           re.search(pattern2, assistant_msg, re.IGNORECASE | re.DOTALL) or \
           re.search(pattern3, assistant_msg, re.IGNORECASE | re.DOTALL):
            success_count += 1
    
    success_ratio = success_count / total_count if total_count > 0 else 0.0
    return success_ratio

def compute_reward_degree(user_msg: str, assistant_msg: str) -> float:
    """
    Compute reward for a single degree QA pair with step-by-step scoring.
    
    Reward = (success_ratio + is_correct) * 0.5
    where:
    - success_ratio: ratio of correctly calculated local degrees in tokens containing target node
    - is_correct: 1.0 if final answer is correct, 0.0 otherwise
    """
    try:
        graph_text, node_a, encoding = parse_graph_degree_user(user_msg)
        G = reconstruct_graph(graph_text, encoding)
        pred = parse_graph_degree_answer(assistant_msg, node_a)
        
        # Calculate is_correct
        if pred is None:
            is_correct = 0.0
        else:
            if node_a not in G.nodes():
                truth = 0
            else:
                truth = int(G.degree[node_a])
            is_correct = 1.0 if pred == truth else 0.0
        
        # Calculate success_ratio for GraphVocab encoding
        success_ratio = 0.0

        try:
            # Tokenize the graph to get token list
            token_list = graph_tokenizer.tokenize(G, strategy="greedy+wl")

            # Find all tokens containing node_a
            indices, tokens_info = find_all_tokens_with_node(token_list, node_a)

            # Parse assistant message to check local degree calculations
            if tokens_info:
                success_ratio = parse_local_degree_from_assistant(assistant_msg, node_a, tokens_info)
        except Exception as e:
            # If tokenization or parsing fails, success_ratio remains 0.0
            print(f"Warning: Failed to calculate success_ratio: {e}")
            success_ratio = 0.0

        
        # Final reward = (success_ratio + is_correct) * 0.5
        final_reward = (success_ratio + is_correct) * 0.5
        return final_reward
        
    except Exception as e:
        print(f"compute_reward_degree error: {e}")
        return 0.0

def evaluate_s2_degree(user_msg: str, assistant_msg: str) -> bool:
    """Verify a single degree QA pair. Returns True if assistant answer is correct."""
    try:
        graph_text, node_a, encoding = parse_graph_degree_user(user_msg)
        return verify_graph_degree(graph_text, node_a, assistant_msg, encoding)
    except Exception as e:
        print(f"evaluate_s2_degree error: {e}")
        return False


def load_and_evaluate_s2_degree(
    dataset_path: str,
    tokenizer_obj=None,
    num_splits: int = 1,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {dataset_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    total_samples = len(raw_lines)
    if total_samples == 0:
        print("No valid samples")
        return []

    if num_splits <= 0:
        num_splits = 1
    num_splits = min(num_splits, total_samples)

    print(f"Now evaluating s2 degree dataset: {dataset_path}")
    print(f"Total samples: {total_samples}, num_splits: {num_splits}")

    # 为每个 split 维护正确样本数
    split_correct = [0 for _ in range(num_splits)]
    split_count = [0 for _ in range(num_splits)]

    results: List[Dict[str, Any]] = []
    total_graph_tokens = 0
    total_assistant_tokens = 0
    tokenizer_sample_count = 0

    for idx, line in enumerate(tqdm(raw_lines)):
        try:
            data = json.loads(line)
        except Exception as e:
            print(f"Line {idx}: json load error -> {e}")
            continue

        messages = data.get("messages", [])
        user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]

        if user_msg is None or assistant_msg is None:
            print(f"Line {idx}: missing user or assistant message")
            continue

        graph_text = ""
        try:
            is_correct = evaluate_s2_degree(user_msg, assistant_msg)
            # reward = compute_reward_degree(user_msg, assistant_msg)
            # print(f"reward: {reward}")
        except Exception as e:
            print(f"Line {idx}: parsing/verification error -> {e}")
            is_correct = False

        split_idx = min(idx * num_splits // total_samples, num_splits - 1)
        split_count[split_idx] += 1
        if is_correct:
            split_correct[split_idx] += 1
        else:
            if verbose:
                print(f"\nBad case at line {idx}:")
                print(f"user_msg: {user_msg}")
                print(f"assistant_msg: {assistant_msg}")

        if tokenizer_obj is not None and graph_text:
            try:
                g_token_list = tokenizer_obj.tokenize(graph_text)
            except Exception as e:
                print(f"Tokenizer.tokenize failed for graph_text at line {idx}: {e}")
                g_token_list = []

            try:
                a_token_list = tokenizer_obj.tokenize(assistant_msg)
            except Exception as e:
                print(f"Tokenizer.tokenize failed for assistant_msg at line {idx}: {e}")
                a_token_list = []

            total_graph_tokens += len(g_token_list)
            total_assistant_tokens += len(a_token_list)
            tokenizer_sample_count += 1

        results.append(
            {
                "line": idx,
                "question": user_msg,
                "output": assistant_msg,
                "correct": is_correct,
            }
        )

    acc_list: List[float] = []
    print("\nPer-split accuracy:")
    for i in range(num_splits):
        cnt = split_count[i]
        if cnt == 0:
            acc = 0.0
        else:
            acc = split_correct[i] / cnt
        acc_list.append(acc)
        print(f"  Split {i}: count={cnt}, correct={split_correct[i]}, Accuracy={acc:.4f}")

    acc_mean, acc_var = mean_and_var(acc_list)
    print("\nOverall split-level statistics (Accuracy):")
    print(f"  mean={acc_mean:.4f}, var={acc_var:.6f}")

    if tokenizer_obj is not None and tokenizer_sample_count > 0:
        avg_graph_tokens = total_graph_tokens / tokenizer_sample_count
        avg_assistant_tokens = total_assistant_tokens / tokenizer_sample_count
        print("\nTokenization statistics:")
        print(f"  Samples tokenized: {tokenizer_sample_count}")
        print(f"  Average graph_text tokens per sample: {avg_graph_tokens:.2f}")
        print(f"  Average assistant_msg tokens per sample: {avg_assistant_tokens:.2f}")

    return results


if __name__ == "__main__":
    data_path = "../data/s2_degree/Incident_Stage2_Degree_CoT_Nodes-11-30_Samples-10_Splits-10_Test.jsonl"
    load_and_evaluate_s2_degree(data_path, num_splits=10)
