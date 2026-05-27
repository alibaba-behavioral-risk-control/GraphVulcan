import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import networkx as nx
from tqdm import tqdm

from graph_vocab.graph_tokenizer import GraphTokenizer
from utils.tools import mean_and_var
from utils.functional import verify_graph_computation_expressions

graph_tokenizer = GraphTokenizer()


def parse_shortest_path_user(user_msg: str):
    """Parse graph text, node_a, node_b, and encoding from user message."""
    edge_match = re.search(
        r"(Nodes:\s*.*?Edges:\s*.*?)(?:\.\s*What is the shortest path length between|\?|$)",
        user_msg,
        re.IGNORECASE | re.DOTALL,
    )
    if edge_match:
        graph_text = edge_match.group(1).strip()
        encoding = "EdgeList"
    else:
        gv_match = re.search(
            r"Given the following graph:\s*(.*?)\.\s*What is the shortest path length between",
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
        r"shortest\s+path\s+length\s+between\s+node\s+(\d+)\s+and\s+node\s+(\d+)",
        user_msg,
        re.IGNORECASE,
    )
    if not node_match:
        raise ValueError(f"Cannot parse node ids from user message: {user_msg}")
    node_a, node_b = int(node_match.group(1)), int(node_match.group(2))

    return graph_text, node_a, node_b, encoding


def parse_shortest_path_answer(assistant_msg: str, node_a: int, node_b: int):
    pattern = rf"shortest\s+path\s+length\s+between\s+node\s+{node_a}\s+and\s+node\s+{node_b}\s+is\s+(-?\d+)"
    matches = re.findall(pattern, assistant_msg, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def reconstruct_graph(graph_text: str, encoding: str) -> nx.Graph:
    if encoding == "GraphVocab":
        return graph_tokenizer.decode_graph_vocab(graph_text)
    elif encoding == "EdgeList":
        return graph_tokenizer.decode_edge_list(graph_text)
    elif encoding == "Incident":
        return graph_tokenizer.decode_incident(graph_text)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def verify_shortest_path(graph_text: str, node_a: int, node_b: int, assistant_msg: str, encoding: str) -> bool:
    try:
        G = reconstruct_graph(graph_text, encoding)
    except Exception as e:
        print(f"Failed to reconstruct graph ({encoding}): {e}")
        return False

    pred = parse_shortest_path_answer(assistant_msg, node_a, node_b)
    if pred is None:
        return False

    if node_a not in G.nodes() or node_b not in G.nodes():
        truth = -1
    else:
        try:
            truth = nx.shortest_path_length(G, source=node_a, target=node_b)
        except nx.NetworkXNoPath:
            truth = -1

    return pred == truth


def load_and_evaluate_s2_shortest_path(
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

    print(f"Now evaluating s2 shortest-path dataset: {dataset_path}")
    print(f"Total samples: {total_samples}, num_splits: {num_splits}")

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
            graph_text, node_a, node_b, encoding = parse_shortest_path_user(user_msg)
            is_correct = verify_shortest_path(graph_text, node_a, node_b, assistant_msg, encoding)
            # reward = compute_reward_shortest_path(user_msg, assistant_msg)
            # print(f"reward:{reward}")
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


def compute_reward_shortest_path(user_msg: str, assistant_msg: str) -> float:
    """Compute reward for a single shortest-path QA pair."""
    try:
        graph_text, node_a, node_b, encoding = parse_shortest_path_user(user_msg)
        G = reconstruct_graph(graph_text, encoding)
        pred = parse_shortest_path_answer(assistant_msg, node_a, node_b)
        
        if pred is None:
            return 0.0
        
        if node_a not in G.nodes() or node_b not in G.nodes():
            truth = -1
        else:
            try:
                truth = nx.shortest_path_length(G, source=node_a, target=node_b)
            except nx.NetworkXNoPath:
                truth = -1
        
        # Base reward: 0.80 for correct answer
        base_reward = 0.70 if pred == truth else 0.0
        
        # Bonus reward: verify graph computation expressions (only for GraphVocab encoding)
        bonus_reward = 0.0

        has_operator_eq = graph_tokenizer.graph_vocab.GRAPH_OP_EQ_TOKEN in assistant_msg
        if has_operator_eq:
            try:
                correct_expression_rate = verify_graph_computation_expressions(assistant_msg, no_eq_reward=0.5)
                bonus_reward = 0.30 * correct_expression_rate
            except Exception:
                # If verification fails, no bonus
                bonus_reward = 0.0
        
        return base_reward + bonus_reward
    except Exception as e:
        print(f"compute_reward_shortest_path error: {e}")
        return 0.0

def evaluate_s2_shortest_path(user_msg: str, assistant_msg: str) -> bool:
    """Verify a single shortest-path QA pair. Returns True if assistant answer is correct."""
    try:
        graph_text, node_a, node_b, encoding = parse_shortest_path_user(user_msg)
        return verify_shortest_path(graph_text, node_a, node_b, assistant_msg, encoding)
    except Exception as e:
        print(f"evaluate_s2_shortest_path error: {e}")
        return False


if __name__ == "__main__":
    data_path = "../data/s2_shortest_path/Incident_Stage2_ShortestPath_CoT_Nodes-11-30_Samples-100_Splits-1_Test.jsonl"
    load_and_evaluate_s2_shortest_path(data_path, num_splits=10)
