import re
import json
from collections import Counter
import argparse
from typing import Optional, List, Dict, Any
from utils.tools import compute_accuracy_f1, mean_and_var
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from graph_vocab.graph_tokenizer import GraphTokenizer
from graph_vocab.graph_vocabulary import GraphVocabulary
from transformers import AutoTokenizer

graph_tokenizer = GraphTokenizer()
graph_vocab = GraphVocabulary()


def parse_user_message(user_msg: str):
    """
    Extract the graph text and node ids (a, b) from the user message.
    Supports the new format from gen_data: "Given the following graph: {graph_text}. Is node {a} and node {b} ..."
    and keeps compatibility with legacy GraphVocab/EdgeList prompts.
    Supports three encoding formats: GraphVocab, EdgeList, and Incident.
    """
    # New default pattern
    pattern = re.search(
        r"Given the following graph:\s*(.*?)\.\s*Is node",
        user_msg,
        re.IGNORECASE | re.DOTALL,
    )
    graph_text = None
    encoding = None
    if pattern:
        graph_text = pattern.group(1).strip()
        # Determine encoding type
        if graph_text.startswith("This graph has nodes"):
            encoding = "Incident"
        elif re.search(r"Nodes:\s*.*?Edges:", graph_text, re.IGNORECASE | re.DOTALL):
            encoding = "EdgeList"
        else:
            encoding = "GraphVocab"

    if graph_text is None:
        graph_vocab_match = re.search(
            r"graph token sequence:\s*(.*?)\.\s*Is there a path between node",
            user_msg,
            re.IGNORECASE | re.DOTALL,
        )
        if graph_vocab_match:
            graph_text = graph_vocab_match.group(1).strip()
            encoding = "GraphVocab"
        else:
            edge_list_match = re.search(
                r"(Nodes:\s*.*?Edges:\s*.*?)(?:\.\s*Is there a path between node|\?|$)",
                user_msg,
                re.IGNORECASE | re.DOTALL,
            )
            if edge_list_match:
                graph_text = edge_list_match.group(1).strip()
                encoding = "EdgeList"

    if graph_text is None:
        raise ValueError(f"Cannot parse graph text from user message: {user_msg}")

    node_match = re.search(r"node\s+(\d+)\s+and\s+node\s+(\d+)", user_msg, re.IGNORECASE)
    if not node_match:
        raise ValueError(f"Cannot parse node ids from user message: {user_msg}")
    node_a = int(node_match.group(1))
    node_b = int(node_match.group(2))

    return graph_text, node_a, node_b, encoding if encoding else "GraphVocab"


def parse_assistant_answer(assistant_msg: str):
    """
    Extract final Yes/No from assistant message (may contain <think> ... </think>).
    """
    matches = re.findall(r"\b(Yes|No)\b", assistant_msg, flags=re.IGNORECASE)
    if not matches:
        return None
    # Return True if last match is "Yes", False otherwise
    return matches[-1].lower() == "yes"


def verify_connectivity(graph_text: str, node_a: int, node_b: int, assistant_msg: str, encoding: str = "GraphVocab"):
    """Reconstruct the graph according to `encoding` and compare assistant answer to ground truth."""
    # Reconstruct graph
    if encoding == "GraphVocab":
        G = graph_tokenizer.decode_graph_vocab(graph_text)
    elif encoding == "EdgeList":
        # Use graph_tokenizer's edge list decoder
        try:
            G = graph_tokenizer.decode_edge_list(graph_text)
        except Exception as e:
            # If decoding fails, treat as incorrect
            print(f"Failed to decode edge list: {e}")
            return False
    elif encoding == "Incident":
        # Use graph_tokenizer's incident decoder
        try:
            G = graph_tokenizer.decode_incident(graph_text)
        except Exception as e:
            # If decoding fails, treat as incorrect
            print(f"Failed to decode incident: {e}")
            return False
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    predicted = parse_assistant_answer(assistant_msg)
    if predicted is None:
        return True, False

    # Ground truth
    if node_a not in G.nodes() or node_b not in G.nodes():
        truth = False
    else:
        truth = nx.has_path(G, node_a, node_b)

    return truth, predicted



def load_and_evaluate_s2_connectivity(
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

    print(f"Now evaluating: {dataset_path}")
    print(f"Total samples: {total_samples}, num_splits: {num_splits}")

    # 为每个 split 维护 TP/FP/TN/FN
    split_stats = [
        {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "count": 0} for _ in range(num_splits)
    ]

    results: List[Dict[str, Any]] = []
    total_graph_tokens = 0
    total_assistant_tokens = 0
    tokenizer_sample_count = 0
    idx = 0
    for line in tqdm(raw_lines):
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
            print(f"Line {idx}: Missing user or assistant message")
            continue

        try:
            is_correct = evaluate_s2_connectivity(user_msg, assistant_msg)
            graph_text, node_a, node_b, encoding = parse_user_message(user_msg)
            truth, pred = verify_connectivity(graph_text, node_a, node_b, assistant_msg, encoding)
        except Exception as e:
            print(f"Line {idx}: parsing/verification error -> {e}")
            continue

        split_idx = min(int(idx - 1) * num_splits // total_samples, num_splits - 1)
        stat = split_stats[split_idx]
        stat["count"] += 1

        if truth and pred:
            stat["tp"] += 1
        elif (not truth) and pred:
            stat["fp"] += 1
        elif (not truth) and (not pred):
            stat["tn"] += 1
        elif truth and (not pred):
            stat["fn"] += 1

        is_correct = (truth == pred)

        if tokenizer_obj is not None:
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

        if not is_correct and verbose:
            print(f"Bad case at line {idx}:")
            print(f"user_msg: {user_msg}")
            print(f"assistant_msg: {assistant_msg}")

        results.append(
            {
                "line": idx,
                "question": user_msg,
                "output": assistant_msg,
                "correct": is_correct,
            }
        )
        idx += 1

    # 按 split 打印 Accuracy / F1，并收集列表
    acc_list: List[float] = []
    f1_list: List[float] = []

    print("\nPer-split metrics:")
    for i, stat in enumerate(split_stats):
        tp, fp, tn, fn, cnt = stat["tp"], stat["fp"], stat["tn"], stat["fn"], stat["count"]
        if cnt == 0:
            acc, f1 = 0.0, 0.0
        else:
            acc, f1 = compute_accuracy_f1(tp, fp, tn, fn)
        acc_list.append(acc)
        f1_list.append(f1)
        print(
            f"  Split {i}: "
            f"count={cnt}, TP={tp}, FP={fp}, TN={tn}, FN={fn}, "
            f"Accuracy={acc:.4f}, F1={f1:.4f}"
        )

    acc_mean, acc_var = mean_and_var(acc_list)
    f1_mean, f1_var = mean_and_var(f1_list)

    print("\nOverall split-level statistics:")
    print(f"  Accuracy mean={acc_mean:.4f}, var={acc_var:.6f}")
    print(f"  F1 mean={f1_mean:.4f}, var={f1_var:.6f}")

    total_tp = sum(s["tp"] for s in split_stats)
    total_fp = sum(s["fp"] for s in split_stats)
    total_tn = sum(s["tn"] for s in split_stats)
    total_fn = sum(s["fn"] for s in split_stats)
    overall_acc, overall_f1 = compute_accuracy_f1(total_tp, total_fp, total_tn, total_fn)
    print("\nGlobal metrics on all samples:")
    print(f"  Accuracy: {overall_acc:.4f}")
    print(f"  F1: {overall_f1:.4f}")

    if tokenizer_obj is not None and tokenizer_sample_count > 0:
        avg_graph_tokens = total_graph_tokens / tokenizer_sample_count
        avg_assistant_tokens = total_assistant_tokens / tokenizer_sample_count
        print("\nTokenization statistics:")
        print(f"  Samples tokenized: {tokenizer_sample_count}")
        print(f"  Average graph_text tokens per sample: {avg_graph_tokens:.2f}")
        print(f"  Average assistant_msg tokens per sample: {avg_assistant_tokens:.2f}")

    return results

def compute_reward_connectivity(user_msg: str, assistant_msg: str) -> float:
    """
    Compute reward for a single connectivity QA pair.
    
    Reward structure:
    - Base reward: 1.0 if answer is correct, 0.0 otherwise
    - Bonus reward: +0.25 if <G_Connect> or <G_Disconnect> tokens are used correctly in reasoning
    
    Returns:
        float: Reward score (0.0 to 1.5)
    """
    try:
        graph_text, node_a, node_b, encoding = parse_user_message(user_msg)
        truth, pred = verify_connectivity(graph_text, node_a, node_b, assistant_msg, encoding)
        if pred is None:
            return 0.0
        
        # Base reward: correctness of the answer
        base_reward = 0.80 if truth == pred else 0.0
        
        # Bonus reward: check if <G_Connect> or <G_Disconnect> tokens are used
        bonus_reward = 0.0

        # Check if assistant message contains <G_Connect> or <G_Disconnect>
        has_connect = graph_vocab.GRAPH_CONNECT_TOKEN in assistant_msg
        has_disconnect = graph_vocab.GRAPH_DISCONNECT_TOKEN in assistant_msg

        if has_connect or has_disconnect:
            # Give bonus if these tokens are used (indicates reasoning about connected components)
            bonus_reward = 0.20
        
        return base_reward + bonus_reward
        
    except Exception as e:
        print(f"compute_reward_connectivity error: {e}")
        return 0.0

def evaluate_s2_connectivity(user_msg: str, assistant_msg: str) -> bool:
    """Verify a single connectivity QA pair. Returns True if assistant answer is correct."""
    try:
        graph_text, node_a, node_b, encoding = parse_user_message(user_msg)
        truth, pred = verify_connectivity(graph_text, node_a, node_b, assistant_msg, encoding)
        if truth is None or pred is None:
            return False
        return truth == pred
    except Exception as e:
        print(f"evaluate_s2_connectivity error: {e}")
        return False

if __name__ == "__main__":
    data_path = "../data/s2_connectivity/Incident_Stage2_Connectivity_CoT_Nodes-11-30_Samples-10_Splits-10_Test.jsonl"
    load_and_evaluate_s2_connectivity(data_path, num_splits=10)