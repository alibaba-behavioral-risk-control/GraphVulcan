import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import networkx as nx
from tqdm import tqdm

from utils.tools import compute_accuracy_f1, mean_and_var
from utils.functional import verify_graph_computation_expressions
from graph_vocab.graph_tokenizer import GraphTokenizer

# Shared tokenizer with decode helpers
graph_tokenizer = GraphTokenizer()


def parse_user_message(user_msg: str):
    """Extract graph text and encoding from the user prompt.

    Patterns come from gen_data_cycle_dectection.convert_to_openai_format:
    "Given the following graph: {graph_text}. Does the graph contain a cycle? ..."
    graph_text can be GraphVocab tokens, EdgeList string, or Incident format.
    """
    match = re.search(r"Given the following graph:\s*(.*?)\.\s*Does the graph contain a cycle?", user_msg, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Cannot parse graph text from user message: {user_msg}")

    graph_text = match.group(1).strip()

    # Determine encoding type
    if graph_text.startswith("This graph has nodes"):
        encoding = "Incident"
    elif re.search(r"Nodes:\s*.*Edges:\s*", graph_text, re.IGNORECASE | re.DOTALL):
        encoding = "EdgeList"
    else:
        encoding = "GraphVocab"

    return graph_text, encoding


def parse_assistant_answer(assistant_msg: str) -> Optional[bool]:
    """Extract final Yes/No decision; look at the last Yes/No occurrence."""
    matches = re.findall(r"\b(Yes|No)\b", assistant_msg, flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].lower() == "yes"


def verify_cycle_detection(graph_text: str, assistant_msg: str, encoding: str = "GraphVocab"):
    """Rebuild graph and compare predicted vs ground truth cycle existence."""
    try:
        if encoding == "GraphVocab":
            G = graph_tokenizer.decode_graph_vocab(graph_text)
        elif encoding == "EdgeList":
            G = graph_tokenizer.decode_edge_list(graph_text)
        elif encoding == "Incident":
            G = graph_tokenizer.decode_incident(graph_text)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
    except Exception as e:
        # Decoding failure counts as incorrect
        print(f"Failed to decode graph: {e}")
        return False, False

    pred = parse_assistant_answer(assistant_msg)
    if pred is None:
        return False, False

    truth = len(nx.cycle_basis(G)) > 0
    return truth, pred


def compute_reward_cycle_detection(user_msg: str, assistant_msg: str) -> float:
    """
    Compute reward for a single cycle detection QA pair.
    
    Reward structure:
    - Base reward: 0.7 if answer is correct, 0.0 otherwise
    - Bonus reward: +0.3 * correct_expression_rate if <G_Operator_Eq> is used
    
    Returns:
        float: Reward score (0.0 to 1.0)
    """
    try:
        graph_text, encoding = parse_user_message(user_msg)
        truth, pred = verify_cycle_detection(graph_text, assistant_msg, encoding)
        if pred is None:
            return 0.0
        
        # Base reward: correctness of the answer
        base_reward = 0.7 if truth == pred else 0.0
        
        # Bonus reward: verify graph computation expressions
        bonus_reward = 0.0
        has_operator_eq = graph_tokenizer.graph_vocab.GRAPH_OP_EQ_TOKEN in assistant_msg

        if has_operator_eq:
            try:
                correct_expression_rate = verify_graph_computation_expressions(assistant_msg, no_eq_reward=0.5)
                bonus_reward = 0.3 * correct_expression_rate
            except Exception:
                # If verification fails, no bonus
                bonus_reward = 0.0
        
        return base_reward + bonus_reward
        
    except Exception as e:
        print(f"compute_reward_cycle_detection error: {e}")
        return 0.0

def evaluate_s2_cycle_detection(user_msg: str, assistant_msg: str) -> bool:
    """Verify a single QA pair for cycle detection."""
    try:
        graph_text, encoding = parse_user_message(user_msg)
        truth, pred = verify_cycle_detection(graph_text, assistant_msg, encoding)
        return truth == pred
    except Exception as e:
        print(f"evaluate_s2_cycle_detection error: {e}")
        return False


def load_and_evaluate_s2_cycle_detection(
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

    split_stats = [
        {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "count": 0} for _ in range(num_splits)
    ]

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
            if msg.get("role") == "user":
                user_msg = msg.get("content")
            elif msg.get("role") == "assistant":
                assistant_msg = msg.get("content")

        if user_msg is None or assistant_msg is None:
            print(f"Line {idx}: Missing user or assistant message")
            continue

        try:
            is_correct = evaluate_s2_cycle_detection(user_msg, assistant_msg)
            graph_text, encoding = parse_user_message(user_msg)
            truth, pred = verify_cycle_detection(graph_text, assistant_msg, encoding)
            # reward = compute_reward_cycle_detection(user_msg, assistant_msg)
            # print(f"reward:{reward}")
        except Exception as e:
            print(f"Line {idx}: parsing/verification error -> {e}")
            continue

        split_idx = min(int(idx * num_splits / total_samples), num_splits - 1)
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
            f"  Split {i}: count={cnt}, TP={tp}, FP={fp}, TN={tn}, FN={fn}, Accuracy={acc:.4f}, F1={f1:.4f}"
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


if __name__ == "__main__":
    data_path = "../data/s2_cycle_detection/GraphVocab_Stage2_CycleDetection_CoT_Nodes-11-50_Samples-100_Splits-1_Train.jsonl"
    load_and_evaluate_s2_cycle_detection(data_path, num_splits=1)

