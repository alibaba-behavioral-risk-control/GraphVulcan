import re
import json
import argparse
from typing import Optional, List, Dict, Any
from utils.tools import mean_and_var
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_classification_answer(assistant_msg: str) -> Optional[str]:
    """
    Extract the predicted category from the assistant message.
    Expected format: 'The graph belongs to category: X.'
    Also handles  reasoning blocks.
    """
    # Remove  blocks if present
    cleaned = re.sub(r"", "", assistant_msg, flags=re.DOTALL).strip()

    # Try standard format: "The graph belongs to category: X."
    match = re.search(
        r"(?:The graph belongs to category|category)\s*[:：]\s*(.+?)(?:\.|$)",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().rstrip(".")

    # Fallback: look for the last quoted or standalone category name
    # after common phrases like "is", "classified as", etc.
    fallback = re.search(
        r"(?:classified as|is|belongs to|answer is|category is)\s*[:：]?\s*[\"']?(.+?)[\"']?\s*(?:\.|$)",
        cleaned,
        re.IGNORECASE,
    )
    if fallback:
        return fallback.group(1).strip().rstrip(".")

    return None


def parse_ground_truth(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Extract the ground truth category from the original assistant message
    in the test data (messages format with system/user/assistant).
    """
    for msg in messages:
        if msg.get("role") == "assistant":
            return parse_classification_answer(msg["content"])
    return None


def load_and_evaluate_graph_classification(
    data_path: str,
    ground_truth_path: Optional[str] = None,
    tokenizer_obj=None,
    num_splits: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate graph classification predictions.

    Args:
        data_path: Path to the inference output JSONL (user + assistant predictions).
        ground_truth_path: Path to the original test JSONL with ground truth labels.
        tokenizer_obj: Optional tokenizer for token counting.
        num_splits: Number of splits (for compatibility; typically 1 for classification).
        verbose: Whether to print per-sample details.

    Returns:
        Dictionary with accuracy statistics.
    """
    # Load predictions
    predictions = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    # Load ground truth if provided
    ground_truths = []
    if ground_truth_path:
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ground_truths.append(json.loads(line))

    total = len(predictions)
    correct = 0
    parse_failures = 0

    total_graph_tokens = 0
    total_assistant_tokens = 0

    for idx, pred_item in enumerate(tqdm(predictions, desc="Evaluating")):
        pred_messages = pred_item.get("messages", [])

        # Extract predicted answer
        pred_answer = None
        for msg in pred_messages:
            if msg.get("role") == "assistant":
                pred_answer = parse_classification_answer(msg["content"])
                if tokenizer_obj:
                    total_assistant_tokens += len(
                        tokenizer_obj.encode(msg["content"])
                    )
                break

        # Extract ground truth
        gt_answer = None
        if ground_truths and idx < len(ground_truths):
            gt_messages = ground_truths[idx].get("messages", [])
            gt_answer = parse_ground_truth(gt_messages)

            # Count graph tokens from user message
            if tokenizer_obj:
                for msg in gt_messages:
                    if msg.get("role") == "user":
                        total_graph_tokens += len(
                            tokenizer_obj.encode(msg["content"])
                        )
                        break

        if pred_answer is None:
            parse_failures += 1
            if verbose:
                print(f"[{idx}] PARSE FAILURE")
                for msg in pred_messages:
                    if msg.get("role") == "assistant":
                        print(f"  Assistant: {msg['content'][:200]}")
            continue

        is_correct = (
            pred_answer.lower().strip() == gt_answer.lower().strip()
            if gt_answer
            else False
        )
        if is_correct:
            correct += 1

        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"[{idx}] {status} Pred: {pred_answer} | GT: {gt_answer}")

    accuracy = correct / total if total > 0 else 0.0
    avg_graph_tokens = total_graph_tokens / total if total > 0 else 0
    avg_assistant_tokens = total_assistant_tokens / total if total > 0 else 0

    results = {
        "total": total,
        "correct": correct,
        "parse_failures": parse_failures,
        "accuracy": accuracy,
        "avg_graph_tokens": avg_graph_tokens,
        "avg_assistant_tokens": avg_assistant_tokens,
    }

    print(f"\n{'='*60}")
    print(f"Graph Classification Evaluation Results")
    print(f"{'='*60}")
    print(f"Total samples:     {total}")
    print(f"Correct:           {correct}")
    print(f"Parse failures:    {parse_failures}")
    print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    if tokenizer_obj:
        print(f"Avg graph tokens:  {avg_graph_tokens:.1f}")
        print(f"Avg asst tokens:   {avg_assistant_tokens:.1f}")
    print(f"{'='*60}")

    return results


def compute_reward_graph_classification(
    pred_messages: List[Dict[str, str]],
    gt_messages: List[Dict[str, str]],
) -> float:
    """Compute reward for graph classification (for RL training)."""
    pred_answer = None
    for msg in pred_messages:
        if msg.get("role") == "assistant":
            pred_answer = parse_classification_answer(msg["content"])
            break

    gt_answer = parse_ground_truth(gt_messages)

    if pred_answer is None or gt_answer is None:
        return 0.0

    if pred_answer.lower().strip() == gt_answer.lower().strip():
        return 1.0
    return 0.0


if __name__ == "__main__":
    data_path = "../data/gc_imdb_binary/GraphVocab_Stage2_GraphClassification_IMDB-BINARY_5Fold_Seed-42_Fold1_Test.jsonl"
    load_and_evaluate_graph_classification(data_path=data_path)
