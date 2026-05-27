import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import networkx as nx
from tqdm import tqdm

from graph_vocab.graph_tokenizer import GraphTokenizer
from utils.tools import mean_and_var
from utils.functional import verify_graph_computation_expressions


graph_tokenizer = GraphTokenizer()


def parse_user_message(user_msg: str) -> Tuple[str, str, str]:
    """Extract graph A text, graph B text, and encoding (GraphVocab, EdgeList, or Incident)."""
    # Try "Graph A:" / "Graph B:" block
    match = re.search(r"Graph A:\s*(.*?)\s*Graph B:\s*(.*?)Provide", user_msg, re.IGNORECASE | re.DOTALL)
    if match:
        g1_text = match.group(1).strip()
        g2_text = match.group(2).strip()
    else:
        raise ValueError(f"Cannot parse graphs from user message: {user_msg}")

    # Determine encoding type
    if g1_text.startswith("This graph has nodes"):
        encoding = "Incident"
    elif "Nodes:" in g1_text and "Edges:" in g1_text:
        encoding = "EdgeList"
    else:
        encoding = "GraphVocab"
    return g1_text, g2_text, encoding


def parse_answer(assistant_msg: str) -> Tuple[List[int], List[int]]:
    """Extract node lists for Graph A and Graph B from assistant message."""
    # Try explicit Graph A / Graph B tags
    match = re.search(r"Graph\s*A:\s*\[([^\]]*)\].*?Graph\s*B:\s*\[([^\]]*)\]", assistant_msg, re.IGNORECASE | re.DOTALL)
    if not match:
        # Fallback: first and second bracket lists
        lists = re.findall(r"\[([^\]]*)\]", assistant_msg)
        if len(lists) < 2:
            return [], []
        a_list, b_list = lists[-2], lists[-1]
    else:
        a_list, b_list = match.group(1), match.group(2)

    def to_int_list(s: str):
        s = s.strip()
        if not s:
            return []
        return [int(x.strip()) for x in s.split(',') if x.strip()]

    return to_int_list(a_list), to_int_list(b_list)


def reconstruct_graph(graph_text: str, encoding: str) -> nx.Graph:
    if encoding == "GraphVocab":
        return graph_tokenizer.decode_graph_vocab(graph_text)
    elif encoding == "EdgeList":
        return graph_tokenizer.decode_edge_list(graph_text)
    elif encoding == "Incident":
        return graph_tokenizer.decode_incident(graph_text)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def compute_mcs_size(G1: nx.Graph, G2: nx.Graph) -> int:
    ismags = nx.algorithms.isomorphism.ISMAGS(G1, G2)
    # ismags_inverse = nx.algorithms.isomorphism.ISMAGS(G2, G1)
    # isomorphic quick path
    if ismags.is_isomorphic():
        if G1.number_of_nodes() == 0:
            return 0
        return G1.number_of_nodes()
    mappings = list(ismags.largest_common_subgraph(symmetry=False))
    # mappings_inverse = list(ismags_inverse.largest_common_subgraph(symmetry=False))
    # print(f"len(mappings): {len(mappings[0])}")
    # print(f"len(mappings_inverse): {len(mappings_inverse[0])}")
    if mappings:
        mapping = mappings[0]
        common_nodes_G1 = list(mapping.keys())
        common_subgraph = G1.subgraph(common_nodes_G1).copy()
        connected_common_subgraph = nx.connected_components(common_subgraph)
        largest_cc = max(connected_common_subgraph, key=len)
        common_subgraph = common_subgraph.subgraph(largest_cc).copy()
        mapping = {k: v for k, v in mapping.items() if k in largest_cc}
        return len(mapping)
    else:
        return 0


def verify_mcs(graph1_text: str, graph2_text: str, assistant_msg: str, encoding: str) -> bool:
    try:
        G1 = reconstruct_graph(graph1_text, encoding)
        G2 = reconstruct_graph(graph2_text, encoding)
    except Exception as e:
        print(f"Failed to reconstruct graphs: {e}")
        return False

    nodes_a, nodes_b = parse_answer(assistant_msg)
    if not nodes_a or not nodes_b or len(nodes_a) != len(nodes_b):
        return False

    subG1 = G1.subgraph(nodes_a)
    subG2 = G2.subgraph(nodes_b)

    if not nx.is_isomorphic(subG1, subG2):
        return False

    true_size = compute_mcs_size(G2, G1)
    if true_size == 0:
        return False

    if len(nodes_a) != true_size or len(nodes_b) != true_size:
        return False

    return True

def compute_reward_mcs(user_msg: str, assistant_msg: str) -> float:
    """Compute reward for a single MCS QA pair."""
    try:
        g1_text, g2_text, encoding = parse_user_message(user_msg)
        G1 = reconstruct_graph(g1_text, encoding)
        G2 = reconstruct_graph(g2_text, encoding)
    except Exception as e:
        print(f"Failed to reconstruct graphs: {e}")
        return 0.0

    nodes_a, nodes_b = parse_answer(assistant_msg)
    if not nodes_a or not nodes_b or len(nodes_a) != len(nodes_b):
        return 0.0

    has_operator_eq = graph_tokenizer.graph_vocab.GRAPH_OP_EQ_TOKEN in assistant_msg
    bonus_reward = 0.0
    if has_operator_eq:
        try:
            correct_expression_rate = verify_graph_computation_expressions(assistant_msg, no_eq_reward=0.5)
            bonus_reward = 0.30 * correct_expression_rate
        except Exception:
            # If verification fails, no bonus
            bonus_reward = 0.0

    subG1 = G1.subgraph(nodes_a)
    subG2 = G2.subgraph(nodes_b)
    true_size = compute_mcs_size(G2, G1)
    if not nx.is_isomorphic(subG1, subG2):
        base_reward = 0.0
    elif len(nodes_a) != true_size or len(nodes_b) != true_size:
        base_reward = 0.35
    else:
        base_reward = 0.7
    reward = base_reward + bonus_reward
    return reward


def evaluate_s2_mcs(user_msg: str, assistant_msg: str) -> bool:
    """Verify a single MCS QA pair. Returns True if assistant answer is correct."""
    try:
        g1_text, g2_text, encoding = parse_user_message(user_msg)
        return verify_mcs(g1_text, g2_text, assistant_msg, encoding)
    except Exception as e:
        print(f"evaluate_s2_mcs error: {e}")
        return False


def load_and_evaluate_s2_mcs(dataset_path: str, tokenizer_obj=None, num_splits: int = 1, verbose: bool = True) -> List[Dict[str, Any]]:
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

    print(f"Now evaluating s2 MCS dataset: {dataset_path}")
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

        try:
            graph1_text, graph2_text, encoding = parse_user_message(user_msg)
            is_correct = evaluate_s2_mcs(user_msg, assistant_msg)
            # reward = compute_reward_mcs(user_msg, assistant_msg)
            # print(f"reward:{reward}")
        except Exception as e:
            print(f"Line {idx}: parsing/verification error -> {e}")
            continue

        split_idx = min(idx * num_splits // total_samples, num_splits - 1)
        split_count[split_idx] += 1
        if is_correct:
            split_correct[split_idx] += 1
        else:
            if verbose:
                print(f"\nBad case at line {idx}:")
                print(f"user_msg: {user_msg}")
                print(f"assistant_msg: {assistant_msg}")

        if tokenizer_obj is not None and graph1_text and graph2_text:
            try:
                g1_tokens = tokenizer_obj.tokenize(graph1_text)
            except Exception as e:
                print(f"Tokenizer.tokenize failed for graph1_text at line {idx}: {e}")
                g1_tokens = []
            try:
                a_tokens = tokenizer_obj.tokenize(assistant_msg)
            except Exception as e:
                print(f"Tokenizer.tokenize failed for assistant_msg at line {idx}: {e}")
                a_tokens = []
            total_graph_tokens += len(g1_tokens)
            total_assistant_tokens += len(a_tokens)
            tokenizer_sample_count += 1

        results.append({
            "line": idx,
            "question": user_msg,
            "output": assistant_msg,
            "correct": is_correct,
        })

    acc_list: List[float] = []
    print("\nPer-split accuracy:")
    for i in range(num_splits):
        cnt = split_count[i]
        acc = (split_correct[i] / cnt) if cnt else 0.0
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
    data_path = "../data/s2_max_common_subgraph/Incident_Stage2_MCS_CoT_Nodes-5-7_Samples-100_Splits-1_Test.jsonl"
    load_and_evaluate_s2_mcs(data_path, num_splits=10)

