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

def parse_user_message(user_msg: str) -> Tuple[str, str]:
    """Extract graph text and encoding (GraphVocab, EdgeList, or Incident)."""
    # Try to find graph text after "You are given the following graph:"
    match = re.search(r"You are given the following graph:\s*(.*?)\s*Provide", user_msg, re.IGNORECASE | re.DOTALL)
    if match:
        graph_text = match.group(1).strip()
    else:
        raise ValueError(f"Cannot parse graph from user message: {user_msg}")

    # Determine encoding type
    if graph_text.startswith("This graph has nodes"):
        encoding = "Incident"
    elif "Nodes:" in graph_text and "Edges:" in graph_text:
        encoding = "EdgeList"
    else:
        encoding = "GraphVocab"
    
    return graph_text, encoding

def parse_answer(assistant_msg: str) -> List[int]:
    """Extract node list from assistant message."""
    # Try to find "The maximum clique is: [...]"
    match = re.search(r"The maximum clique is:\s*\[([^\]]*)\]", assistant_msg, re.IGNORECASE)
    if not match:
        # Fallback: find last bracket list
        lists = re.findall(r"\[([^\]]*)\]", assistant_msg)
        if not lists:
            return []
        node_list_str = lists[-1]
    else:
        node_list_str = match.group(1)

    node_list_str = node_list_str.strip()
    if not node_list_str:
        return []
    
    return [int(x.strip()) for x in node_list_str.split(',') if x.strip()]

def reconstruct_graph(graph_text: str, encoding: str) -> nx.Graph:
    """Reconstruct graph from text representation."""
    if encoding == "GraphVocab":
        return graph_tokenizer.decode_graph_vocab(graph_text)
    elif encoding == "EdgeList":
        return graph_tokenizer.decode_edge_list(graph_text)
    elif encoding == "Incident":
        return graph_tokenizer.decode_incident(graph_text)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

def is_clique(G: nx.Graph, nodes: List[int]) -> bool:
    """Check if the given nodes form a clique in graph G."""
    if len(nodes) == 0:
        return False
    
    # Check if all nodes exist in the graph
    for node in nodes:
        if node not in G.nodes():
            return False
    
    # Check if all pairs of nodes are connected
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not G.has_edge(nodes[i], nodes[j]):
                return False
    
    return True

def find_maximum_clique_size(G: nx.Graph) -> int:
    """Find the size of the maximum clique in graph G."""
    try:
        # Use networkx's find_cliques to find all maximal cliques
        max_clique = max(nx.find_cliques(G), key=len)
        return len(max_clique)
    except ValueError:
        # No cliques found (empty graph)
        return 0

def verify_max_clique(graph_text: str, assistant_msg: str, encoding: str) -> bool:
    """
    Verify if the assistant's answer is correct.
    
    Logic:
    1. Reconstruct graph from user_msg
    2. Parse answer nodes from assistant_msg
    3. Check if the nodes form a clique
    4. Check if the clique size equals the maximum clique size
    """
    try:
        # Step 1: Reconstruct graph
        G = reconstruct_graph(graph_text, encoding)
    except Exception as e:
        print(f"Failed to reconstruct graph: {e}")
        return False

    # Step 2: Parse answer nodes
    answer_nodes = parse_answer(assistant_msg)
    if not answer_nodes:
        return False

    # Check if nodes form a clique
    if not is_clique(G, answer_nodes):
        return False

    # Check if clique size equals maximum clique size
    max_clique_size = find_maximum_clique_size(G)
    if len(answer_nodes) != max_clique_size:
        return False

    return True

def compute_reward_max_clique(user_msg: str, assistant_msg: str) -> float:
    """
    Compute reward for a single max clique QA pair.
    
    Reward structure:
    - Base reward: 0.7 if correct, 0.35 if partial, 0.0 if wrong
    - Bonus reward: +0.3 * correct_expression_rate if <G_Operator_Eq> is used
    
    Returns:
        float: Reward score (0.0 to 1.0)
    """
    try:
        graph_text, encoding = parse_user_message(user_msg)
        G = reconstruct_graph(graph_text, encoding)
        answer_nodes = parse_answer(assistant_msg)
        
        if not answer_nodes:
            return 0.0
        
        # Check if nodes form a clique
        if not is_clique(G, answer_nodes):
            base_reward = 0.0
        else:
            # Check if clique size equals maximum clique size
            max_clique_size = find_maximum_clique_size(G)
            if len(answer_nodes) != max_clique_size:
                base_reward = 0.35
            else:
                base_reward = 0.7
        
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
        print(f"compute_reward_max_clique error: {e}")
        return 0.0

def evaluate_s2_max_clique(user_msg: str, assistant_msg: str) -> bool:
    """Verify a single max clique QA pair. Returns True if assistant answer is correct."""
    try:
        graph_text, encoding = parse_user_message(user_msg)
        return verify_max_clique(graph_text, assistant_msg, encoding)
    except Exception as e:
        print(f"evaluate_s2_max_clique error: {e}")
        return False

def load_and_evaluate_s2_max_clique(dataset_path: str, tokenizer_obj=None, num_splits: int = 1, verbose: bool = True) -> List[Dict[str, Any]]:
    """Load and evaluate max clique dataset."""
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

    print(f"Now evaluating s2 Max Clique dataset: {dataset_path}")
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
            is_correct = evaluate_s2_max_clique(user_msg, assistant_msg)
            # reward = compute_reward_max_clique(user_msg, assistant_msg)
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

        if tokenizer_obj is not None:
            try:
                graph_text, _ = parse_user_message(user_msg)
                g_tokens = tokenizer_obj.tokenize(graph_text)
            except Exception as e:
                print(f"Tokenizer.tokenize failed for graph_text at line {idx}: {e}")
                g_tokens = []
            try:
                a_tokens = tokenizer_obj.tokenize(assistant_msg)
            except Exception as e:
                print(f"Tokenizer.tokenize failed for assistant_msg at line {idx}: {e}")
                a_tokens = []
            total_graph_tokens += len(g_tokens)
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
    data_path = "../data/s2_max_clique/GraphVocab_Stage2_MaxClique_CoT_Nodes-5-10_Samples-100_Splits-1_Train.jsonl"
    load_and_evaluate_s2_max_clique(data_path, num_splits=1)
