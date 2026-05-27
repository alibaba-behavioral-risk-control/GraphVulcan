import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import argparse
from tqdm import tqdm
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from graph_vocab.graph_tokenizer import GraphTokenizer
from graph_vocab.graph_vocabulary import GraphVocabulary

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT-MULTI-5K", help="TUDataset name (e.g., IMDB-BINARY, IMDB-MULTI)")
parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
parser.add_argument("--seed", type=int, default=42, help="Random seed for cross-validation split")
parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel workers (0=auto, 1=sequential)")
args = parser.parse_args()

graph_vocab = GraphVocabulary()
tokenizer = GraphTokenizer()

LABEL_NAMES = {
    "IMDB-BINARY": {0: "Action", 1: "Romance"},
    "IMDB-MULTI": {0: "Comedy", 1: "Romance", 2: "Sci-Fi"},
    "REDDIT-BINARY": {0: "Question/Answer-based", 1: "Discussion-based"},
    "REDDIT-MULTI-5K": {0: "worldnews", 1: "videos", 2: "AdviceAnimals", 3: "aww", 4: "mildlyinteresting"},
    "COLLAB": {0: "High Energy Physics", 1: "Condensed Matter Physics", 2: "Astro Physics"},
}


def _ensure_tu_raw_files(dataset_name: str, root: str = "./tmp/TUDataset"):
    """
    Ensure TUDataset raw files exist in the directory PyG expects.
    If not present, download via curl (bypasses SSL issues on macOS).
    """
    import subprocess
    import zipfile

    raw_dir = os.path.join(root, dataset_name, "raw")
    indicator_file = os.path.join(raw_dir, f"{dataset_name}_A.txt")

    if os.path.exists(indicator_file):
        return  # already downloaded

    os.makedirs(raw_dir, exist_ok=True)
    zip_url = f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset_name}.zip"
    zip_path = os.path.join(raw_dir, f"{dataset_name}.zip")

    print(f"Pre-downloading {dataset_name} via curl to {raw_dir}...")
    result = subprocess.run(
        ["curl", "-L", "-o", zip_path, zip_url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl download failed: {result.stderr}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            # Extract files directly into raw_dir, stripping the top-level folder
            filename = os.path.basename(member)
            if not filename:
                continue
            with zf.open(member) as src, open(os.path.join(raw_dir, filename), 'wb') as dst:
                dst.write(src.read())

    os.remove(zip_path)
    print(f"Raw files ready in {raw_dir}")


def load_tu_dataset(dataset_name: str):
    """
    Load a TUDataset and convert to list of (nx.Graph, label) pairs.
    Uses torch_geometric if available, otherwise downloads and parses raw files.
    """
    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx

        # Pre-download raw files via curl to bypass aiohttp SSL issues
        pyg_root = "./tmp/TUDataset"
        _ensure_tu_raw_files(dataset_name, root=pyg_root)

        dataset = TUDataset(root=pyg_root, name=dataset_name)
        graphs = []
        for data in dataset:
            graph = to_networkx(data, to_undirected=True)
            graph = nx.Graph(graph)
            label = int(data.y.item())
            if graph.number_of_nodes() > 0:
                graphs.append((graph, label))
        return graphs
    except ImportError:
        print("torch_geometric not installed, falling back to raw file download...")
        return _load_tu_dataset_raw(dataset_name)
    except Exception as download_error:
        print(f"PyG loading failed ({download_error}), falling back to raw file download...")
        return _load_tu_dataset_raw(dataset_name)


def _load_tu_dataset_raw(dataset_name: str):
    """
    Fallback: download and parse TUDataset raw files without torch_geometric.
    """
    import subprocess
    import zipfile

    base_url = f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset_name}.zip"
    cache_dir = f"./tmp/TUDataset_raw/{dataset_name}"
    zip_path = f"./tmp/TUDataset_raw/{dataset_name}.zip"
    data_dir = f"{cache_dir}/{dataset_name}"

    if not os.path.exists(data_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Downloading {dataset_name} from {base_url}...")
        try:
            # Try urllib with SSL verification disabled (macOS Python SSL issue)
            import ssl
            import urllib.request
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            response = urllib.request.urlopen(base_url, context=ssl_context)
            with open(zip_path, 'wb') as f:
                f.write(response.read())
        except Exception:
            # Fallback to curl
            print("urllib failed, falling back to curl...")
            result = subprocess.run(
                ["curl", "-L", "-o", zip_path, base_url],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to download {base_url}: {result.stderr}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        print(f"Extracted to {cache_dir}")

    # Parse raw files
    prefix = data_dir

    # Read adjacency (edge list)
    edges = []
    with open(f"{prefix}/{dataset_name}_A.txt", 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            edges.append((int(parts[0].strip()) - 1, int(parts[1].strip()) - 1))  # 1-indexed to 0-indexed

    # Read graph indicator (which graph each node belongs to)
    graph_indicator = []
    with open(f"{prefix}/{dataset_name}_graph_indicator.txt", 'r') as f:
        for line in f:
            graph_indicator.append(int(line.strip()) - 1)  # 1-indexed to 0-indexed

    # Read graph labels
    graph_labels = []
    with open(f"{prefix}/{dataset_name}_graph_labels.txt", 'r') as f:
        for line in f:
            graph_labels.append(int(line.strip()))

    # Normalize labels to start from 0
    unique_labels = sorted(set(graph_labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    graph_labels = [label_map[label] for label in graph_labels]

    # Build graphs
    num_graphs = max(graph_indicator) + 1
    node_to_graph = graph_indicator

    graph_nodes = [[] for _ in range(num_graphs)]
    for node_idx, graph_idx in enumerate(node_to_graph):
        graph_nodes[graph_idx].append(node_idx)

    graphs = []
    for graph_idx in range(num_graphs):
        nodes = graph_nodes[graph_idx]
        if len(nodes) == 0:
            continue

        graph = nx.Graph()
        node_set = set(nodes)
        # Relabel nodes to 0-indexed within this graph
        node_mapping = {old: new for new, old in enumerate(nodes)}
        graph.add_nodes_from(range(len(nodes)))

        for u, v in edges:
            if u in node_set and v in node_set and u != v:
                graph.add_edge(node_mapping[u], node_mapping[v])

        label = graph_labels[graph_idx]
        graphs.append((graph, label))

    return graphs

def _encode_single_graph(task_args):
    """
    Encode a single graph into a sample. Designed for multiprocessing.
    Each worker creates its own GraphTokenizer to avoid shared state issues.
    """
    idx, graph, label, dataset_name, label_name_map, num_classes = task_args

    worker_tokenizer = GraphTokenizer()

    token_list = worker_tokenizer.tokenize(graph, strategy="greedy+wl")
    graph_text_vocab = worker_tokenizer.token_list_to_text(token_list)
    graph_text_edge = worker_tokenizer.encode_edge_list(graph)
    graph_text_incident = worker_tokenizer.encode_incident(graph)

    if label_name_map:
        label_str = label_name_map[label]
    else:
        label_str = f"Class {label}"

    return {
        "task": "Graph_Classification",
        "dataset": dataset_name,
        "graph_idx": idx,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "graph_text_vocab": graph_text_vocab,
        "graph_text_edge": graph_text_edge,
        "graph_text_incident": graph_text_incident,
        "token_list": token_list,
        "label": label,
        "label_str": label_str,
        "num_classes": num_classes,
        "label_name_map": label_name_map,
    }


def _encode_samples(all_graphs, target_indices, dataset_name, label_name_map, num_classes):
    """
    Encode a subset of graphs into samples, optionally using multiprocessing.

    Args:
        all_graphs: list of (nx.Graph, label) pairs
        target_indices: set of indices to process
        dataset_name: name of the dataset
        label_name_map: mapping from label id to label name
        num_classes: total number of classes
    """
    sorted_indices = sorted(target_indices)

    task_args_list = [
        (idx, all_graphs[idx][0], all_graphs[idx][1], dataset_name, label_name_map, num_classes)
        for idx in sorted_indices
    ]

    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = min(multiprocessing.cpu_count(), len(task_args_list))

    if num_workers <= 1:
        samples = []
        for task_arg in tqdm(task_args_list, desc="Generating samples"):
            samples.append(_encode_single_graph(task_arg))
        return samples

    print(f"  Using {num_workers} parallel workers")
    samples = [None] * len(task_args_list)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_pos = {
            executor.submit(_encode_single_graph, task_arg): pos
            for pos, task_arg in enumerate(task_args_list)
        }
        for future in tqdm(as_completed(future_to_pos), total=len(future_to_pos), desc="Generating samples"):
            pos = future_to_pos[future]
            samples[pos] = future.result()

    return samples


def generate_graph_classification_samples(dataset_name: str, n_folds: int, seed: int):
    """
    Load TUDataset and generate graph classification samples using K-Fold cross-validation.

    Returns:
        all_graphs: list of (nx.Graph, label) pairs
        folds: list of (train_indices, test_indices) for each fold
        label_name_map: mapping from label id to label name
        num_classes: total number of classes
    """
    print(f"Loading {dataset_name}...")
    all_graphs = load_tu_dataset(dataset_name)
    print(f"Loaded {len(all_graphs)} graphs")

    # Print label distribution
    label_counts = {}
    for _, label in all_graphs:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")

    # K-Fold cross-validation split
    random.seed(seed)
    indices = list(range(len(all_graphs)))
    random.shuffle(indices)

    fold_size = len(indices) // n_folds
    folds = []
    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < n_folds - 1 else len(indices)
        test_indices = set(indices[test_start:test_end])
        train_indices = set(indices) - test_indices
        folds.append((train_indices, test_indices))

    label_name_map = LABEL_NAMES.get(dataset_name, None)
    num_classes = len(label_counts)

    return all_graphs, folds, label_name_map, num_classes


def generate_graph_structure_summary(token_list, num_nodes, num_edges):
    """
    Generate a structural summary of the graph based on its graphlet tokens.
    This helps the model reason about the graph's structure for classification.
    """
    token_counts = {}
    for item in token_list:
        token = item["token"]
        if token in [graph_vocab.GRAPH_CONNECT_TOKEN, graph_vocab.GRAPH_DISCONNECT_TOKEN]:
            continue
        token_counts[token] = token_counts.get(token, 0) + 1

    summary_parts = []
    for token_name, count in sorted(token_counts.items(), key=lambda x: -x[1]):
        summary_parts.append(f"{token_name}: {count}")

    return ", ".join(summary_parts)


def generate_cot_reasoning_classification(sample):
    """
    Generate CoT reasoning for graph classification.
    """
    token_list = sample["token_list"]
    num_nodes = sample["num_nodes"]
    num_edges = sample["num_edges"]
    label_str = sample["label_str"]
    label_name_map = sample["label_name_map"]

    structure_summary = generate_graph_structure_summary(token_list, num_nodes, num_edges)

    # Count connected components
    num_components = sum(
        1 for item in token_list
        if item["token"] == graph_vocab.GRAPH_DISCONNECT_TOKEN
    )

    # Compute density
    max_edges = num_nodes * (num_nodes - 1) / 2
    density = num_edges / max_edges if max_edges > 0 else 0

    reasoning = "\n"

    return reasoning


def convert_to_openai_format(samples, encoding_mode="GraphVocab", system_prompt=None):
    print(f"Converting to OpenAI format ({encoding_mode})...")

    try:
        from gen_data.system_prompts import get_system_prompt
    except ImportError:
        from system_prompts import get_system_prompt

    if system_prompt is None:
        system_prompt = get_system_prompt(encoding_mode)

    # Append classification-specific instruction to system prompt
    dataset_name = samples[0]["dataset"] if samples else "IMDB-BINARY"
    label_name_map = samples[0]["label_name_map"] if samples else LABEL_NAMES.get(dataset_name, {})
    if label_name_map:
        class_names = ", ".join(label_name_map.values())
    else:
        class_names = ", ".join(f"Class {i}" for i in range(samples[0]["num_classes"]))

    classification_instruction = (
        f"You are performing a graph classification task. "
        f"Analyze the structural properties of the given graph "
        f"and classify it into one of the following categories: {class_names}. "
    )

    openai_data = []

    for sample in tqdm(samples):
        if encoding_mode == "GraphVocab":
            graph_text = sample["graph_text_vocab"]
        elif encoding_mode == "Incident":
            graph_text = sample["graph_text_incident"]
        else:
            graph_text = sample["graph_text_edge"]

        label_str = sample["label_str"]

        # User message
        user_msg = (
            f"{classification_instruction}"
            f"Given the following graph: {graph_text}. "
            f"Classify this graph into one of the following categories: {class_names}. "
            f"Your answer should be in the format: 'The graph belongs to category: X.'"
        )

        # Assistant message
        assistant_msg = f"The graph belongs to category: {label_str}."

        openai_sample = {
            "task": "graph_classification",
            "dataset": sample["dataset"],
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
    all_graphs, folds, label_name_map, num_classes = generate_graph_classification_samples(
        dataset_name=args.dataset,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    # Setup save path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, "data", f"gc_{args.dataset.lower().replace('-', '_')}") + "/"
    os.makedirs(save_path, exist_ok=True)

    base_suffix = f"_{args.n_folds}Fold"
    base_suffix += f"_Seed-{args.seed}"

    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{args.n_folds}: {len(train_indices)} train / {len(test_indices)} test")
        print(f"{'='*60}")

        train_samples = _encode_samples(
            all_graphs, train_indices, args.dataset, label_name_map, num_classes,
        )
        test_samples = _encode_samples(
            all_graphs, test_indices, args.dataset, label_name_map, num_classes,
        )

        fold_suffix = f"{base_suffix}_Fold{fold_idx + 1}"

        # Save train and test splits for all three encodings
        for encoding_mode in ["GraphVocab", "EdgeList", "Incident"]:
            train_data = convert_to_openai_format(train_samples, encoding_mode=encoding_mode)
            save_to_jsonl(
                train_data,
                save_path + f"{encoding_mode}_Stage2_GraphClassification_{args.dataset}{fold_suffix}_Train.jsonl",
            )
            test_data = convert_to_openai_format(test_samples, encoding_mode=encoding_mode)
            save_to_jsonl(
                test_data,
                save_path + f"{encoding_mode}_Stage2_GraphClassification_{args.dataset}{fold_suffix}_Test.jsonl",
            )

        # Print examples for first fold only
        if fold_idx == 0:
            print("\n--- Example train samples (Fold 1) ---")
            for i, sample in enumerate(train_samples[:3]):
                print(f"\nSample {i + 1}:")
                print(f"  Dataset: {sample['dataset']}")
                print(f"  Nodes: {sample['num_nodes']}, Edges: {sample['num_edges']}")
                print(f"  Label: {sample['label_str']} (id={sample['label']})")
                print(f"  Graph (GraphVocab): {sample['graph_text_vocab'][:200]}...")
                print(f"  Graph (EdgeList): {sample['graph_text_edge'][:200]}...")

    print(f"\nAll {args.n_folds} folds generated successfully!")
