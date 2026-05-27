#!/usr/bin/env python3
"""
Statistics of graph token frequency across real-world graph classification datasets.
Loads full TUDataset graphs, tokenizes them with GraphVocab, and reports
the frequency distribution of each graphlet token type per category.

Only structural tokens are counted (excluding <G_Connect> and <G_Disconnect>).
Statistics are computed separately for each class within each dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import networkx as nx
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")

from graph_vocab.graph_tokenizer import GraphTokenizer
from graph_vocab.graph_vocabulary import GraphVocabulary

# Only IMDB-BINARY and REDDIT-BINARY
DATASET_CONFIGS = {
    "IMDB-BINARY": {
        "label_names": {0: "Class 0", 1: "Class 1"},
        "max_samples": None,
    },
    "REDDIT-BINARY": {
        "label_names": {0: "Class 0", 1: "Class 1"},
        "max_samples": None,  # Large graphs (~429 nodes avg), sample 25/class for efficiency
    },
}

# Tokens to exclude from statistics (connection markers, not structural)
EXCLUDED_TOKENS = {"<G_Connect>", "<G_Disconnect>"}

OUTPUT_DIR = "paper"
IMAGE_DIR = "image"
SEED = 42


def _load_tu_dataset_raw(dataset_name: str):
    """Download and parse TUDataset raw files."""
    import subprocess
    import zipfile
    import ssl
    import urllib.request

    base_url = f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset_name}.zip"
    cache_dir = f"./tmp/TUDataset_raw/{dataset_name}"
    zip_path = f"./tmp/TUDataset_raw/{dataset_name}.zip"
    data_dir = f"{cache_dir}/{dataset_name}"

    if not os.path.exists(data_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Downloading {dataset_name} from {base_url}...")
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            response = urllib.request.urlopen(base_url, context=ssl_context)
            with open(zip_path, 'wb') as f:
                f.write(response.read())
        except Exception:
            result = subprocess.run(
                ["curl", "-k", "-L", "-o", zip_path, base_url],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to download {base_url}: {result.stderr}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        print(f"Extracted to {cache_dir}")

    prefix = data_dir

    edges = []
    with open(f"{prefix}/{dataset_name}_A.txt", 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            edges.append((int(parts[0].strip()) - 1, int(parts[1].strip()) - 1))

    graph_indicator = []
    with open(f"{prefix}/{dataset_name}_graph_indicator.txt", 'r') as f:
        for line in f:
            graph_indicator.append(int(line.strip()) - 1)

    graph_labels = []
    with open(f"{prefix}/{dataset_name}_graph_labels.txt", 'r') as f:
        for line in f:
            graph_labels.append(int(line.strip()))

    unique_labels = sorted(set(graph_labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    graph_labels = [label_map[label] for label in graph_labels]

    num_graphs = max(graph_indicator) + 1
    graph_nodes = [[] for _ in range(num_graphs)]
    for node_idx, graph_idx in enumerate(graph_indicator):
        graph_nodes[graph_idx].append(node_idx)

    graphs = []
    for graph_idx in range(num_graphs):
        nodes = graph_nodes[graph_idx]
        if len(nodes) == 0:
            continue

        graph = nx.Graph()
        node_set = set(nodes)
        node_mapping = {old: new for new, old in enumerate(nodes)}
        graph.add_nodes_from(range(len(nodes)))

        for u, v in edges:
            if u in node_set and v in node_set and u != v:
                graph.add_edge(node_mapping[u], node_mapping[v])

        label = graph_labels[graph_idx]
        graphs.append((graph, label))

    return graphs


def load_tu_dataset(dataset_name: str):
    """Load TUDataset, trying PyG first, falling back to raw file parsing."""
    try:
        from torch_geometric.datasets import TUDataset as PyGTU
        from torch_geometric.utils import to_networkx

        pyg_root = "./tmp/TUDataset"
        dataset = PyGTU(root=pyg_root, name=dataset_name)
        graphs = []
        for data in dataset:
            graph = to_networkx(data, to_undirected=True)
            graph = nx.Graph(graph)
            label = int(data.y.item())
            if graph.number_of_nodes() > 0:
                graphs.append((graph, label))
        return graphs
    except (ImportError, Exception):
        return _load_tu_dataset_raw(dataset_name)


def _tokenize_single_graph(graph):
    """Worker function for multiprocessing: tokenize one graph and return counters."""
    tokenizer = GraphTokenizer()
    token_list = tokenizer.tokenize(graph, strategy="greedy+wl")

    token_counter = Counter()
    structural_tokens = []
    for token_info in token_list:
        token = token_info["token"]
        if token not in EXCLUDED_TOKENS:
            token_counter[token] += 1
            node_set = set(token_info.get("node_ids", []))
            if node_set:
                structural_tokens.append((token, node_set))

    # Count co-occurrences: pairs sharing at least one node
    cooccurrence_counter = Counter()
    for i in range(len(structural_tokens)):
        for j in range(i + 1, len(structural_tokens)):
            token_a, nodes_a = structural_tokens[i]
            token_b, nodes_b = structural_tokens[j]
            if nodes_a & nodes_b:
                pair = tuple(sorted([token_a, token_b]))
                cooccurrence_counter[pair] += 1

    return token_counter, cooccurrence_counter

def count_tokens_by_category(dataset_name: str, label_names: dict, max_samples: int = None):
    """
    Load dataset and count graph token occurrences per category.
    Excludes <G_Connect> and <G_Disconnect>.

    Returns:
        dict: {category_name: {"counter": Counter, "num_graphs": int}}
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    all_graphs = load_tu_dataset(dataset_name)
    print(f"  Total graphs: {len(all_graphs)}")

    # Group by label
    graphs_by_label = defaultdict(list)
    for graph, label in all_graphs:
        graphs_by_label[label].append(graph)

    for label_id, graphs in graphs_by_label.items():
        cat_name = label_names.get(label_id, f"Class-{label_id}")
        print(f"  {cat_name}: {len(graphs)} graphs")

    # Sample if needed
    if max_samples:
        random.seed(SEED)
        for label_id in graphs_by_label:
            pool = graphs_by_label[label_id]
            samples_per_class = max_samples // len(label_names)
            if len(pool) > samples_per_class:
                graphs_by_label[label_id] = random.sample(pool, samples_per_class)
        print(f"  Sampled ~{max_samples} graphs total for efficiency")

    # Tokenize and count per category (also collect co-occurrence by shared nodes)
    total_graphs = sum(len(g) for g in graphs_by_label.values())
    print(f"\n  Tokenizing {total_graphs} graphs (multiprocess)...")
    results = {}

    for label_id, graphs in graphs_by_label.items():
        cat_name = label_names.get(label_id, f"Class-{label_id}")

        # Use multiprocessing for tokenization
        from multiprocessing import Pool, cpu_count
        num_workers = min(cpu_count(), 8)
        chunksize = max(1, len(graphs) // (num_workers * 4))
        with Pool(num_workers) as pool:
            token_results = list(tqdm(
                pool.imap(_tokenize_single_graph, graphs, chunksize=chunksize),
                total=len(graphs),
                desc=f"  {dataset_name}/{cat_name}",
            ))

        # Aggregate results
        token_counter = Counter()
        cooccurrence_counter = Counter()
        for single_counter, single_cooc in token_results:
            token_counter.update(single_counter)
            cooccurrence_counter.update(single_cooc)

        results[cat_name] = {
            "counter": token_counter,
            "cooccurrence": cooccurrence_counter,
            "num_graphs": len(graphs),
        }

    return results


def get_structural_token_order(all_counters: list):
    """Determine display order for structural tokens."""
    all_tokens = set()
    for counter in all_counters:
        all_tokens.update(counter.keys())

    # Remove non-structural tokens
    all_tokens -= EXCLUDED_TOKENS

    token_order = []

    # k=1
    if "<G1>" in all_tokens:
        token_order.append("<G1>")

    # k=2
    if "<G2_edge>" in all_tokens:
        token_order.append("<G2_edge>")

    # k=3
    for t in ["<G3_path>", "<G3_triangle>"]:
        if t in all_tokens:
            token_order.append(t)

    # k=4
    k4_order = ["<G4_path>", "<G4_star>", "<G4_cycle>", "<G4_paw>", "<G4_diamond>", "<G4_k4>"]
    for t in k4_order:
        if t in all_tokens:
            token_order.append(t)

    # k=5
    k5_tokens = sorted(
        [t for t in all_tokens if t.startswith("<G5_")],
        key=lambda x: int(x.replace("<G5_", "").replace(">", ""))
    )
    token_order.extend(k5_tokens)

    return token_order


def print_category_table(dataset_name: str, cat_name: str, counter: Counter, num_graphs: int):
    """Print formatted frequency table for one category."""
    total_tokens = sum(counter.values())
    print(f"\n--- {dataset_name} / {cat_name} ({num_graphs} graphs, {total_tokens} tokens) ---")
    print(f"{'Token':<15} {'Count':>8} {'Avg/Graph':>10} {'Percentage':>10}")
    print("-" * 48)

    for token, count in counter.most_common():
        avg_per_graph = count / num_graphs
        percentage = count / total_tokens * 100
        print(f"{token:<15} {count:>8} {avg_per_graph:>10.2f} {percentage:>9.2f}%")

    print(f"{'TOTAL':<15} {total_tokens:>8} {total_tokens / num_graphs:>10.2f} {'100.00':>9}%")


def plot_token_frequency_by_category(all_dataset_results: dict, output_dir: str):
    """
    Plot one figure per dataset with two subplots (left=class0, right=class1).
    Uses large bold fonts, no overall suptitle, dataset name in subplot titles.
    """
    colors = ["#4C72B0", "#C44E52", "#55A868", "#DD8452"]

    for dataset_name, category_results in all_dataset_results.items():
        categories = list(category_results.keys())
        all_counters = [category_results[c]["counter"] for c in categories]
        token_order = get_structural_token_order(all_counters)

        # Filter tokens with zero count across both categories
        token_order = [t for t in token_order if any(
            category_results[c]["counter"].get(t, 0) > 0 for c in categories
        )]

        # Keep only top-15 most frequent tokens (by combined avg frequency)
        combined_avg_freq = {}
        for t in token_order:
            total_freq = sum(
                category_results[c]["counter"].get(t, 0) / category_results[c]["num_graphs"]
                for c in categories
            )
            combined_avg_freq[t] = total_freq
        num_to_keep = 15
        top_tokens = set(
            sorted(combined_avg_freq.keys(), key=lambda t: combined_avg_freq[t], reverse=True)[:num_to_keep]
        )
        token_order = [t for t in token_order if t in top_tokens]

        num_tokens = len(token_order)
        display_names = [t.replace("<", "").replace(">", "") for t in token_order]

        fig, axes = plt.subplots(1, 2, figsize=(24, 9), dpi=150, sharey=True)

        for idx, cat_name in enumerate(categories):
            ax = axes[idx]
            data = category_results[cat_name]
            counter = data["counter"]
            num_graphs = data["num_graphs"]
            color = colors[idx % len(colors)]

            frequencies = [counter.get(t, 0) / num_graphs for t in token_order]
            x_positions = np.arange(num_tokens)

            ax.bar(x_positions, frequencies, color=color, alpha=0.55,
                   edgecolor=color, linewidth=5)
            ax.set_title(f"{dataset_name}\n({cat_name})", fontsize=42, fontweight="bold", pad=20)
            ax.set_ylabel("Avg Frequency per Graph", fontsize=35, fontweight="bold")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(display_names, rotation=50, ha="center", fontsize=30, fontweight="bold")
            ax.tick_params(axis="y", labelsize=24)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)

        # Only left subplot shows y-label
        axes[1].set_ylabel("")

        plt.tight_layout()

        safe_name = dataset_name.replace("-", "_").lower()
        output_path = os.path.join(output_dir, f"graph_token_frequency_{safe_name}.pdf")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"\nPlot saved to {output_path}")
        plt.close()


def save_results_json(all_dataset_results: dict, output_path: str):
    """Save detailed frequency statistics as JSON."""
    output_data = {}
    for dataset_name, category_results in all_dataset_results.items():
        dataset_data = {}
        for cat_name, data in category_results.items():
            counter = data["counter"]
            num_graphs = data["num_graphs"]
            total_tokens = sum(counter.values())

            dataset_data[cat_name] = {
                "num_graphs": num_graphs,
                "total_structural_tokens": total_tokens,
                "avg_tokens_per_graph": round(total_tokens / num_graphs, 2),
                "token_frequencies": {
                    token: {
                        "count": count,
                        "avg_per_graph": round(count / num_graphs, 4),
                        "percentage": round(count / total_tokens * 100, 4),
                    }
                    for token, count in counter.most_common()
                },
            }
        output_data[dataset_name] = dataset_data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    all_dataset_results = {}

    for dataset_name, config in DATASET_CONFIGS.items():
        category_results = count_tokens_by_category(
            dataset_name,
            label_names=config["label_names"],
            max_samples=config.get("max_samples"),
        )
        all_dataset_results[dataset_name] = category_results

        # Print tables
        for cat_name, data in category_results.items():
            print_category_table(dataset_name, cat_name, data["counter"], data["num_graphs"])

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_output = os.path.join(OUTPUT_DIR, "graph_token_frequency.json")
    save_results_json(all_dataset_results, json_output)

    # Plot (one figure per dataset, saved to image/)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    plot_token_frequency_by_category(all_dataset_results, IMAGE_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset_name, category_results in all_dataset_results.items():
        print(f"\n{dataset_name}:")
        for cat_name, data in category_results.items():
            counter = data["counter"]
            cooccurrence = data["cooccurrence"]
            num_graphs = data["num_graphs"]
            total = sum(counter.values())
            top3 = counter.most_common(3)
            top3_str = ", ".join(f"{t}({c/num_graphs:.1f}/graph)" for t, c in top3)
            print(f"  {cat_name} ({num_graphs} graphs): {total} tokens, top-3: {top3_str}")
            top3_cooc = cooccurrence.most_common(3)
            if top3_cooc:
                cooc_str = ", ".join(
                    f"{a}+{b}({c/num_graphs:.1f}/graph)" for (a, b), c in top3_cooc
                )
                print(f"    Top co-occurrences: {cooc_str}")

    print("\nDone!")
