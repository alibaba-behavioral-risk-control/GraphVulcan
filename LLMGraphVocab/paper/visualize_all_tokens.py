import matplotlib.pyplot as plt
import networkx as nx

from graph_vocab.graph_vocabulary import GraphVocabulary


def visualize_all_tokens(graph_vocab: GraphVocabulary, rows: int = 3, cols: int = 10, figsize=(24, 8)):
    """Draw all unique graph tokens in a grid using circular layout."""
    tokens = list(graph_vocab.GRAPH_STR_TOKENS)
    tokens = [t for t in tokens if t != "<G1>"]
    total = len(tokens)

    layout_func = lambda G: nx.circular_layout(G) if G.number_of_nodes() > 1 else {n: (0, 0) for n in G.nodes()}

    # Generate full 3x10 grid
    print("Generating full visualization with circular layout (3x10)...")
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        if idx < total:
            token = tokens[idx]
            G = graph_vocab.GRAPH_VOCAB[token]
            pos = layout_func(G)
            nx.draw(G, pos, with_labels=False, node_color="#1E90FF", edge_color="#333",
                    node_size=500, font_size=12, width=3.0, ax=ax)
            ax.text(0.5, -0.15, token, transform=ax.transAxes,
                    fontsize=18, fontweight='bold', ha='center', va='top')
        ax.axis("off")

    fig.tight_layout(h_pad=0.5, w_pad=1.0)
    out_path = "image/graph_tokens_grid_circular.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved full graph token grid to {out_path}")

    # Generate 3x5 grid for first 15 tokens
    print("Generating first 15 tokens visualization with circular layout (3x5)...")
    fig_small, axes_small = plt.subplots(3, 5, figsize=(15, 8))
    axes_small_flat = axes_small.flatten()

    for idx, ax in enumerate(axes_small_flat):
        if idx < min(15, total):
            token = tokens[idx]
            G = graph_vocab.GRAPH_VOCAB[token]
            pos = layout_func(G)
            nx.draw(G, pos, with_labels=False, node_color="#1E90FF", edge_color="#333",
                    node_size=500, font_size=12, width=3.0, ax=ax)
            ax.text(0.5, -0.15, token, transform=ax.transAxes,
                    fontsize=18, fontweight='bold', ha='center', va='top')
        ax.axis("off")

    fig_small.tight_layout(h_pad=0.5, w_pad=1.0)
    out_path_small = "image/graph_tokens_grid_circular_first15.png"
    fig_small.savefig(out_path_small, dpi=300, bbox_inches='tight')
    plt.close(fig_small)
    print(f"Saved first 15 graph tokens grid to {out_path_small}")


if __name__ == "__main__":
    graph_vocab = GraphVocabulary()
    visualize_all_tokens(graph_vocab, rows=3, cols=10)

