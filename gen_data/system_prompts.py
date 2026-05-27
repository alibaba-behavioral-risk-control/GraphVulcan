import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_vocab.graph_vocabulary import GraphVocabulary

# Initialize graph vocabulary for token information
graph_vocab = GraphVocabulary()


def get_system_prompt(encoding_mode: str) -> str:
    """
    Get the system prompt for a specific encoding mode.
    
    Args:
        encoding_mode: Either "GraphVocab", "EdgeList", or "Incident"
    
    Returns:
        str: The system prompt for the specified encoding mode
    
    Raises:
        ValueError: If encoding_mode is not "GraphVocab", "EdgeList", or "Incident"
    """
    if encoding_mode not in ["GraphVocab", "EdgeList", "Incident"]:
        raise ValueError(f"Unknown encoding mode: {encoding_mode}. Must be 'GraphVocab', 'EdgeList', or 'Incident'.")
    
    if encoding_mode == "GraphVocab":
        return _get_graph_vocab_prompt()
    elif encoding_mode == "EdgeList":
        return _get_edge_list_prompt()
    else:
        return _get_incident_prompt()


def _get_graph_vocab_prompt() -> str:
    """Generate universal GraphVocab system prompt."""
    
    base_intro = "You are a graph reasoning assistant. "
    
    tokens_desc = (
        "The following are graph tokens: {" + ", ".join(f"{t}" for t in graph_vocab.GRAPH_STR_TOKENS) + "}. "
        f"Each graph token represents a connected subgraph. Node IDs precede the token in this format: "
        f"{graph_vocab.NODE_ID_BEGIN_TOKEN}1{graph_vocab.NODE_ID_SPLIT_TOKEN}2{graph_vocab.NODE_ID_END_TOKEN}<graph_token>. "
        f"where {graph_vocab.NODE_ID_SPLIT_TOKEN} separates node IDs, and {graph_vocab.NODE_ID_BEGIN_TOKEN}/{graph_vocab.NODE_ID_END_TOKEN} mark the start/end of node IDs. "
        "The order of node IDs reflects their relative positions within the graph. "
    )
    
    operators_desc = (
        "Operators: "
        f"{graph_vocab.GRAPH_OP_EQ_TOKEN}: Indicates the graphs on both sides are identical in structure and node IDs. "
        f"{graph_vocab.GRAPH_CONNECT_TOKEN}: The graphs on both sides belong to the same connected component. "
        f"{graph_vocab.GRAPH_DISCONNECT_TOKEN}: Marks that the left and right graphs belong to separate connected components."
    )
    
    task_instruction = "Given an undirected graph represented by graph tokens with node IDs, perform the requested reasoning task. Think step by step."
    
    return base_intro + tokens_desc + operators_desc + task_instruction


def _get_edge_list_prompt() -> str:
    """Generate universal EdgeList system prompt."""
    
    base_intro = "You are a graph reasoning assistant. "
    
    format_desc = (
        "Graphs are represented in edge list format: 'Nodes: 1,2,3 Edges: (0,1), (2,3), ...' "
        "where Nodes lists all node IDs and Edges lists all edges as (u,v) pairs. "
    )
    
    task_instruction = "Given an undirected graph in this format, perform the requested reasoning task. Think step by step."
    
    return base_intro + format_desc + task_instruction

def _get_incident_prompt() -> str:
    """Generate universal Incident list system prompt."""
    
    base_intro = "You are a graph reasoning assistant. "
    
    format_desc = (
        "Graphs are represented in incident list format: 'This graph has nodes 0, 1, 2, ..., n. "
        "In this graph: Node 0 is connected to nodes 1, 2. Node 1 is connected to nodes 0, 3. ...' "
        "where each node's neighbors are explicitly listed. Isolated nodes are marked as 'Node X is isolated'. "
    )
    
    task_instruction = "Given an undirected graph in this format, perform the requested reasoning task. Think step by step."
    
    return base_intro + format_desc + task_instruction


# Example usage
if __name__ == "__main__":
    print("=== GraphVocab System Prompt ===\n")
    print(get_system_prompt("GraphVocab"))
    print("\n" + "="*50 + "\n")
    
    print("\n=== EdgeList System Prompt ===\n")
    print(get_system_prompt("EdgeList"))
    print("\n" + "="*50 + "\n")
    
    print("\n=== Incident System Prompt ===\n")
    print(get_system_prompt("Incident"))
