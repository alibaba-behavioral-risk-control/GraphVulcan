import networkx as nx
from networkx.utils import graphs_equal
from typing import List, Dict
import re

try:
    from graph_vocab.graph_vocabulary import GraphVocabulary
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph_vocab.graph_vocabulary import GraphVocabulary



class GraphTokenizer:
    def __init__(self, seed_strategy: str = "clustering"):
        """
        Args:
            seed_strategy: Strategy for selecting seed nodes in greedy+wl tokenization.
                - "clustering": Use clustering coefficient (deterministic, slower).
                - "random": Use random selection (faster, non-deterministic).
        """
        self.graph_vocab = GraphVocabulary()
        self.seed_strategy = seed_strategy

    def tokenize(self, G: nx.Graph, strategy: str = "greedy+wl", max_size: int = 5) -> List[Dict]:
        """
        Decompose G into edge-disjoint connected subgraphs with <=4 nodes.
        Each subgraph is represented as a token.
        """
        if G.number_of_edges() == 0:
            # Only isolated nodes → use <G1>
            return [{"token": "<G1>", "node_ids": [v]} for v in G.nodes()]
        
        result = []
        G_components = nx.connected_components(G)
        for N_conn in G_components:
            G_conn = G.subgraph(N_conn)
            remaining_graph = G_conn.copy()
            
            # Detect isolated nodes
            isolated = list(nx.isolates(remaining_graph))
            if isolated:
                for node in isolated:
                    result.append({
                        "token": "<G1>",
                        "node_ids": [node]
                    })
            remaining_graph.remove_nodes_from(isolated)
            # Process until no edges left
            while remaining_graph.number_of_edges() > 0:
                if strategy == "random":
                    sub_nodes = self.bfs_grow_subgraph_random(remaining_graph, max_size=max_size)
                elif strategy == "greedy":
                    sub_nodes = self.bfs_grow_subgraph_greedy(remaining_graph, max_size=max_size)
                elif strategy == "greedy+wl":
                    sub_nodes = self.bfs_grow_subgraph_greedy_wl(remaining_graph, max_size=max_size)

                subG = remaining_graph.subgraph(sub_nodes).copy()

                assert nx.is_connected(subG), "Subgraph must be a connected component"
                # Match to vocabulary
                match = self.graph_vocab.wl_hash_match_with_mapping(subG)
                if match:
                    token, mappings = match
                    node_list = mappings[0]
                else:
                    raise ValueError("Failed to match subgraph to vocabulary")
                

                result.append({
                    "token": token,
                    "node_ids": node_list
                })

                # Remove all edges of this subgraph from remaining_graph
                remaining_graph.remove_edges_from(subG.edges())
                
                # Remove nodes that became isolated after edge removal
                # This prevents infinite loops where isolated nodes are selected as start nodes
                newly_isolated = [node for node in sub_nodes if remaining_graph.degree(node) == 0]
                remaining_graph.remove_nodes_from(newly_isolated)
                
                if remaining_graph.number_of_edges() > 0: # if remaining_graph is not empty
                    result.append({
                        "token": self.graph_vocab.GRAPH_CONNECT_TOKEN,
                        "node_ids": []
                    })
            result.append({
                "token": self.graph_vocab.GRAPH_DISCONNECT_TOKEN,
                "node_ids": []
            })

        return result

    def bfs_grow_subgraph_random(self, G: nx.Graph, max_size: int = 5):
        """
        Grow a connected subgraph up to `max_size` nodes by randomly
        selecting a start node and randomly expanding to neighbors.
        No greedy heuristic — purely random baseline for ablation.
        """
        import random
        start = random.choice(list(G.nodes()))
        visited = {start}

        while len(visited) < max_size:
            candidates = set()
            for v in visited:
                candidates.update(G.neighbors(v))
            candidates -= visited

            if not candidates:
                break

            chosen = random.choice(list(candidates))
            visited.add(chosen)

        return list(visited)

    def bfs_grow_subgraph_greedy(self, G: nx.Graph, max_size: int = 5):
        """
        Grow a connected subgraph from `start` up to `max_size` nodes,
        preferring neighbors that add more edges (greedy dense expansion).
        """

        degs = dict(G.degree())
        start = max(degs, key=degs.get)
        visited = {start}
        # queue = deque([start])
        while len(visited) < max_size:
            # u = queue.popleft()
            # Get neighbors not yet in subgraph
            candidates = []
            for i in visited:
                neighbors_i = [v for v in G.neighbors(i) if v not in visited]
                candidates.extend(neighbors_i)
            candidates = list(set(candidates))

            if not candidates:
                break

            # Sort candidates by number of edges they bring into current subgraph
            # (degree within visited ∪ {candidate})
            def score(v):
                metric = [1 for w in visited if G.has_edge(v, w)]
                return sum(metric)

            candidates.sort(key=score, reverse=True)

            # Add as many as needed to reach max_size
            v = candidates[0]
            visited.add(v)
        return list(visited)


    def bfs_grow_subgraph_greedy_wl(self, G: nx.Graph, max_size: int = 5):
        """
        Grow a connected subgraph up to `max_size` nodes.
        Uses structural features (WL colors + local subgraph hash) to ensure
        consistent ordering across isomorphic graphs, even with random node IDs.
        """
        if G.number_of_nodes() == 0:
            return []

        node_signatures = self._weisfeiler_lehman_colors(G, iterations=5)

        if self.seed_strategy == "random":
            import random
            node_priority = {v: random.random() for v in G.nodes()}
        else:
            node_priority = nx.clustering(G)

        start_candidates = sorted(
            G.nodes(),
            key=lambda v: (node_priority[v], node_signatures[v]), reverse=True
        )
        start = start_candidates[0]

        visited = {start}

        while len(visited) < max_size:
            candidates = set()
            for v in visited:
                candidates.update(G.neighbors(v))
            candidates -= visited
            candidates = list(candidates)

            if not candidates:
                break

            def edge_score(v):
                return sum(1 for w in visited if G.has_edge(v, w))

            candidates.sort(
                key=lambda v: (
                    edge_score(v),
                    node_priority[v],
                    node_signatures[v]
                ),
                reverse=True,
            )

            best = candidates[0]
            visited.add(best)

        return sorted(visited, key=lambda v: node_signatures[v])

    def _weisfeiler_lehman_colors(self, G: nx.Graph, iterations: int = 2) -> dict:
        """Return WL colors after `iterations` rounds using NetworkX's WL hash.
        """
        wl_hashes = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=iterations, digest_size=16)

        colors = {node: hashes[-1] if hashes else str(G.degree(node)) for node, hashes in wl_hashes.items()}

        return colors

    def token_list_to_text(self, tokenized: List[Dict], mark_connected_components:bool=True, mark_last_disconnect:bool=True) -> str:
        """Format as: <NidB>n1<NidS>n2<...><NidE><Token>"""
        parts = []
        for item in tokenized:
            if (item["token"] == self.graph_vocab.GRAPH_CONNECT_TOKEN or item["token"] == self.graph_vocab.GRAPH_DISCONNECT_TOKEN):
                if mark_connected_components:
                    part = item["token"]
                    parts.append(part)
            else:
                node_str = "<NidS>".join(str(n) for n in item["node_ids"])
                part = f"<NidB>{node_str}<NidE>{item['token']}"
                parts.append(part)
        
        # Remove last disconnect token if mark_last_disconnect is False
        if not mark_last_disconnect and parts and parts[-1] == self.graph_vocab.GRAPH_DISCONNECT_TOKEN:
            parts.pop()
        
        return " ".join(parts)

    def token_list_to_graph(self, tokenized: List[Dict]):
        """Reconstruct graph from token list."""
        G = nx.Graph()
        for item in tokenized:
            token = item["token"]
            nodes = item["node_ids"]
            if token == self.graph_vocab.GRAPH_CONNECT_TOKEN or token == self.graph_vocab.GRAPH_DISCONNECT_TOKEN:
                continue
            subG = self.graph_vocab.instantiate_graph_from_token(token, nodes)
            G.add_nodes_from(subG.nodes())
            G.add_edges_from(subG.edges())
        return G

    def decode_graph_vocab(self, text: str) -> nx.Graph:
        """Parse a text representation of a graph into a networkx graph."""
        graph_strings = re.findall(r'(<NidB>.*?<NidE><G[^>]+>)', text)
        G = nx.Graph()
        for graph_str in graph_strings:
            token, nodes = self.graph_vocab.parse_graph_string(graph_str)
            subG = self.graph_vocab.instantiate_graph_from_token(token, nodes)
            G.add_nodes_from(subG.nodes())
            G.add_edges_from(subG.edges())
        return G

    def encode_graph_vocab(self, G: nx.Graph, mark_connected_components=True, mark_last_disconnect=True, max_size=5) -> str:
        """Format a networkx graph into a text representation."""
        token_list = self.tokenize(G, strategy="greedy+wl", max_size=max_size)
        text = self.token_list_to_text(token_list, mark_connected_components=mark_connected_components, mark_last_disconnect=mark_last_disconnect)
        return text

    def encode_edge_list(self, G: nx.Graph) -> str:
        """
        Encode an undirected graph G as a plain edge list string.
        New format: "Nodes: 1,2,3 Edges: (0,1), (2,3)"
        - Nodes are always listed (to preserve isolates)
        - Edges sorted lexicographically, each as (u,v) with u<=v
        """
        nodes = sorted(G.nodes())
        edges = sorted([tuple(sorted(e)) for e in G.edges()])
        nodes_part = ", ".join(str(n) for n in nodes)
        edges_part = ", ".join(f"({u},{v})" for u, v in edges)
        return f"Nodes: {nodes_part} Edges: {edges_part}"

    def decode_edge_list(self, text: str) -> nx.Graph:
        """
        Decode an edge list string produced by encode_edge_list back to a graph.
        """
        # Expected rough pattern: "Nodes: ... Edges: ..."
        if "Nodes:" not in text or "Edges:" not in text:
            raise ValueError("Invalid edge list encoding: expected 'Nodes: ... Edges: ...' format")

        try:
            prefix, edges_part = text.split("Edges:", 1)
            _, nodes_part = prefix.split("Nodes:", 1)
        except ValueError:
            raise ValueError("Invalid edge list encoding: cannot split nodes/edges")

        G = nx.Graph()
        # Add nodes (handle empty list)
        nodes_part = nodes_part.strip()
        if nodes_part:
            for n_str in nodes_part.split(","):
                n_str = n_str.strip()
                if n_str:
                    G.add_node(int(n_str))

        # Add edges
        edges_part = edges_part.strip()
        if edges_part:
            # Use regex to robustly find all occurrences of "(u,v)" allowing spaces
            # This avoids splitting on commas which can break '(u,v), (x,y)'
            edge_matches = re.findall(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)", edges_part)
            if not edge_matches:
                # If no matches found but edges_part is non-empty, raise informative error
                raise ValueError(f"Invalid edge list format, no edge tuples found in: '{edges_part}'")
            for u_str, v_str in edge_matches:
                u = int(u_str)
                v = int(v_str)
                G.add_edge(u, v)
        return G

    def encode_incident(self, G: nx.Graph) -> str:
        """
        Encode a graph G in incident list format.
        
        Format: "This graph has nodes 0, 1, 2, ..., n. 
                 In this graph: Node 0 is connected to nodes 1, 2. 
                 Node 1 is connected to nodes 0, 3. ..."
        
        Args:
            G: NetworkX graph to encode
            
        Returns:
            String representation in incident list format
        """
        if G.number_of_nodes() == 0:
            return "This graph has no nodes."
        
        nodes = sorted(G.nodes())
        
        # Build node list part - simple comma-separated list
        nodes_str = ", ".join(str(n) for n in nodes)
        
        # Build incident list part
        incident_parts = []
        for node in nodes:
            neighbors = sorted(G.neighbors(node))
            if len(neighbors) == 0:
                # Isolated node
                incident_parts.append(f"Node {node} is isolated")
            elif len(neighbors) == 1:
                incident_parts.append(f"Node {node} is connected to node {neighbors[0]}")
            else:
                # Multiple neighbors - use comma-separated list
                neighbors_str = ", ".join(str(n) for n in neighbors)
                incident_parts.append(f"Node {node} is connected to nodes {neighbors_str}")
        
        incident_str = ". ".join(incident_parts) + "."
        
        return f"This graph has nodes {nodes_str}. In this graph: {incident_str}"

    def decode_incident(self, text: str) -> nx.Graph:
        """
        Decode an incident list format string back to a NetworkX graph.
        
        Expected format: "This graph has nodes 0, 1, 2, ..., n. 
                         In this graph: Node 0 is connected to nodes 1, 2. 
                         Node 1 is connected to nodes 0, 3. ..."
        
        Args:
            text: String in incident list format
            
        Returns:
            NetworkX graph
        """
        if not text.startswith("This graph"):
            raise ValueError("Invalid incident format: must start with 'This graph'")
        
        # Handle empty graph
        if "no nodes" in text.lower():
            return nx.Graph()
        
        G = nx.Graph()
        
        # Extract nodes from the first part
        # Pattern: "This graph has nodes 0, 1, 2, 3"
        nodes_match = re.search(r'has nodes\s+([\d\s,]+?)\.', text)
        if nodes_match:
            nodes_str = nodes_match.group(1)
            # Extract all numbers from the nodes string
            node_ids = [int(n) for n in re.findall(r'\d+', nodes_str)]
            G.add_nodes_from(node_ids)
        
        # Extract incident information
        # Pattern: "Node X is connected to node(s) Y, Z" or "Node X is isolated"
        if "In this graph:" in text:
            incident_part = text.split("In this graph:", 1)[1]
            
            # Find all node connection statements
            # Match patterns like:
            # - "Node 0 is connected to nodes 1, 2"
            # - "Node 1 is connected to node 3"
            # - "Node 2 is isolated"
            node_statements = re.findall(
                r'Node\s+(\d+)\s+is\s+(?:connected to|isolated)',
                incident_part
            )
            
            for node_id_str in node_statements:
                node_id = int(node_id_str)
                
                # Find the full statement for this node
                # Updated pattern to match comma-separated list without "and"
                pattern = rf'Node\s+{node_id}\s+is\s+connected to\s+nodes?\s+([\d\s,]+?)(?:\.|$)'
                match = re.search(pattern, incident_part)
                
                if match:
                    neighbors_str = match.group(1)
                    # Extract all neighbor IDs
                    neighbor_ids = [int(n) for n in re.findall(r'\d+', neighbors_str)]
                    
                    # Add edges
                    for neighbor_id in neighbor_ids:
                        G.add_edge(node_id, neighbor_id)
        
        return G
        

    def validate_tokenization(self, G, tokens):
        all_edges_in_G = {frozenset(e) for e in G.edges()}
        all_nodes_in_G = set(G.nodes())
        covered_edges = set()
        total_nodes = set()

        for item in tokens:
            token = item["token"]
            nodes = item["node_ids"]
            if token == self.graph_vocab.GRAPH_CONNECT_TOKEN or token == self.graph_vocab.GRAPH_DISCONNECT_TOKEN:
                continue
            k = len(nodes)
            
            # reconstruct subgraph
            sub_graph = self.graph_vocab.instantiate_graph_from_token(token, nodes)
            sub_edges = set(frozenset(e) for e in sub_graph.edges())
            
            # has edge overlap?
            assert not (sub_edges & covered_edges), "Edge overlap detected!"
            covered_edges |= sub_edges

            # is connected component?
            if k > 1:
                subG = G.subgraph(nodes)
                assert nx.is_connected(subG), f"Subgraph {nodes} is not connected!"

            total_nodes.update(nodes)

        # check if all edges are covered
        assert covered_edges == all_edges_in_G, f"Missing edges: {all_edges_in_G - covered_edges}"

        # check if all isolated nodes are covered
        isolated_in_G = set(nx.isolates(G))
        isolated_in_tokens = {item["node_ids"][0] for item in tokens if len(item["node_ids"]) == 1}
        assert isolated_in_tokens == isolated_in_G, "Isolated nodes mismatch!"

        print("Token list Validation passed!")

    def validate(self, G: nx.Graph, *, name: str = "", verbose: bool = True):
        if name:
            print(f"\n--- {name} ---")
        token_list = self.tokenize(G)
        text = self.token_list_to_text(token_list)

        if verbose:
            print("Tokens:", token_list)
            print("Text:", text)

        self.validate_tokenization(G, token_list)

        reconstructed = self.decode_graph_vocab(text)
        assert graphs_equal(G, reconstructed), "Graph reconstruction failed!"
        print("Graph reconstruction passed ")
        return token_list, text


if __name__ == "__main__":

    # tokenizer init
    tokenizer = GraphTokenizer()

    G0 = nx.complete_graph(20)
    tokenizer.validate(G0, name="Test 0: Large complete graph")

    # Test Case 1: Single node (isolated)
    G1 = nx.Graph()
    G1.add_node(5)
    tokenizer.validate(G1, name="Test 1: Single isolated node")

    # Test Case 2: Single edge
    G2 = nx.Graph()
    G2.add_edge(10, 20)
    tokenizer.validate(G2, name="Test 2: Single edge")

    # Test Case 3: Triangle (should match <G3_triangle>)
    G3 = nx.complete_graph(3)
    nx.relabel_nodes(G3, {0: 100, 1: 200, 2: 300}, copy=False)
    tokenizer.validate(G3, name="Test 3: Triangle")

    # Test Case 4: Paw graph (triangle + pendant)
    G4 = nx.Graph()
    G4.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3)])  # triangle 0-1-2 + edge 1-3
    tokenizer.validate(G4, name="Test 4: Paw graph")

    # Test Case 5: Path of 5 nodes (P5) → should split into P4 + edge or two P3s etc.
    G5 = nx.path_graph(5)
    nx.relabel_nodes(G5, {i: i * 10 for i in range(5)}, copy=False)  # [0,10,20,30,40]
    tokenizer.validate(G5, name="Test 5: Path P5")

    # Test Case 6: Cycle C5 (5-cycle) – no perfect k<=4 cover, must split
    G6 = nx.cycle_graph(5)
    nx.relabel_nodes(G6, {i: i + 100 for i in range(5)}, copy=False)
    tokenizer.validate(G6, name="Test 6: Cycle C5")

    # Test Case 7: Star K1,4 (center + 4 leaves) → one <G4_star> if matched
    G7 = nx.star_graph(4)  # center=0, leaves=1,2,3,4 → 5 nodes! So cannot be one token.
    tokenizer.validate(G7, name="Test 7: Star K1,4")

    # Test Case 8: Complete graph K4
    G8 = nx.complete_graph(4)
    nx.relabel_nodes(G8, {i: i + 50 for i in range(4)}, copy=False)
    tokenizer.validate(G8, name="Test 8: Complete graph K4")

    # Test Case 9: Disconnected graph (triangle + isolated + edge)
    G9 = nx.Graph()
    G9.add_edges_from([(1,2), (2,3), (3,1)])  # triangle
    G9.add_edge(10, 11)                       # edge
    G9.add_node(99)                           # isolated
    G9.add_node(100)
    tokenizer.validate(G9, name="Test 9: Disconnected graph")

    G10 = nx.gnm_random_graph(10, 30)
    tokenizer.validate(G10, name="Test 10: Random nm graph")
    

    print("\n All tests passed!")
