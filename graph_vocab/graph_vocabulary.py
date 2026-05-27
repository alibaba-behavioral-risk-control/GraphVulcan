import networkx as nx
import re
from itertools import combinations


class GraphVocabulary():
    def __init__(self):
        self.GRAPH_VOCAB = self.create_graph_vocabulary()
        self.GRAPH_STR_TOKENS = list(self.GRAPH_VOCAB.keys())
        # self.GRAPH_OP_DIFF_TOKEN = "<G_Operator_Diff>"
        # self.GRAPH_OP_MERGE_TOKEN = "<G_Operator_Merge>"
        self.GRAPH_OP_EQ_TOKEN = "<G_Operator_Eq>"
        self.NODE_ID_BEGIN_TOKEN = "<NidB>"
        self.NODE_ID_END_TOKEN = "<NidE>"
        self.NODE_ID_SPLIT_TOKEN = "<NidS>"
        self.GRAPH_CONNECT_TOKEN = "<G_Connect>"
        self.GRAPH_DISCONNECT_TOKEN = "<G_Disconnect>"
        self.GRAPH_UNKNOWN_TOKEN = "<G_Unknown>"
        self.GRAPH_TOKENS = self.get_graph_token()
        self.GRAPH_WL_HASH_MAP = self.get_weisfeiler_lehman_graph_hash_table()
        self.GRAPH_WL_HASH_TO_NAME = {v: k for k, v in self.GRAPH_WL_HASH_MAP.items()}
        self.NODE_WL_HASH_MAP = self.get_weisfeiler_lehman_node_hash_table()
    
    def get_graph_token(self):
        """
        Return all predefined graph tokens.
        """
        GRAPH_TOKENS = []
        GRAPH_TOKENS.extend(self.GRAPH_STR_TOKENS)
        # GRAPH_TOKENS.append(self.GRAPH_OP_DIFF_TOKEN)
        # GRAPH_TOKENS.append(self.GRAPH_OP_MERGE_TOKEN)
        GRAPH_TOKENS.append(self.GRAPH_OP_EQ_TOKEN)
        GRAPH_TOKENS.append(self.NODE_ID_BEGIN_TOKEN)
        GRAPH_TOKENS.append(self.NODE_ID_END_TOKEN)
        GRAPH_TOKENS.append(self.NODE_ID_SPLIT_TOKEN)
        GRAPH_TOKENS.append(self.GRAPH_CONNECT_TOKEN)
        GRAPH_TOKENS.append(self.GRAPH_DISCONNECT_TOKEN)
        GRAPH_TOKENS.append(self.GRAPH_UNKNOWN_TOKEN)
        return GRAPH_TOKENS

    def all_connected_graphlets_k(self, k):
        """
        List all connected non-isomorphic graphlets with k nodes.
        Args:
            k (int): Number of nodes in the graphlets.
        Returns:
            graphs (list)
        """
        nodes = list(range(k))
        all_edges = list(combinations(nodes, 2))

        graphs = []
        canon_labels = set()

        for r in range(1, len(all_edges) + 1):
            for edges in combinations(all_edges, r):
                G = nx.Graph()
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)
                if not nx.is_connected(G):
                    continue
                key = self.calculate_weisfeiler_lehman_graph_hash(G)
                if key in canon_labels:
                    continue
                canon_labels.add(key)
                graphs.append(G)

        return graphs

    def create_graph_vocabulary(self):
        vocab = {}
        # k=1
        g1 = nx.Graph()
        g1.add_node(0)
        vocab['<G1>'] = g1

        # k=2
        g2 = nx.Graph()
        g2.add_edge(0, 1)
        vocab['<G2_edge>'] = g2

        # k=3
        g3_path = nx.path_graph(3)  # 0-1-2
        g3_tri = nx.complete_graph(3)  # triangle
        vocab['<G3_path>'] = g3_path
        vocab['<G3_triangle>'] = g3_tri

        # k=4
        g4_path = nx.path_graph(4)  # P4
        g4_star = nx.star_graph(3)  # K1,3 (center=0, leaves=1,2,3)
        g4_cycle = nx.cycle_graph(4)  # C4
        # Diamond: K4 minus one edge
        g4_diamond = nx.complete_graph(4)
        g4_diamond.remove_edge(2, 3)
        # Paw: triangle + pendant edge
        g4_paw = nx.Graph()
        g4_paw.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3)])  # triangle 0-1-2 + edge 1-3
        # Complete graph K4
        g4_k4 = nx.complete_graph(4)

        vocab.update({
            '<G4_path>': g4_path,
            '<G4_star>': g4_star,
            '<G4_cycle>': g4_cycle,
            '<G4_diamond>': g4_diamond,
            '<G4_paw>': g4_paw,
            '<G4_k4>': g4_k4
        })

        g5_list = self.all_connected_graphlets_k(5)
        for i, G in enumerate(g5_list, start=1):
            vocab[f'<G5_{i}>'] = G


        return vocab

    def calculate_weisfeiler_lehman_node_hash(self, G: nx.Graph, n_iter=5):
        subgraph_hashes = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=n_iter)
        return subgraph_hashes

    def get_weisfeiler_lehman_node_hash_table(self):
        """
        Compute the Weisfeiler-Lehman node hash for graph G.

        Returns:
            dict: A mapping from node to its WL hash.
        """
        node_wl_hash_map = {}
        for name, g in self.GRAPH_VOCAB.items():
            node_wl_hash_map[name] = self.calculate_weisfeiler_lehman_node_hash(G=g)
        return node_wl_hash_map

    def calculate_weisfeiler_lehman_graph_hash(self, G: nx.Graph):
        wl_graph_hash = nx.weisfeiler_lehman_graph_hash(G, iterations=5)
        return wl_graph_hash

    def get_weisfeiler_lehman_graph_hash_table(self):
        """
        Compute the Weisfeiler-Lehman graph hash for graph G.

        Returns:
            str: The WL graph hash.
        """
        graph_wl_hash_map = {name: self.calculate_weisfeiler_lehman_graph_hash(g) for name, g in self.GRAPH_VOCAB.items()}
        return graph_wl_hash_map

    def wl_hash_match_with_mapping(self, G: nx.Graph, list_all_mappings=False):
        """
        Find the corresponding graphlet using Weisfeiler-Lehman hash and return the graph6 string and the mapping from graphlet node to G node

        Returns:
            tuple or None:
                - graph6_str: graphlet graph6 encoding
                - node_mapping_list: list，the mapping from graphlet node to G node
                If no match found, return None
        """
        wl_hash = self.calculate_weisfeiler_lehman_graph_hash(G)
        if wl_hash in self.GRAPH_WL_HASH_TO_NAME:
            template_name = self.GRAPH_WL_HASH_TO_NAME[wl_hash]
            template = self.GRAPH_VOCAB[template_name]
            gm = nx.algorithms.isomorphism.GraphMatcher(template, G)

            # Collect all mappings
            if gm.is_isomorphic():
                all_node_mapping_list = []
                sorted_graphlet_nodes = sorted(template.nodes())
                isomorphism_iter = gm.isomorphisms_iter()
                if list_all_mappings:
                    for mapping in isomorphism_iter:
                        node_mapping_list = [mapping[node] for node in sorted_graphlet_nodes]
                        all_node_mapping_list.append(node_mapping_list)
                else:
                    first_mapping = next(isomorphism_iter, None)
                    node_mapping_list = [first_mapping[node] for node in sorted_graphlet_nodes]
                    all_node_mapping_list = [node_mapping_list]
                return template_name, all_node_mapping_list
            else:
                error_msg = (
                    "Weisfeiler-Lehman hash matched but not isomorphic\n"
                    f"WL Hash:{wl_hash}"
                    "Graph Info:"
                )
                error_msg += str(G.edges) + "\n"
                print(error_msg)

        else:
            error_msg = (
                f"Weisfeiler-Lehman hash not found {wl_hash}\n"
            )
            print(error_msg)
        return self.GRAPH_UNKNOWN_TOKEN, list(G.nodes)


    def parse_graph_string(self, string: str):
        """
        Parse a string like '<NidB>22<NidS>57<NidS>71<NidS>28<NidE><G4_star>'
        Returns: (node_ids: List[int], graph_token: str)
        """
        # Extract everything between <NidB> and <NidE>
        match = re.search(r'<NidB>(.*?)<NidE>(<G[^>]+>)', string)
        if not match:
            raise ValueError(f"Invalid graph string format: {string}")
        
        ids_part = match.group(1)
        graph_token = match.group(2)

        # Split by <NidS>
        id_strs = ids_part.split('<NidS>')
        node_ids = [int(x) for x in id_strs]
        return graph_token, node_ids

    def graph_token_to_text(self, graph_token: str, node_list: list):
        """
        Convert a graph token and node list to a string representation.
        Example: ('<G4_star>', [22, 57, 71, 28]) -> '<NidB>22<NidS>57<NidS>71<NidS>28<NidE><G4_star>'
        """
        node_strs = [str(node) for node in node_list]
        ids_part = self.NODE_ID_SPLIT_TOKEN.join(node_strs)
        graph_string = f"{self.NODE_ID_BEGIN_TOKEN}{ids_part}{self.NODE_ID_END_TOKEN}{graph_token}"
        return graph_string
    
    def instantiate_graph_from_token(self, graph_token: str, node_list: list):
        if graph_token not in self.GRAPH_VOCAB:
            raise KeyError(f"Graph token '{graph_token}' not found in vocabulary.")
        
        base_graph = self.GRAPH_VOCAB[graph_token]
        original_nodes = sorted(base_graph.nodes())  # 通常是 [0, 1, 2, ...]
        
        if len(node_list) != len(original_nodes):
            raise ValueError(
                f"Length of node_list ({len(node_list)}) does not match "
                f"number of nodes in graph '{graph_token}' ({len(original_nodes)})."
            )

        node_mapping = dict(zip(original_nodes, node_list))

        new_graph = nx.relabel_nodes(base_graph, node_mapping, copy=True)
        
        return new_graph


if __name__ == "__main__":
    graph_vocab = GraphVocabulary()
    for item in graph_vocab.GRAPH_VOCAB.items():
        print(item)