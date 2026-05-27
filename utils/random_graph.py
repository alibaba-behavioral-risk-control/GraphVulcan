import networkx as nx
import random


class NLGraphGenerator:
    def __init__(self,num_of_nodes = 10, edge_probability = 0.2):
        self.num_of_nodes = num_of_nodes
        self.edge_probability = edge_probability

    def generate_graph(self):
        idx = list(range(self.num_of_nodes))
        random.shuffle(idx)
        G = nx.Graph()
        G.add_nodes_from(range(self.num_of_nodes))
        for u in list(G.nodes()):
            for v in list(G.nodes()):
                if u < v and random.random() < self.edge_probability:
                    G.add_edge(idx[u], idx[v])
        return G

def generate_NLGraph_random_graph(num_nodes):
    """
    Generate a random graph and return its NetworkX graph object.
    """
    generator = NLGraphGenerator(num_of_nodes=num_nodes, edge_probability=0.2)
    G = generator.generate_graph()
    return G

def generate_nm_random_graph(n_nodes, num_edges=None):
    if num_edges is None:
        max_edges = n_nodes * (n_nodes - 1) // 2
        min_edges = n_nodes - 1
        num_edges = random.randint(min_edges, max_edges)
    G = nx.gnm_random_graph(n_nodes, num_edges)
    return G

def generate_erdos_renyi_graph(n_nodes, p=0.2):
    G = nx.erdos_renyi_graph(n_nodes, p)
    return G


def generate_random_graph(min_nodes, max_nodes):
    """
    Generate a random graph with number of nodes between min_nodes and max_nodes.
    Uses various graph generation strategies for diversity.
    """
    n_nodes = random.randint(min_nodes, max_nodes)
    
    # Randomly choose graph generation strategy
    strategy = random.choice([
        'erdos_renyi',
        'barabasi_albert',
        'watts_strogatz',
        'random_tree',
        'complete',
        'path',
        'cycle',
        'star'
    ])
    
    try:
        if strategy == 'erdos_renyi':
            p = random.uniform(0.1, 0.5)
            G = nx.erdos_renyi_graph(n_nodes, p)
        elif strategy == 'barabasi_albert':
            m = max(1, min(3, n_nodes // 3))
            G = nx.barabasi_albert_graph(n_nodes, m)
        elif strategy == 'watts_strogatz':
            k = max(2, min(4, n_nodes // 2))
            p = random.uniform(0.1, 0.5)
            G = nx.watts_strogatz_graph(n_nodes, k, p)
        elif strategy == 'random_tree':
            G = nx.random_tree(n_nodes)
        elif strategy == 'complete':
            G = nx.complete_graph(min(n_nodes, 10))  # Limit complete graphs
        elif strategy == 'path':
            G = nx.path_graph(n_nodes)
        elif strategy == 'cycle':
            if n_nodes >= 3:
                G = nx.cycle_graph(n_nodes)
            else:
                G = nx.path_graph(n_nodes)
        elif strategy == 'star':
            if n_nodes >= 2:
                G = nx.star_graph(n_nodes - 1)
            else:
                G = nx.Graph()
                G.add_node(0)
        
        # Relabel nodes with random integers for better diversity
        node_labels = random.sample(range(1, 100), n_nodes)
        mapping = {old: new for old, new in zip(G.nodes(), node_labels)}
        G = nx.relabel_nodes(G, mapping, copy=True)
        
        return G
    except:
        # Fallback to simple graph
        G = nx.Graph()
        for i in range(n_nodes):
            G.add_node(i)
        for i in range(min(n_nodes - 1, random.randint(0, n_nodes))):
            if i + 1 < n_nodes:
                G.add_edge(i, i + 1)
        return G


def random_relabel(G: nx.Graph) -> nx.Graph:
    # labels = random.sample(range(10_000, 10_000_000), len(nodes))
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    random_labels = []
    for i in range(n_nodes):
        while True:
            digit = random.randint(1, 3)
            max_label = 10 ** digit - 1
            min_label = 0 if digit == 1 else 10 ** (digit - 1)
            random_label = random.sample(range(min_label, max_label), 1)
            if random_label[0] not in random_labels:
                # loop until find a unique label
                random_labels.append(random_label[0])
                break
    new_G = nx.relabel_nodes(G, {node: random_labels[i] for i, node in enumerate(nodes)}, copy=True)
    return new_G
    # return {node: random_labels[i] for i, node in enumerate(nodes)}