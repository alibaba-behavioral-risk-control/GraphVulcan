import re
import networkx as nx
from graph_vocab.graph_tokenizer import GraphTokenizer


def verify_graph_computation_expressions(assistant_msg: str, no_eq_reward:float=0.0) -> float:
    graph_tokenizer = GraphTokenizer()
    # Pattern to match graph computation expressions:
    # <NidB>...<NidE><Gx_...> <G_Operator_Eq> <NidB>...<NidE><Gx_...> <G_Connect>/<G_Disconnect> <NidB>...<NidE><Gx_...>
    eq_token = graph_tokenizer.graph_vocab.GRAPH_OP_EQ_TOKEN
    connect_token = graph_tokenizer.graph_vocab.GRAPH_CONNECT_TOKEN
    disconnect_token = graph_tokenizer.graph_vocab.GRAPH_DISCONNECT_TOKEN

    # Pattern: left_expr <G_Operator_Eq> right_expr <G_Connect>/<G_Disconnect> ...
    graph_token_pattern = r'<NidB>.*?<NidE><G\d+_[^>]+>'

    # Find all occurrences of <G_Operator_Eq> and verify surrounding graph tokens
    eq_positions = [m.start() for m in re.finditer(re.escape(eq_token), assistant_msg)]

    correct_rate = 0.0
    if not eq_positions:
        # No graph computation expressions found
        return no_eq_reward
    else:
        total_expressions = 0
        correct_expressions = 0

        for eq_pos in eq_positions:
            total_expressions += 1
            # Extract text before and after <G_Operator_Eq>
            before_eq = assistant_msg[:eq_pos]
            after_eq = assistant_msg[eq_pos + len(eq_token):]

            # Find all graph tokens before <G_Operator_Eq>
            left_graphs = re.findall(graph_token_pattern, before_eq)
            if not left_graphs:
                # No graph token on left side
                continue

            # For left side: collect all consecutive graph tokens connected by <G_Connect> or <G_Disconnect>
            # Start from the last graph token and work backwards
            left_graph_tokens = [left_graphs[-1]]
            search_pos = before_eq.rfind(left_graphs[-1])

            # Look backwards for <G_Connect> or <G_Disconnect> tokens
            while search_pos > 0:
                # Check if there's a <G_Connect> or <G_Disconnect> token before this graph token
                prefix = before_eq[:search_pos].rstrip()
                found_connector = False
                
                # Check for <G_Connect>
                if prefix.endswith(connect_token):
                    prefix = prefix[:-len(connect_token)].rstrip()
                    found_connector = True
                # Check for <G_Disconnect>
                elif prefix.endswith(disconnect_token):
                    prefix = prefix[:-len(disconnect_token)].rstrip()
                    found_connector = True
                
                if found_connector:
                    # Find the graph token before the connector
                    prev_graphs = re.findall(graph_token_pattern, prefix)
                    if prev_graphs:
                        left_graph_tokens.insert(0, prev_graphs[-1])
                        search_pos = prefix.rfind(prev_graphs[-1])
                    else:
                        break
                else:
                    break

            # For right side: collect all consecutive graph tokens connected by <G_Connect> or <G_Disconnect>
            # Start from the first graph token and work forwards
            right_graphs = re.findall(graph_token_pattern, after_eq)
            if not right_graphs:
                # No graph token on right side
                continue

            right_graph_tokens = [right_graphs[0]]
            search_pos = after_eq.find(right_graphs[0]) + len(right_graphs[0])

            # Look forwards for <G_Connect> or <G_Disconnect> tokens
            while search_pos < len(after_eq):
                # Check if there's a <G_Connect> or <G_Disconnect> token after this graph token
                suffix = after_eq[search_pos:].lstrip()
                found_connector = False
                
                # Check for <G_Connect>
                if suffix.startswith(connect_token):
                    suffix = suffix[len(connect_token):].lstrip()
                    found_connector = True
                # Check for <G_Disconnect>
                elif suffix.startswith(disconnect_token):
                    suffix = suffix[len(disconnect_token):].lstrip()
                    found_connector = True
                
                if found_connector:
                    # Find the graph token after the connector
                    next_graphs = re.findall(graph_token_pattern, suffix)
                    if next_graphs:
                        right_graph_tokens.append(next_graphs[0])
                        search_pos = after_eq.find(next_graphs[0], search_pos) + len(next_graphs[0])
                    else:
                        break
                else:
                    break

            # Decode and combine graph tokens on both sides
            # Decode left side graphs and combine them
            left_G = nx.Graph()
            for token in left_graph_tokens:
                g = graph_tokenizer.decode_graph_vocab(token)
                left_G = nx.compose(left_G, g)

            # Decode right side graphs and combine them
            right_G = nx.Graph()
            for token in right_graph_tokens:
                g = graph_tokenizer.decode_graph_vocab(token)
                right_G = nx.compose(right_G, g)

            # Check if the two graphs are identical
            if nx.utils.graphs_equal(left_G, right_G):
                correct_expressions += 1
            # else:
            #     print(f"Found incorrect graph match:")
            #     print(f"Left side: {left_graph_tokens}")
            #     print(f"Right side: {right_graph_tokens}")
    
    correct_rate = correct_expressions / total_expressions if total_expressions > 0 else no_eq_reward
    # if correct_rate != 1.0:
    #     print(f"Found {correct_expressions} correct expressions out of {total_expressions}")
    return correct_rate