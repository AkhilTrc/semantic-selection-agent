import os
import json
import networkx as nx
from typing import Set, Tuple, List

def load_graph_from_file(filepath: str) -> nx.DiGraph:
    """Load a graph from a JSON file using node-link format."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)

def analyze_knowledge_graph_depth(graph: nx.DiGraph, known_nodes: Set[str]) -> Tuple[int, List[str]]:
    """
    Analyze the depth of the knowledge graph and identify newly discovered nodes.
    
    Args:
        graph (nx.DiGraph): The knowledge graph.
        known_nodes (set): Previously seen node names.
    
    Returns:
        Tuple[int, List[str]]: Max depth of the graph, and list of new nodes discovered.
    """
    # Find new nodes
    new_nodes = [n for n in graph.nodes if n not in known_nodes]

    if not new_nodes:
        return 0, []

    max_depth = 0
    for node in new_nodes:
        try:
            # BFS shortest paths from this new node to all others
            lengths = nx.single_source_shortest_path_length(graph, node)
            depth_from_node = max(lengths.values(), default=0)
            if depth_from_node > max_depth:
                max_depth = depth_from_node
        except Exception as e:
            print(f"[WARN] Could not evaluate depth from node {node}: {e}")

    return max_depth, new_nodes

def process_graph_directory(directory: str, known_nodes: Set[str]) -> Tuple[int, List[str]]:
    """
    Load all graph files in the directory and analyze them.
    
    Args:
        directory (str): Path to directory containing saved graph JSONs.
        known_nodes (set): Previously seen nodes.

    Returns:
        Tuple[int, List[str]]: Highest depth seen, and all new nodes discovered.
    """
    max_depth_overall = 0
    all_new_nodes = []

    if not os.path.exists(directory):
        print(f"[ERROR] Directory '{directory}' does not exist.")
        return 0, []

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue
        filepath = os.path.join(directory, file)
        try:
            graph = load_graph_from_file(filepath)
            depth, new_nodes = analyze_knowledge_graph_depth(graph, known_nodes)
            max_depth_overall = max(max_depth_overall, depth)
            all_new_nodes.extend(new_nodes)
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

    return max_depth_overall, list(set(all_new_nodes))
