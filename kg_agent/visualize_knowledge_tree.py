import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph

def load_graph(filepath):
    """Load a NetworkX graph from a JSON node-link file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)

def get_root_nodes(graph):
    """Identify nodes with no incoming edges — candidates for tree roots."""
    return [n for n in graph.nodes if graph.in_degree(n) == 0]

def draw_knowledge_tree(graph, root=None):
    """
    Visualize a knowledge graph in a tree layout from a root node.
    If no root is provided, the first zero in-degree node is used.
    """
    if root is None:
        roots = get_root_nodes(graph)
        if not roots:
            raise ValueError("No root node found (no node with in-degree 0).")
        root = roots[0]

    # Build a tree by traversing the graph from the root
    tree = nx.DiGraph()
    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for _, child, data in graph.out_edges(node, data=True):
            label = data.get("relation", "")
            tree.add_edge(node, child, relation=label)
            dfs(child)

    dfs(root)

    # Compute tree layout
    pos = hierarchy_pos(tree, root)
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in tree.edges(data=True)}

    plt.figure(figsize=(14, 8))
    nx.draw(tree, pos, with_labels=True, node_color='lightblue', node_size=1500, arrows=True, font_size=10)
    nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_color="gray")
    plt.title(f"Knowledge Graph Tree (root: {root})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Position nodes in a hierarchy layout (tree-style).
    Credit: Joel’s answer on StackOverflow (adapted).
    """
    def _hierarchy_pos(G, root, leftmost, width, vert_gap, vert_loc, xcenter, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        children = list(G.successors(root))
        if not children:
            return pos
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos[child] = (nextx, vert_loc - vert_gap)
            pos = _hierarchy_pos(G, child, leftmost, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_gap, vert_loc, xcenter)

# ====== Usage Example ======
if __name__ == "__main__":
    GRAPH_FILE = "inventory_graphs/graph_latest.json"  # <-- Adjust path
    if not os.path.exists(GRAPH_FILE):
        print(f"[ERROR] Graph file not found: {GRAPH_FILE}")
    else:
        G = load_graph(GRAPH_FILE)
        draw_knowledge_tree(G)
