import os
import json
import time
import networkx as nx
from critic_agent import analyze_knowledge_graph_depth  # assumes depth eval for KG
from actor_agent import create_thread, create_dynamic_agent, add_user_message, get_response, delete_agent

INVENTORY_DIR = "inventory_graphs"
USER_ID = "user_123"
AGENT_NAME = "Cool Assistant"

def save_graph(graph, filepath):
    data = nx.node_link_data(graph)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def add_new_inventory_graph(base_elements, FLAG):
    if not FLAG:
        create_thread(USER_ID)
        create_dynamic_agent(USER_ID, AGENT_NAME)
        FLAG = True

    prompt = (
        f"Using the base elements: {base_elements}, generate a knowledge graph of valid fusions. "
        "Each node should represent a unique elemental or derived material. "
        "Each edge should represent a fusion event with label 'fused_with'. "
        "Use semantic constraints to prevent meaningless fusions. "
        "Output only a list of triples: (source, fused_with, result)"
    )
    add_user_message(USER_ID, prompt)
    reply = get_response(USER_ID)

    try:
        triples = json.loads(reply)
        print("[INFO] Parsed KG triples successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to parse response: {e}")
        return None, FLAG

    G = nx.DiGraph()
    for source, target, result in triples:
        G.add_node(source)
        G.add_node(target)
        G.add_node(result)
        G.add_edge(source, result, relation="fused_with")
        G.add_edge(target, result, relation="fused_with")

    return G, FLAG

def merge_graph(graph, inventory_dir):
    if not os.path.exists(inventory_dir):
        os.makedirs(inventory_dir)

    # Save with timestamp for historical tracking
    timestamp_filename = f"graph_{int(time.time())}.json"
    timestamp_filepath = os.path.join(inventory_dir, timestamp_filename)
    save_graph(graph, timestamp_filepath)

    # Save latest graph to fixed filename for easy reference
    latest_filepath = os.path.join(inventory_dir, "graph_latest.json")
    save_graph(graph, latest_filepath)

    print(f"[INFO] Graph saved to {timestamp_filepath}")
    print(f"[INFO] Also updated: {latest_filepath}")


def main(max_iterations=10):
    base_elements = ["air", "water", "fire", "earth"]
    FLAG = False
    seen_nodes = set(base_elements)

    for i in range(max_iterations):
        print(f"\n=== Iteration {i+1} ===")
        graph, FLAG = add_new_inventory_graph(base_elements, FLAG)

        if not graph:
            break

        merge_graph(graph, INVENTORY_DIR)
        depth, new_nodes = analyze_knowledge_graph_depth(graph, seen_nodes)

        if not new_nodes:
            print("[INFO] Stopping: no new fusions discovered.")
            break

        base_elements += new_nodes
        base_elements = list(set(base_elements))
        seen_nodes.update(new_nodes)

    delete_agent(USER_ID)

if __name__ == "__main__":
    main()
