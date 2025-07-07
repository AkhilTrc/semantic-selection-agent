import json
import os
import traceback
import random
from glob import glob

def collect_all_fusion_elements(tree):
    """
    Recursively collects all fusion elements from a tree.
    Returns a flat list of names (or any identifying attribute).
    """
    elements = []

    def traverse(node):
        if not isinstance(node, dict):
            return
        if "name" in node:
            elements.append(node["name"])
        for child in node.get("children", []):
            traverse(child)

    if isinstance(tree, list):
        for item in tree:
            traverse(item)
    elif isinstance(tree, dict):
        traverse(tree)

    return elements

def process_json_files(directory):
    """
    Processes the most recent JSON file in a directory,
    collects all fusion elements, and selects one randomly.
    """
    json_files = glob(os.path.join(directory, '**', '*.json'), recursive=True)
    if not json_files:
        print("No JSON files found.")
        return None

    json_files.sort(key=os.path.getmtime)
    json_files = [json_files[-1]]  # Only use the most recently modified file

    for file_path in json_files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                all_elements = collect_all_fusion_elements(data)
                print(f"File: {file_path} - Elements: {all_elements}")

                if not all_elements:
                    print("No fusion elements found.")
                    return None

                random_selection = random.choice(all_elements)
                print(f"Randomly selected fusion element: {random_selection}")
                return random_selection

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                traceback.print_exc()
                return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python critic.py <directory>")
        sys.exit(1)
    target_directory = sys.argv[1]
    process_json_files(target_directory)
