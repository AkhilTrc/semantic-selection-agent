import json
from glob2 import glob
import os

def calculate_tree_depth(node):
    """
    Recursively calculates the depth of a tree structure where each node
    contains a 'children' array of descendant nodes.
    """
    if not isinstance(node, dict):
        return 0
    max_depth = 0
    for child in node.get('children', []):
        current_depth = calculate_tree_depth(child)
        max_depth = max(max_depth, current_depth)
    return 1 + max_depth

def process_json_files(directory):
    """
    Processes all JSON files in a directory (including subdirectories),
    calculates their maximum tree depth, and identifies the deepest structure.
    """
    json_files = glob(os.path.join(directory, '**/*.json'))
    max_depth = 0
    deepest_files = []
    results = {}

    for file_path in json_files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                current_depth = calculate_tree_depth(data)
                results[file_path] = current_depth
                print(f"File: {file_path} - Depth: {current_depth}")

                # Track deepest files
                if current_depth > max_depth:
                    max_depth = current_depth
                    deepest_files = [file_path]
                elif current_depth == max_depth:
                    deepest_files.append(file_path)
            except json.JSONDecodeError:
                print(f"Invalid JSON in {file_path}, skipping.")
                continue

    # Print final results
    print("\nDeepest tree structure(s):")
    for file in deepest_files:
        print(f"File: {file} - Depth: {max_depth}")
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python critic.py json_trees")
        sys.exit(1)
    target_directory = sys.argv[1]
    process_json_files(target_directory)
