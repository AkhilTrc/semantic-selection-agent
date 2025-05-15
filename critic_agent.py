import json
import os
from glob import glob

def calculate_tree_depth(node):
    """
    Recursively calculates the depth of a tree structure where each node
    contains a 'children' array of descendant nodes.
    """
    if not isinstance(node, dict):
        return 0
    children = node.get('children', [])
    if not children:
        return 1
    return 1 + max(calculate_tree_depth(child) for child in children)

def process_json_files(directory):
    """
    Processes all JSON files in a directory, calculates their maximum tree depth,
    and identifies the deepest structure(s).
    """
    json_files = glob(os.path.join(directory, '**', '*.json'), recursive=True)
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
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print("\nDeepest tree structure(s):")
    for file in deepest_files:
        print(f"File: {file} - Depth: {max_depth}")

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python critic.py <directory>")
        sys.exit(1)
    target_directory = sys.argv[1]
    process_json_files(target_directory)

"""
{
  "name": "result_item_1",
  "elements_combined": "item1+item2",
  "children": [
    {
      "name": "result_item_2",
      "elements_combined": "item3+item4",
      "children": []
    },
    {
      "name": "result_item_n",
      "elements_combined": "itemn+itemm",
      "children": []
    }
  ]
}
"""