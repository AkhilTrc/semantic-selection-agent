import json
import os
import traceback
from glob import glob


def calculate_tree_depth(tree):
    def traverser(node, depth):
        children = node.get("children", [])
        node_results = [str(x) for x in node.get("results", [])] 
        if not children:
            return depth, node_results
        max_depth = depth
        max_branch_results = []
        for child in children:
            child_depth, child_branch_results = traverser(child, depth + 1)
            if child_depth > max_depth:
                max_depth = child_depth
                max_branch_results = child_branch_results
        return max_depth, node_results + max_branch_results

    if isinstance(tree, list):
        if not tree:
            return 0, []
        max_depth = 0
        branch_results = []
        for node in tree:
            depth, results = traverser(node, 1)
            if depth > max_depth:
                max_depth = depth
                branch_results = results
        return max_depth, branch_results

    if isinstance(tree, dict):
        return traverser(tree, 1)

    return 0, []

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
                current_depth,result = calculate_tree_depth(data)
                results[file_path] = current_depth
                print(f"File: {file_path} - Depth: {current_depth}, Results: {result}")

                # Track deepest files
                if current_depth > max_depth:
                    max_depth = current_depth
                    deepest_files = [file_path]
                elif current_depth == max_depth:
                    deepest_files.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                traceback.print_exc()

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