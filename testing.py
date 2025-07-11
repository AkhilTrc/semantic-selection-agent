import json
import os

def count_unique_elements(tree, unique_elements=None):
    if unique_elements is None:
        unique_elements = set()
    for node in tree:
        unique_elements.add(node['name'])
        if 'children' in node and node['children']:
            count_unique_elements(node['children'], unique_elements)
    return unique_elements

def count_unique_elements_in_folder(folder_path):
    unique_elements = set()
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            print(f"\njsonfile: {file_path}\n")
            with open(file_path, 'r') as f:
                try:
                    tree = json.load(f)
                    if tree is None:  # File is null or empty
                        continue     # Skip this file
                    # Process your JSON here
                except json.JSONDecodeError:
                    # File is empty or not valid JSON, skip it
                    continue
                unique_elements.update(count_unique_elements(tree))
    return unique_elements

def count_unique_elements_in_file(filename):
    filename = "inventory_jsons_v4/" + filename
    unique_elements = set()
    print(f"\njsonfile: {filename}\n")
    with open(filename, 'r') as f:
            tree = json.load(f)
    unique_elements.update(count_unique_elements(tree))
    return unique_elements

def evaluate_folder(folder_path):
    unique_elements = count_unique_elements_in_folder(folder_path)
    total_unique = len(unique_elements)
    return total_unique

if __name__ == "__main__":
    folder1_path = 'inventory_jsons_v4'           # Empowerment
    folder2_path = 'inventory_jsons_rand'              # Random choice
    unique_count_1 = evaluate_folder(folder1_path)
    unique_count_2 = evaluate_folder(folder2_path)
    print(f"Total unique elements in folder 1 (Empowerment): {unique_count_1}")
    print(f"Total unique elements in folder 2 (Random choice): {unique_count_2}")



