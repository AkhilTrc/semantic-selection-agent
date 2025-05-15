import os
import json
import time

# Import your critic functions (adjust import paths as needed)
from critic_agent import process_json_files  # The function to process JSON files and calculate depths
from actor_agent import create_thread, create_dynamic_agent, add_user_message, get_response, delete_agent

# Directory where JSON inventory files reside
INVENTORY_DIR = "inventory_jsons"

# User ID for the actor agent session
USER_ID = "user_123"
AGENT_NAME = "Cool Assistant"

def save_json_to_file(data, filepath):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Failed to save JSON to file: {e}")
        with open("text.txt", "w", encoding="utf-8") as f:
            f.write(str(data))

def load_json_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def add_new_inventory_item(base_elements,FLAG):
    """
    Use the actor_agent to generate a new inventory tree JSON based on base elements.
    Returns the parsed JSON object.
    """
    if FLAG == False:
        create_thread(USER_ID)
        create_dynamic_agent(USER_ID, AGENT_NAME)
        FLAG = True

    # Construct prompt for generating a new inventory tree
    print(base_elements)
    prompt = (
        f"Using the base elements: {base_elements}, generate the deepest possible tree "
        "of realistic combinations, following the system rules. Output only the final JSON."
    )
    add_user_message(USER_ID, prompt)
    reply = get_response(USER_ID)

    try:
        # Clean and parse JSON response from the agent
        new_tree = json.loads(reply)
        print("[INFO] New inventory tree generated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON from actor agent response: {e}")
        new_tree = None

    return new_tree,FLAG

def merge_new_tree_into_inventory(new_tree, inventory_dir):
    """
    Save the new tree as a new JSON file in the inventory directory.
    """
    if not os.path.exists(inventory_dir):
        os.makedirs(inventory_dir)

    # Generate a unique filename based on timestamp
    filename = f"inventory_{int(time.time())}.json"
    filepath = os.path.join(inventory_dir, filename)

    save_json_to_file(new_tree, filepath)
    print(f"[INFO] New inventory saved to {filepath}")

def main(iterations=5, delay_seconds=5):
    """
    Main loop: generate new inventory items, save them, then run the critic process.
    """
    base_elements = ["air", "water", "fire", "earth"]

    for i in range(iterations):
        print(f"\n=== Iteration {i+1} ===")

        # Step 1: Generate new inventory tree JSON
        FLAG = False
        new_tree,FLAG = add_new_inventory_item(base_elements, FLAG)

        # Step 2: Save new inventory JSON file
        merge_new_tree_into_inventory(new_tree, INVENTORY_DIR)

        # Step 3: Run critic process on all inventory JSON files
        print("[INFO] Running critic process on inventory JSON files...")
        depth_results = process_json_files(INVENTORY_DIR)
        print(f"[INFO] Critic process completed. Depth results: {depth_results}")
        base_elements += [depth_results]
        base_elements = list(set(base_elements)) 
        print(base_elements)

        # Wait before next iteration
        time.sleep(delay_seconds)
    # Clean up: delete the agent after all iterations
    delete_agent(USER_ID)
if __name__ == "__main__":
    main()