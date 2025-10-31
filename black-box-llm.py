import os
import json
import csv
import time
import openai
import numpy as np
from openai import OpenAI
import itertools
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4.1-mini'

SYS_PROMPT = """
Let's play Little ALchemy 2, a game where you need to craft items by combining existing items in pairs. For example, if you combine
"human" and "human" you get a new item: "love". Combinations are inspired from the real-world: combining "love" and "primordial soup"
will not give you a new word ... I will give you a set of initials items and you need to craft as many as possible. ... Let's start!
"""

def run_multiple_trials(
    start_inventory,
    rules_lookup,
    num_trials=20,
    max_iters=50,
    openai_key=None
):
    # Create directory for saving trial CSV files
    save_dir = "plain-black-box-history"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    all_runs = []
    i = 1
    for _ in range(num_trials):
        print(f"\n---------- Trial #{i} Starts ----------")
        inventory, history, inventory_sizes = run_black_box_trials(
            list(start_inventory), rules_lookup, max_iters=max_iters, openai_key=openai_key
        )
        # Pad inventory_sizes so all are length max_iters:
        if len(inventory_sizes) < max_iters:
            inventory_sizes += [inventory_sizes[-1]] * (max_iters - len(inventory_sizes))
        all_runs.append(inventory_sizes)

        # Save trial inventory sizes to CSV with header "inventory_size"
        filename = os.path.join(save_dir, f"trial_{i}_inventory_sizes.csv")
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["inventory_size"])
            for size in inventory_sizes:
                writer.writerow([size])

        print(f"\nSaved inventory sizes for Trial #{i} to {filename}")
        print(f"\n---------- Trial #{i} End!----------")
        i = i + 1

    # Compute average inventory size at each iteration
    avg_inventory_sizes = np.mean(all_runs, axis=0)
    return avg_inventory_sizes, all_runs


# ----------- 1. Self-Refine Technique -----------

def call_openai(inventory, tried_combos, history, model: str = MODEL_NAME) -> str:
    """Call OpenAI API to get an initial response."""
    # Import openai here so script doesn't break if not available
    import openai

    print("\nGetting possible combos...")
    possible_combos = [
        tuple(sorted(pair)) for pair in itertools.combinations(inventory, 2)
        if tuple(sorted(pair)) not in tried_combos
    ]
    if not possible_combos:
        return None

    items_list_str = ', '.join(f'"{x}"' for x in inventory)
    prompt = (
        f"Current inventory: [{items_list_str}]\n"
        f"Previously attempted combinations: {history}\n"
        f"{SYS_PROMPT}\n"
        )
    client = OpenAI()
    print("\nGetting original response from LLM...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an agent that plays the Little Alchemy 2 game. Keep track of the valid and invalid combinations during this task, and use this information to perform better internal reasoning. Perform reasoning by step-by-step."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
    )
    text = response.choices[0].message.content
    time.sleep(2)
    return prompt, text

# ----------- 2. Load JSON Ruleset -----------
def load_ruleset(json_path):
    print("\nLoading and processing ruleset...")
    with open(json_path, "r") as f:
        combo_dict = json.load(f)
    # Store key as frozenset for order-independence, and value as a list (for uniformity)
    normalized = {}
    for k, v in combo_dict.items():
        key_items = tuple(sorted([item.strip() for item in k.split(',')]))
        if isinstance(v, list):
            results = [str(result) for result in v]
        else:
            results = [str(v)]
        normalized[key_items] = results
    return normalized

def call_black_box(inventory, tried_combos, history, openai_key=None):
    
    print("\nGetting possible combos...")
    possible_combos = [
        tuple(sorted(pair)) for pair in itertools.combinations(inventory, 2)
        if tuple(sorted(pair)) not in tried_combos
    ]
    if not possible_combos:
        return None
    
    prompt, response = call_openai(inventory, tried_combos, history)
    print(response)
    time.sleep(2)    
    
    try:
        combo = eval(response.strip())
        if (
            isinstance(combo, tuple)
            and len(combo) == 2
            and all(isinstance(x, str) for x in combo)
        ):
            return tuple(sorted(combo))
    except Exception:
        pass
    # Fallback: pick first unseen combo
    return possible_combos[0]

# ----------- 4. Run iteration loop -----------
def run_black_box_trials(
    start_inventory,
    rules_lookup,
    max_iters=50,
    openai_key=None
):
    inventory = set(start_inventory)
    tried_combos = set()
    inventory_sizes = []
    history = []

    for iteration in range(max_iters):
        print(f"\n---------- Combination attempt #{iteration+1} ----------")
        combo = call_black_box(list(inventory), tried_combos, history, openai_key)
        if combo is None:
            break  # exhausted all pairs
        tried_combos.add(combo)
        # Validate combo
        if combo in rules_lookup:
            results = rules_lookup[combo]
            new_items = [r for r in results]
            for item in new_items:
                inventory.add(item)
            history.append((combo, results, bool(new_items)))
            new_items = list(set(new_items))
        else:
            history.append((combo, None, False))
        inventory_sizes.append(len(inventory))
    print("\nDone!\n")
    return inventory, history, inventory_sizes

def plot_inventory_growth(inventory_sizes):

    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(inventory_sizes)+1), inventory_sizes, marker='o')     # remove parameter (, marker='o') for dot-by-dot graphing
    plt.xlabel("Iteration")
    plt.ylabel("Inventory Size")
    plt.title("Inventory Size Growth per Iteration (FLARE Empowerment)")
    plt.grid(True)
    plt.show()

def plot_average_inventory_growth(avg_inventory_sizes):
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(avg_inventory_sizes)+1), avg_inventory_sizes, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Average Inventory Size")
    plt.title("Average Inventory Size Growth per Iteration (across runs)")
    plt.grid(True)
    plt.show()

# ----------- 5. Usage Example -----------
if __name__ == "__main__":
    # Path to your JSON file
    json_ruleset_path = "elements.JSON"
    ruleset = load_ruleset(json_ruleset_path)

    initial_inventory = ["fire", "air", "water", "earth"]
    max_iterations = 50        # Change this for different iteration values.
    num_trials = 20
    
    avg_inventory_sizes, all_runs = run_multiple_trials(
        initial_inventory, ruleset, num_trials=num_trials, max_iters=max_iterations
    )

    print("Average inventory sizes at each iteration:")
    print(avg_inventory_sizes)

    print("\nWriting Average Inventory Sizes to csv...")
    with open(f"avg_inventory_sizes_{max_iterations}iters_{num_trials}trials.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for value in avg_inventory_sizes:
            writer.writerow([value])

    plot_average_inventory_growth(avg_inventory_sizes)
    
