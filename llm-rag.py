import os
import json
import openai
from openai import OpenAI
import itertools
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4.1-nano'

# ----------- 1. FLARE Empowerment Axioms -----------
EMPOWERMENT_AXIOMS = [
    "Empowerment is the information-theoretic capacity of an agent to influence the world via its actions.",
    "Empowerment maximization means selecting actions that maximize the number of possible future states.",
    "To increase empowerment, favor item combinations that create new, unique elements or pathways.",
    "Avoid combinations that limit or collapse the diversity of subsequent actions.",
    "Empowerment is agent-centric: it is about keeping future options open."
]

FLARE_PROMPT = """
You are an agent maximizing empowerment, as defined below.
Use these axioms to reason step by step and choose TWO items from the inventory to combine for maximum future options.
First, consider the current inventory and which item pairs can generate unique new elements.
Second, avoid pairs that were already tried.
Respond ONLY with a Python tuple ('item1','item2'). Do all reasoning before your answer.
"""

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

# ----------- 3. FLARE-based LLM combo suggestion -----------
def select_flare_context():
    print("\nCalling FLARE context...")
    return "\n".join(f"- {axiom}" for axiom in EMPOWERMENT_AXIOMS)

def call_llm_flare(inventory, tried_combos, openai_key=None):
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
    attempted = ', '.join(f'"{a} + {b}"' for (a, b) in tried_combos) if tried_combos else "None"
    
    context = select_flare_context()
    prompt = (
        f"Current inventory: [{items_list_str}]\n"
        f"Previously attempted combinations: {attempted}\n"
        f"{FLARE_PROMPT}\n"
        f"Empowerment axioms:\n{context}"
    )
    client = OpenAI()
    print("\nGetting response from LLM...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an empowerment-maximizing agent using FLARE. Reason step-by-step."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content
    try:
        combo = eval(text.strip())
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
def run_empowerment_combiner_flare(
    start_inventory,
    rules_lookup,
    max_iters=20,
    openai_key=None
):
    inventory = set(start_inventory)
    tried_combos = set()
    inventory_sizes = []
    history = []

    for iteration in range(max_iters):
        print(f"\n------- Iter #{iteration+1} -------")
        combo = call_llm_flare(list(inventory), tried_combos, openai_key)
        if combo is None:
            break  # exhausted all pairs
        tried_combos.add(combo)
        # Validate combo
        if combo in rules_lookup:
            results = rules_lookup[combo]
            new_items = [r for r in results if r not in inventory]
            for item in new_items:
                inventory.add(item)
            history.append((combo, results, bool(new_items)))
        else:
            history.append((combo, None, False))
        inventory_sizes.append(len(inventory))
    print("\nDone!\n")
    return inventory, history, inventory_sizes

def plot_inventory_growth(inventory_sizes):
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(inventory_sizes)+1), inventory_sizes, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Inventory Size")
    plt.title("Inventory Size Growth per Iteration (FLARE Empowerment)")
    plt.grid(True)
    plt.show()

# ----------- 5. Usage Example -----------
if __name__ == "__main__":
    # Path to your JSON file
    json_ruleset_path = "elements.JSON"
    ruleset = load_ruleset(json_ruleset_path)

    initial_inventory = ["fire", "air", "water", "earth"]
    max_iterations = 10

    inventory, history, inventory_sizes = run_empowerment_combiner_flare(
        initial_inventory, ruleset, max_iters=max_iterations
    )

    print(f"Final Inventory: {sorted(inventory)}")
    print("History:")
    for combo, result, success in history:
        print(f"Combined {combo} => {result if result is not None else 'No result'} -- {'Added' if success else 'No effect'}")
    plot_inventory_growth(inventory_sizes)
