import os
import json
import time
import openai
from openai import OpenAI
import itertools
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4.1-mini'

# ----------- 1. FLARE Empowerment Axioms -----------   # Can readjust the main pdf document and the axioms to fit the problem, with more detail

with open("empowerment_axiom.md", 'r', encoding="utf-8") as file:
    EMPOWERMENT_AXIOMS = file.read()

"""
EMPOWERMENT_AXIOMS = [
    
    Empowerment as a Universal Utility Function: Empowerment is proposed as a universal utility function that applies to any agent, regardless of its specific sensorimotor apparatus or environment. It is defined as the information-theoretic capacity of an agent’s actuation channel, reflecting the agent's potential to influence the world.

Local and Universal Nature: Empowerment is local because it does not rely on extensive historical data or global knowledge. It is universal because it can be applied across different species and contexts, adapting to the agent's morphology and ecological niche.

Agent-Centric Perspective: The concept of empowerment is rooted in an agent-centric view, focusing on the interaction between the agent's sensors and actuators without needing to reference the external environment explicitly.

Information-Theoretic Basis: Empowerment is quantified using information theory, specifically as the channel capacity of the agent's actuation channel. This approach ensures that the measure is task-independent and does not rely on the specific meanings of actions or states.

Maximizing Future Options: Empowerment encourages agents to seek situations where they have the most control or influence over their environment, thereby maximizing their future options and potential actions.

Embodiment and Control: Empowerment is tied to the concept of embodiment, where the agent's control over its environment is perceived through the coupling of its sensors and actuators. The more an agent can influence its sensor readings through its actions, the higher its empowerment.

Adaptation to Environmental Dynamics: Empowerment naturally adapts to changes in the environment's dynamics without needing explicit encoding. It captures changes in the agent's potential actions and control as the environment evolves.

Practical Applications: Empowerment can be used as a fitness function in evolutionary algorithms or as a guiding principle for adaptive behavior in agents, promoting actions that keep options open and maintain control over the environment.

Experiments and Illustrations: The paper illustrates empowerment through experiments in grid worlds and mazes, showing how empowerment measures can capture intuitive features of the environment and guide agent behavior effectively.
    
]
"""
FLARE_PROMPT = """
You are a creative and analytical agent tasked with maximizing empowerment by selecting item combinations that open up the greatest number of future possibilities. Follow these guidelines and use a self-consistent, step-by-step reasoning process internally before delivering your answer.

Inventory Review:

Examine the current inventory of items.
Identify potential item pairs that can generate unique, valid new elements in the real world.
Consider the conceptual nature (function, semantic meaning) of the items.
Combination Criteria:

Select a pair that is most likely to maximize future options.
Avoid pairing items whose combination is semantically similar or redundant.
Ensure the chosen pair is conceptually and practically valid in the real world. The resultant item should be a meaningful compound or portmanteau of the selected items (e.g., "Bat" and "Human" results in "Batman").
Previously Attempted Pairs:

Do not repeat any pair that has already been attempted.
Use the provided list of previously attempted combinations to track history and inform your choices:
Example: {(item1, item2), result1, 1}, {(item3, item5), None, 0}, {(item2, item10), result2, 1}
Note: Pairs that resulted in None did not adhere to the rule set. Analyze these failures to reduce future None outcomes.
Self-Consistency and Reasoning:

Use a chain-of-thought approach to evaluate all possibilities rigorously.
Internally document your reasoning to ensure the selected pair best meets the criteria (do not include this in the final output).
Ensure your final reasoning is consistent, maximizing future options while avoiding previously tried or semantically similar pairs.
Final Response:

After completing all internal reasoning, respond ONLY with a Python tuple in the format ('item1', 'item2') representing the chosen combination.
Instructions Recap:

Work through the reasoning step by step internally.
Analyze why some attempted pairs resulted in None and use this insight to inform your current decision-making.
Submit only the final Python tuple without exposing your internal chain-of-thought.
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

def call_llm_flare(inventory, tried_combos, history, openai_key=None):
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
        f"Previously attempted combinations: {history}\n"
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
        temperature=1.0,
    )
    text = response.choices[0].message.content
    time.sleep(5)
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
    max_iters=10,
    openai_key=None
):
    inventory = set(start_inventory)
    tried_combos = set()
    inventory_sizes = []
    history = []

    for iteration in range(max_iters):
        print(f"\n---------- Iter #{iteration+1} ----------")
        combo = call_llm_flare(list(inventory), tried_combos, history, openai_key)
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
    max_iterations = 50

    inventory, history, inventory_sizes = run_empowerment_combiner_flare(
        initial_inventory, ruleset, max_iters=max_iterations
    )

    print(f"Final Inventory: {sorted(inventory)} {len(inventory)}")
    print("History:")
    for combo, result, success in history:
        print(f"Combined {combo} => {result if result is not None else 'No result'} -- {'Added' if success else 'No effect'}")
    plot_inventory_growth(inventory_sizes)
