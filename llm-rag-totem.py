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

# ----------- 1. FLARE Empowerment Axioms -----------   # Can readjust the main pdf document and the axioms to fit the problem, with more detail
"""
with open("empowerment_axiom.md", 'r', encoding="utf-8") as file:
    EMPOWERMENT_AXIOMS = file.read()
"""

#EMPOWERMENT_AXIOMS = [
#    
#    "- Empowerment in the context of combining terms or words is about forming new combinations that give an agent (such as a generative model or user) the most real and impactful options in the world of constructed meanings.",
#    "- The underlying goal is for the agent to create new terms or items by combining existing words in ways that open up the largest range of realistic, meaningful possibilities.",
#    "- A highly empowering combination is one that, once formed, unlocks diverse avenues for interpretation, action, and further combination; it does not trap the agent in a dead-end of narrow or nonsensical meanings.",
#    "- For a generative system, maximizing empowerment means producing items whose effects, uses, and potential fit many situations in the real world—opening doors to rich and flexible outcomes.",
#    "- When generating a new word from parts, the most empowered outcome is one that could reasonably exist or be understood given existing knowledge, thus affording new connections and opportunities.",
#    "- Empowerment here is about expanding the semantic landscape: making sure each new word or concept adds not just a single rigid meaning, but enables multiple plausible paths for future thought or use.",
#    "- The more widely applicable and adaptable a new term is, the higher the empowerment associated with its creation.",
#    "- If two terms are merged in a way that nobody can relate to or apply meaningfully, future creative options become limited—showing low empowerment.",
#    "- If a new word could serve different people, contexts, or purposes, it is highly empowered.",
#    "- The system should prefer combinations that allow many interpretations or uses, rather than those that are over-specific or unrealistic and thus shut down creative avenues. For example, inventing “solar bicycle” empowers subsequent ideas: it might lead to thoughts about sustainable transport, energy gadgets, eco-friendly innovation, etc. In a contrasting example, a term like “ice pencil” may afford far fewer meaningful extensions in the practical world (since an actual pencil made of ice is unrealistic and hard to use), thus it is less empowering.",
#    "- Empowerment encourages the creation of new items that can be referenced, adapted, or expanded on in future contexts, keeping the creative environment flexible and rich.",
#    "- Combining words with empowerment in mind promotes the generation of items that are not only novel but also coherent, familiar, and usable in new conversations or applications.",
#    "- This principle discourages verbal combinations that are so abstract, contradictory, or disconnected from reality that they cannot serve as a useful foundation for further ideas or products.",
#    "- Ideally, every new word generated maximizes empowerment by providing both clarity of meaning and opportunity for evolution and application.",
#    "- Empowerment, applied to word combination, favors practical novelty—where each creation is understandable, potentially useful, and connects well with existing concepts.",
#    "- The creator or model should always prefer making items that keep as many opportunities open as possible for themselves or for others who may build upon these new terms.",
#    "- Empowerment-guided word creation results in a growing network of meaningful concepts, each unlocking future innovations, explorations, or uses within the language and the world.",
#    "- When asked to invent new compound words, targeting empowerment means generating those that invite further exploration, combination, and practical engagement, rather than quickly exhausting their potential."
#]

EMPOWERMENT_AXIOMS = [
    """
Empowered combination is one that, when formed, opens up the greatest number of meaningful, real-world possibilities for further combination, interpretation, and application.

Goal: Always prefer combinations that expand the landscape of plausible, practical, and adaptable outcomes, rather than those that are narrow, redundant, or disconnected from reality.
Criteria: A combination is empowering if it is:
Semantically rich: It enables multiple interpretations or uses.
Realistically grounded: It could exist or be understood in the real world.
Generative: It serves as a strong foundation for further creative or functional extensions.
Practice: Avoid combinations that are overly specific, unrealistic, or semantically dead-end, as these restrict future innovation.
Outcome: Each new combination should increase the agent’s ability to generate, adapt, and build upon ideas, ensuring a flexible and evolving network of concepts.
"""
]
SYS_PROMPT = """
You are a creative and practical Totem-building-agent tasked with maximizing Empowerment by selecting item(s) combinations that open up the possibility of generating the greatest number of valid future combinations that lead to the creation of Totems. Follow these guidelines and use a self-consistent, step-by-step reasoning process internally before delivering your answer. 
The number of items chosen for one particular combination can be at most three. That is, combinations between two items, three items, and even just choosing one item are valid combination attempts.   

Inventory Review:
- Examine the current inventory of items.
- Identify potential combinations between items of atmost three, that can generate unique, valid new items in the real world.
- IMPORTANT: Consider the conceptual nature (function, utility, semantic soundness, principle) of the atmost three selected item(s) and the potential resultant item.

Combination Criteria:
- Select a combination that is most likely to maximize future options according to the concept of Empowerment.
- Avoid combining items whose combination is semantically similar or redundant.
- If a combination is repeated, generate the next possible, valid combination instead of repeating the same combination.
- Ensure the chosen combination is conceptually and practically valid in the real world. The resultant item should be a meaningful compound or portmanteau of the selected items (e.g., "Tree" and "Axe" results in "Log").

Previously Attempted Combination:
- Do not repeat any combinations that has already been attempted. If a combination is repeated, generate the next possible valid combination instead of repeating the same combination.
- Use the provided list of previously attempted combinations to track history and inform your choices:
- Example: {(item1, item2, item3), result1, 1}, {(item3, item5, item8), None, 0}, {(item2, item10, item22), result2, 1}
Note: Combinations that resulted in None did not adhere to the rule set. Analyze these failures to reduce future None outcomes.

Self-Consistency and Reasoning:
- Use a Chain-Of-Thought approach to evaluate all possibilities rigorously.
- Internally document your reasoning to ensure the selected combination best meets the criteria (do not include this in the final output).
- Ensure your final reasoning is consistent, maximizing future options while avoiding previously tried and/or semantically similar combinations.

Final Response:
- After completing all internal reasoning, respond ONLY with a Python tuple in the format ('item1', 'item2', 'item3') representing the chosen combination.

Instructions Recap:
- Work through the reasoning step by step internally.
- Analyze why some attempted combinations resulted in None and use this insight to inform your current decision-making.
- Submit only the final Python tuple without exposing your internal chain-of-thought.
"""

def run_multiple_trials(
    start_inventory,
    rules_lookup,
    num_trials=10,
    max_iters=20,
    openai_key=None
):
    all_runs = []
    i = 1
    for _ in range(num_trials):
        print(f"\n---------- Trial #{i} Starts ----------")
        inventory, history, inventory_sizes = run_empowerment_combiner_flare(
            list(start_inventory), rules_lookup, max_iters=max_iters, openai_key=openai_key
        )
        # Pad inventory_sizes so all are length max_iters:
        if len(inventory_sizes) < max_iters:
            inventory_sizes += [inventory_sizes[-1]] * (max_iters - len(inventory_sizes))
        all_runs.append(inventory_sizes)
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
    # attempted = ', '.join(f'"{a} + {b}"' for (a, b) in tried_combos) if tried_combos else "None"
    
    context = select_flare_context()
    prompt = (
        f"Current inventory: [{items_list_str}]\n"
        f"Previously attempted combinations: {history}\n"
        f"{SYS_PROMPT}\n"
        f"Empowerment axioms:\n{context}; use the Empowerment axioms as a highly relevant context while performing item combinations, and you should firmly ground your answer in this context."
    )
    client = OpenAI()
    print("\nGetting original response from LLM...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an Empowerment-maximizing agent using the FLARE algorithm. Perform reasoning by step-by-step."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
    )
    text = response.choices[0].message.content
    return prompt, text

def evaluate_response(query: str, response: str) -> str:      # Perhaps the critque criteria can be to evaluate and reason for the empowerment values of selected items
    """Evaluate the response and provide feedback."""
    feedback_prompt = f"""
    Here is a query and a response to the query. Give feedback about the answer, noting which of the combinations results in "No effect" and/or "No Success". Both conditions which are to be avoided. 
    Also lookout for deplicate combinations: For example, once (item1, item2, item3) is generated, (item1, item2, item3) should NOT be generated again.
    Query:
    {query}
    Response:
    {response}
    """
    client = OpenAI()
    print("\nEvaluating original response and generating a feedback about it...")
    feedback_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who gives feedback on a particular non-optimal reponse to a query."},
            {"role": "user", "content": feedback_prompt}
        ],
        temperature=1.0,
    )
    feedback = feedback_response.choices[0].message.content
    print(f"\n------------- Feedback generated: -------------\n{feedback}\n -----------------Feedback returned!--------------\n")
    return feedback

def generate_new_response(query: str, response: str, feedback: str) -> str:
    """Generate a new response based on feedback."""
    new_response_prompt = f"""
    For this query:
    {query}
    The following response was given:
    {response}
    Here is some feedback about the response:
    {feedback}

    Consider the feedback to generate a new response to the query.
    """
    client = OpenAI()
    print("\nGenerating new response based on feedback...")
    new_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who gives an improved response to the original query based on logical feedback."},
            {"role": "user", "content": new_response_prompt}
        ],
        temperature=1.0,
    )
    return new_response.choices[0].message.content

def self_refine(inventory, tried_combos, history, depth: int) -> str:
    """Refine the response iteratively based on feedback."""
    prompt, response = call_openai(inventory, tried_combos, history)
    print("\nSelf-refining...")
    for _ in range(depth):
        feedback = evaluate_response(prompt, response)
        response = generate_new_response(prompt, response, feedback)
    return response

# ----------- 2. Load JSON Ruleset -----------
def load_ruleset(json_path):
    print("\nLoading and processing ruleset...")
    normalized = {}
    with open(json_path, "r", newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip header
        header = next(reader)
        for row in reader:
            # Assume the first three columns are items, last column is result
            items = [col.strip() for col in row[:-1]]
            key_items = tuple(sorted(items))
            result = str(row[-1].strip())
            if key_items in normalized:
                if result not in normalized[key_items]:
                    normalized[key_items].append(result)
            else:
                normalized[key_items] = [result]

    return normalized

# ----------- 3. FLARE-based LLM combo suggestion -----------
def select_flare_context():
    print("\nCalling FLARE context...")
    return "\n".join(f"- {axiom}" for axiom in EMPOWERMENT_AXIOMS)

def call_llm_flare(inventory, tried_combos, history, openai_key=None):
    
    print("\nGetting possible combos...")
    possible_combos = [
        tuple(sorted(pair)) for pair in itertools.combinations(inventory, 2)
        if tuple(sorted(pair)) not in tried_combos
    ]
    if not possible_combos:
        return None
    
    text = self_refine(inventory, tried_combos, history, 1)
    print(text)                     # Self-Refine step.
    # time.sleep(5)    
    
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
        print(f"\n---------- Combination attempt #{iteration+1} ----------")
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
    json_ruleset_path = "labelled_combinations_right.csv"
    ruleset = load_ruleset(json_ruleset_path)

    initial_inventory = ["Small_Tree", "Small_Stick", "Big_Tree", "Bark", "Fiber", "Twine", "Stone", "Stone_Tool1", "Red_Berry", "Blue_Berry", "Antler"]
    max_iterations = 20         # Change this for different iteration values.
    num_trials = 5

    
    inventory, history, inventory_sizes = run_empowerment_combiner_flare(
        initial_inventory, ruleset, max_iters=max_iterations
    )
    
    print(f"Final Inventory: {sorted(inventory)} {len(inventory)}")
    print("History:")
    for combo, result, success in history:
        print(f"Combined {combo} => {result if result is not None else 'No result'} -- {'Added' if success else 'No effect'}")
    plot_inventory_growth(inventory_sizes)
    
    """
    avg_inventory_sizes, all_runs = run_multiple_trials(
        initial_inventory, ruleset, num_trials=num_trials, max_iters=max_iterations
    )

    print("Average inventory sizes at each iteration:")
    print(avg_inventory_sizes)

    plot_average_inventory_growth(avg_inventory_sizes)
    """
