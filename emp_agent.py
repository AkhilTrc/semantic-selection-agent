import os
import openai
import time
from collections import defaultdict

openai.api_key = os.getenv("OPENAI_API_KEY")
BASE_ELEMENTS = {"air", "earth", "fire", "water"}

empowerment_cache = {}  # Empowerment cache to avoid redundant LLM calls

# Helper: Query GPT-4.1-mini for combination result and empowerment estimate
def gpt_combine_and_empowerment(elem1, elem2, discovered):
    prompt = f"""
You are an expert in the game Little Alchemy 2. 
Given the current discovered elements: {', '.join(sorted(discovered))}.
If you combine "{elem1}" and "{elem2}", what is the most likely resulting element? 
If the combination is not valid, answer "None".
If valid, also estimate (as an integer) how many new unique elements this resulting element could likely help create in future combinations (its "empowerment value").
Respond in the following format:
Result: <element or None>
Empowerment: <integer>
"""
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()
    # Parse response
    result = None
    emp_value = 0
    for line in text.splitlines():
        if line.startswith("Result:"):
            result = line.split(":", 1)[1].strip()
            if result.lower() == "none":
                result = None
        if line.startswith("Empowerment:"):
            try:
                emp_value = int(line.split(":", 1)[1].strip())
            except ValueError:
                emp_value = 0
    return result, emp_value

# Recursive empowerment calculation (memoized)
def empowerment(element, discovered):
    if element in empowerment_cache:
        return empowerment_cache[element]
    max_emp = 0
    for partner in discovered:
        if partner == element:
            continue
        result, emp_val = gpt_combine_and_empowerment(element, partner, discovered)
        if result and result not in discovered:
            # Recursively calculate empowerment of the new element
            total_emp = emp_val + empowerment(result, discovered | {result})
            if total_emp > max_emp:
                max_emp = total_emp
        time.sleep(0.5)  # To avoid hitting rate limits
    empowerment_cache[element] = max_emp
    return max_emp

# Select the best combination to maximize empowerment
def select_best_combination(discovered):
    best_score = -1
    best_pair = None
    best_result = None
    for e1 in discovered:
        for e2 in discovered:
            if e1 == e2:
                continue
            result, emp_val = gpt_combine_and_empowerment(e1, e2, discovered)
            if result and result not in discovered and emp_val > best_score:
                best_score = emp_val
                best_pair = (e1, e2)
                best_result = result
            time.sleep(0.5)  # To avoid hitting rate limits
    return best_pair, best_result, best_score

def play_little_alchemy2(max_steps=50):
    discovered = set(BASE_ELEMENTS)
    steps = 0
    print(f"Starting elements: {discovered}")
    while steps < max_steps:
        pair, result, emp = select_best_combination(discovered)
        if not pair or not result:
            print("No more valid empowering combinations found.")
            break
        print(f"Step {steps+1}: Combine {pair[0]} + {pair[1]} → {result} (Empowerment: {emp})")
        discovered.add(result)
        steps += 1
    print(f"Game ended after {steps} steps. Discovered elements: {discovered}")

if __name__ == "__main__":
    play_little_alchemy2(max_steps=20)
