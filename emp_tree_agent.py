import os
import openai
import time
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_ELEMENTS = {"air", "earth", "fire", "water"}
empowerment_cache = {}

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

def build_json_tree(element, discovered, depth=0, max_depth=2):
    """
    Recursively builds the JSON tree for the current element.
    Limits recursion to max_depth for demonstration.
    """
    if depth > max_depth:
        return []
    children = []
    for partner in sorted(discovered):
        if partner == element:
            continue
        sorted_pair = "+".join(sorted([element, partner]))
        result, emp_val = gpt_combine_and_empowerment(element, partner, discovered)
        if result and result not in discovered:
            # Recursively build children for the new result
            next_discovered = discovered | {result}
            child_json = build_json_tree(result, next_discovered, depth + 1, max_depth)
            children.append({
                "name": result,
                "elements_combined": sorted_pair,
                "children": child_json
            })
        time.sleep(0.5)  # To avoid rate limits
    return children

def play_little_alchemy2_json(max_steps=5, max_depth=2):
    discovered = set(BASE_ELEMENTS)
    steps = 0
    json_output = []

    while steps < max_steps:
        best_score = -1
        best_pair = None
        best_result = None
        for e1 in discovered:
            for e2 in discovered:
                if e1 == e2:
                    continue
                sorted_pair = "+".join(sorted([e1, e2]))
                result, emp_val = gpt_combine_and_empowerment(e1, e2, discovered)
                if result and result not in discovered and emp_val > best_score:
                    best_score = emp_val
                    best_pair = (e1, e2)
                    best_result = result
                time.sleep(0.5)
        if not best_pair or not best_result:
            print("No more valid empowering combinations found.")
            break

        # Build the JSON tree for this combination
        children = build_json_tree(best_result, discovered | {best_result}, depth=1, max_depth=max_depth)
        json_obj = {
            "name": best_result,
            "elements_combined": "+".join(sorted(best_pair)),
            "children": children
        }
        json_output.append(json_obj)
        print(json.dumps([json_obj], indent=2))
        discovered.add(best_result)
        steps += 1

    # Final output for all steps
    print("Final JSON output for all steps:")
    print(json.dumps(json_output, indent=2))

if __name__ == "__main__":
    play_little_alchemy2_json(max_steps=3, max_depth=1)
