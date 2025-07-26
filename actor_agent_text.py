import os
import json
from typing import List, Tuple, Dict, Set
import openai
from load_dotenv import load_dotenv
load_dotenv()
# Configuration
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4.1-nano'



class PromptEngine:
    def __init__(self):
        self.cache: Dict[Tuple[str, str], str] = {}

    def build(self, pairs: List[Tuple[str, str]]) -> List[Tuple[Tuple[str, str], str]]:
        prompts = []
        for e1, e2 in pairs:
            if (e1, e2) not in self.cache:
                prompt = f"""
                You are a creative alchemist. Your job is to invent new elements by combining two existing ones in imaginative yet plausible ways.

                When given two elements, combine their characteristics to create a new, unique element.
                Express each result as:
                `[Element1] and [Element2] gives me [ResultElement]`

                **Rules:**

                1. For each new element generated, attempt to combine it with each of the original input elements (the base set), but **do not** combine new elements with each other unless one is from the base set.
                2. Continue this process for several generations, always only pairing a newly created element with any element from the original base set.
                3. Each combination should produce a plausible new element, grounded in the properties or concepts of the ingredients.
                4. Do not repeat combinations or reverse orderings (e.g., if "Water and Fire" is done, skip "Fire and Water").

                **Template for output:**

                ```
                [Element1] and [Element2] gives me [ResultElement]
                ```

                **Input Elements:**

                * {e1}
                * {e2}

                **Step-by-step:**

                1. Combine {e1} and {e2} to create [O1].
                2. Combine [O1] with {e1} (if not already used).
                3. Combine [O1] with {e2} (if not already used).
                4. For each new element produced, only combine with the original input elements, not with other new elements.

                ---

                **Example Workflow:**
                Suppose Input Elements: **Fire** and **Water**

                ```
                Fire and Water gives me Steam
                Steam and Fire gives me Energy
                Steam and Water gives me Cloud
                ```
             """
                prompts.append(((e1, e2), prompt))
        return prompts

    def cache_result(self, pair: Tuple[str, str], result: str):
        self.cache[pair] = result

# 4. LLM Client
class LLMClient:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def batch_query(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            resp = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            text = resp.choices[0].message.content.strip()
            responses.append(text)
        return responses

