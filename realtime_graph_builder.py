import networkx as nx
import json
import time

# --- Mocks for Demonstration ---

def mock_llm_call(prompt: str) -> str:
    """Mocks a call to an LLM. In a real system, this would be an API call."""
    print(f"\n🤖 LLM received prompt: \"{prompt}\"")
    # Based on the prompt, we return a pre-defined JSON string.
    if "Apollo space program" in prompt:
        return """
        [
            ["Apollo program", "had goal", "landing humans on the Moon"],
            ["Apollo program", "was operated by", "NASA"],
            ["Saturn V", "was the rocket for", "Apollo program"]
        ]
        """
    if "Saturn V" in prompt:
        return """
        [
            ["Saturn V", "had stage", "S-IC"],
            ["Saturn V", "had stage", "S-II"],
            ["Saturn V", "had stage", "S-IVB"]
        ]
        """
    return "[]" # Default empty response

# --- Core System Components ---

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.seen_nodes = set()

    def add_triples_to_graph(self, triples: list):
        """Adds a list of [subject, relation, object] triples to the graph."""
        for subject, relation, obj in triples:
            subject, obj = subject.lower(), obj.lower() # Normalize nodes
            if subject not in self.graph:
                self.graph.add_node(subject)
            if obj not in self.graph:
                self.graph.add_node(obj)
            
            # Avoid duplicate edges for the same relation
            if not self.graph.has_edge(subject, obj, key=relation):
                self.graph.add_edge(subject, obj, key=relation, label=relation)
                print(f"  ✨ Added new edge: ({subject}) -> [{relation}] -> ({obj})")

    def get_unexplored_node(self) -> str | None:
        """A simple strategy to find a node that has been seen but not used as a prompt."""
        all_nodes = set(self.graph.nodes())
        unexplored = all_nodes - self.seen_nodes
        return unexplored.pop() if unexplored else None

    def run_loop(self, initial_prompt: str, iterations: int = 3):
        """The main actor-critic loop."""
        current_prompt = initial_prompt

        for i in range(iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            # 1. Actor: Get response from LLM
            response_text = mock_llm_call(current_prompt)
            self.seen_nodes.add(current_prompt.split("'")[1].lower() if "'" in current_prompt else initial_prompt)
            
            try:
                # 2. Extractor: Parse the response
                triples = json.loads(response_text)
                
                # 3. Graph Manager: Update the graph
                self.add_triples_to_graph(triples)
            except json.JSONDecodeError:
                print(f"  ⚠️ Could not parse LLM response.")
                continue

            # 4. Critic/Strategist: Decide the next prompt
            next_node_to_explore = self.get_unexplored_node()

            if next_node_to_explore:
                current_prompt = f"Tell me more about '{next_node_to_explore}'"
            else:
                print("\n--- Loop End: No more nodes to explore. ---")
                break
            
            time.sleep(1) # Pause for dramatic effect

        print(f"\nFinal Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")


if __name__ == "__main__":
    builder = GraphBuilder()
    builder.run_loop("Tell me about the Apollo space program") 