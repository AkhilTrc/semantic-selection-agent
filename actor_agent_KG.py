import os
import json
import time
import networkx as nx
from typing import List, Tuple
from load_dotenv import load_dotenv
load_dotenv()
import openai
import matplotlib.pyplot as plt
from typing import Optional
from actor_agent_text import PromptEngine, LLMClient  

openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4.1-nano'


class GraphBuilder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.graph = nx.MultiDiGraph()        
        self.seen_nodes = set()
        self.prompt_engine = PromptEngine()   
        self.llm = LLMClient(model=model_name)
    def llm_call(self, node: str) -> Optional[str]:
        """
        Build a simple triple-extraction prompt for `node`,
        send it via LLMClient, and return the raw JSON text.
        """
        
        built = self.prompt_engine.build([(node, node)])
        
        (_, _), prompt_str = built[0]

        # 3) Send to LLM
        resp = self.llm.batch_query([prompt_str])
        return resp[0] if resp else None

    def add_triples_to_graph(self, triples):
        for subj, rel, obj in triples:
            subj, obj = subj.lower(), obj.lower()
            if subj not in self.graph:
                self.graph.add_node(subj)
            if obj not in self.graph:
                self.graph.add_node(obj)
            # now add a keyed edge in the MultiDiGraph
            if not self.graph.has_edge(subj, obj, key=rel):
                self.graph.add_edge(subj, obj, key=rel, label=rel)
                print(f"  ✨ Added edge: ({subj}) -[{rel}]-> ({obj})")

    def get_unexplored_nodes(self) -> List[str]:
        return [n for n in self.graph.nodes() if n not in self.seen_nodes]

    def save_graph(self, path: str = "kg.graphml"):
        """
        Serialize the graph to GraphML (or you can use GEXF, GPickle, etc.)
        """
        nx.write_graphml(self.graph, path)
        print(f"Graph saved to {path}")
        
    def preprocess_response(self,raw: str) -> List[Tuple[str, str, str]]:
        """
        Parses lines like:
        Fire and Earth gives me Lava
        Lava and Fire gives me Magma

        into a list of (subject, relation, object) triples.
        """
        triples = []
        raw = raw.replace("*","")
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            # Expect exactly one " gives me "
            if " gives me " not in line:
                print(f"  ⚠️ Skipping unrecognized line: {line!r}")
                continue

            left, obj = line.split(" gives me ", 1)
            obj = obj.strip()
            
            # Expect exactly one " and " on the left
            if " and " not in left:
                print(f"  ⚠️ Skipping unrecognized left part: {left!r}")
                continue

            subj, base = left.split(" and ", 1)
            subj, base = subj.strip().lower(), base.strip().lower()
            subj = subj+"_and_"+base
            triples.append([subj, "gives", obj.lower()])
        return triples

    def run_loop(
        self,
        base_elements: List[str],
        max_iters: int = 5,
        batch_size: int = 2
    ):
        """

        """

        self.base = [e.lower() for e in base_elements]
        for node in self.base:
            self.graph.add_node(node)
        print(f"Seeded graph with base elements: {base_elements}")

        for iteration in range(1, max_iters + 1):
            to_explore = self.get_unexplored_nodes()[:batch_size]
            if not to_explore:
                print("No more new nodes — stopping early.")
                break

            print(f"\n--- Iteration {iteration}: exploring {to_explore} ---")

            pairs = [(new, base) 
                     for new in to_explore 
                     for base in self.base 
                     if new != base]

            built = self.prompt_engine.build(pairs)
            pairs_batch, prompts = zip(*built)
            responses = self.llm.batch_query(list(prompts))

            for (node, _), raw in zip(pairs_batch, responses):
                print(f"\n🔄 Response for '{node}':\n{raw}")
                try:
                    triples = self.preprocess_response(raw)
                    print(f"  Extracted triples: {triples}")
                    
                except Exception as e:
                    print(f"  ⚠️ Could not parse JSON for '{node}'. Marking as seen.")
                    self.seen_nodes.add(node)
                    continue

                self.add_triples_to_graph(triples)
                self.seen_nodes.add(node)

            time.sleep(1)

        print(
            f"\nDone! Graph has {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges."
        )

if __name__ == "__main__":
    builder = GraphBuilder()
    base = ['Water', 'Earth', 'Fire', 'Air']
    builder.run_loop(base)
    builder.save_graph("kg.graphml")
