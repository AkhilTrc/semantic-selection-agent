from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Any

from collections import deque

from openai import OpenAI

from dotenv import load_dotenv

import json, os, re, traceback

import matplotlib.pyplot as plt

load_dotenv()

client = OpenAI()

Pair = Tuple[str, str]
Validator = Callable[[Pair], Tuple[bool, List[str]]]

JSON_PATH = "elements.JSON"
START_INVENTORY = ["fire", "air", "water", "earth"]
MODEL_NAME = "gpt-4.1-mini"
MAX_ITERS = 50
TRIALS = 10
ORDERED = False
ALLOW_SELF = False
EMPOWER_DEPTH = 2
OUTPUT_HISTORY = "run_history.jsonl"
OUTPUT_SUMMARY = True
TRIED_CONTEXT_MAX = 700

OPENAI_TEMPERATURE = 0.1
OPENAI_PRESENCE_PENALTY = 0.9
OPENAI_FREQUENCY_PENALTY = 0.7
FLATLINE_WINDOW = 4
FALLBACK_TOPK = 6
TUPLE_RE = re.compile(r"\('([^']+)'\s*,\s*'([^']+)'\)")

def _canon_pair(a: str, b: str, ordered: bool=False) -> Pair:
    a, b = a.strip(), b.strip()
    return (a, b) if ordered else tuple(sorted((a, b)))

def load_elements_rules(json_path: str, *, ordered: bool=False) -> Dict[Pair, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rules: Dict[Pair, List[str]] = {}
    for key, outs in raw.items():
        parts = [p.strip() for p in key.split(",")]
        if len(parts) != 2: continue
        pair = _canon_pair(parts[0], parts[1], ordered=ordered)
        dedup = []
        seen = set()
        for r in outs:
            r = r.strip()
            if r not in seen:
                seen.add(r); dedup.append(r)
        rules[pair] = dedup
    return rules

def make_rules_validator(rules: Dict[Pair, List[str]], *, ordered: bool=False) -> Validator:
    def _norm(pair: Pair) -> Pair:
        return pair if ordered else tuple(sorted(pair))
    def _validate(pair: Pair) -> Tuple[bool, List[str]]:
        key = _norm(pair)
        return (True, list(rules[key])) if key in rules else (False, [])
    return _validate

def pair_empowerment(pair: Pair, inventory_set: Set[str], rules_lookup: Dict[Pair, List[str]], depth: int=2) -> int:
    if pair not in rules_lookup: return 0
    start_new = [x for x in rules_lookup[pair] if x not in inventory_set]
    if depth <= 1: return len(start_new)
    seen = set(inventory_set) | set(start_new)
    frontier = deque(start_new)
    for _ in range(depth-1):
        nxt = []
        while frontier:
            x = frontier.popleft()
            for (a,b), outs in rules_lookup.items():
                if x in (a,b):
                    for r in outs:
                        if r not in seen:
                            seen.add(r); nxt.append(r)
        if not nxt: break
        frontier = deque(nxt)
    return max(0, len(seen) - len(inventory_set))

def top_pairs_by_empowerment(inventory: Sequence[str], tried_combos: Set[Pair], rules_lookup: Dict[Pair, List[str]], top_k: int=8, depth: int=2) -> List[Pair]:
    inv = list(inventory)
    tried = set(tried_combos)
    scored = []
    inv_set = set(inv)
    for i in range(len(inv)):
        for j in range(i+1, len(inv)):
            p = _canon_pair(inv[i], inv[j])
            if p in tried: continue
            s = pair_empowerment(p, inv_set, rules_lookup, depth=depth)
            if s > 0: scored.append((s, p))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in scored[:top_k]]

def per_item_empowerment(inventory: Sequence[str], tried_combos: Set[Pair], rules_lookup: Dict[Pair, List[str]], depth: int=2) -> List[Tuple[str,int]]:
    inv_set = set(inventory)
    best = {x:0 for x in inventory}
    for a in inventory:
        for b in inventory:
            if a >= b: continue
            p = _canon_pair(a,b)
            if p in tried_combos: continue
            s = pair_empowerment(p, inv_set, rules_lookup, depth=depth)
            if s > best[a]: best[a] = s
            if s > best[b]: best[b] = s
    return sorted(best.items(), key=lambda t: t[1], reverse=True)

def is_flatline(inv_sizes: List[int], window: int=4) -> bool:
    if len(inv_sizes) < window: return False
    return inv_sizes[-1] - inv_sizes[-window] == 0

def parse_single_tuple(text: str) -> Optional[Pair]:
    if not text: return None
    m = TUPLE_RE.search(text)
    if m: return (m.group(1), m.group(2))
    try:
        data = json.loads(text)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return (str(data[0]), str(data[1]))
    except Exception:
        return None
    return None

def build_prompt(inventory: Sequence[str], tried_combos: Set[Pair], item_emp_hints: List[Tuple[str,int]], taboo_items: Sequence[str], escape_mode: bool, ordered: bool, allow_self: bool) -> Tuple[str,str]:
    items_list_str = ', '.join(f"'{x}'" for x in inventory)
    tried_list = list(tried_combos)
    tried_str = ', '.join(f"('{a}','{b}')" for (a,b) in tried_list[-TRIED_CONTEXT_MAX:])
    emp_hint_str = ', '.join(f"('{x}', {s})" for x,s in item_emp_hints[:12])
    taboo_str = ', '.join(f"'{x}'" for x in taboo_items[:10])
    sys_msg = "Return exactly one Python tuple like ('a','b'). Output only the tuple. Think silently."
    stage = (
        "Two-item crafting game. Empowerment is the tendency for a pair to unlock many new items over the next few steps. "
        "You will internally consider several candidate pairs from the inventory and output only the best one by a loss-averse utility: "
        "repeated/malformed/non-inventory/self-combo is very negative; invalid is negative; valid & novel with high-empowerment items is positive."
    )
    checklist = (
        "Checklist before output: tuple syntax only; both items in inventory; not in blacklist; not taboo; "
        f"{'self-combos allowed' if allow_self else 'no self-combo'}; prefer higher empowerment items; avoid items repeated often."
    )
    escape_note = "Escape mode: recent growth stalled; favor high-empowerment, underused directions." if escape_mode else ""
    user_msg = (
        f"{stage}\n"
        f"Inventory: [{items_list_str}]\n"
        f"Blacklist (do not repeat): [{tried_str}]\n"
        f"Per-item empowerment signals: [{emp_hint_str}]\n"
        f"Taboo items (must not use): [{taboo_str}]\n"
        f"Rules: {'ordered allowed' if ordered else 'order does not matter'}\n"
        f"{escape_note}\n"
        f"{checklist}\n"
        "Output only one tuple."
    )
    return sys_msg, user_msg

def llm_propose_one_pair(inventory: Sequence[str], tried_combos: Set[Pair], item_emp_hints: List[Tuple[str,int]], taboo_items: Sequence[str], escape_mode: bool, ordered: bool, allow_self: bool, model: str) -> Optional[Pair]:
    sys_msg, user_msg = build_prompt(inventory, tried_combos, item_emp_hints, taboo_items, escape_mode, ordered, allow_self)
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=OPENAI_TEMPERATURE,
            presence_penalty=OPENAI_PRESENCE_PENALTY,
            frequency_penalty=OPENAI_FREQUENCY_PENALTY,
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}]
        )
        return parse_single_tuple((resp.choices[0].message.content or "").strip())
    except Exception:
        return None

def run_empowerment_loop(
    start_inventory: Iterable[str], rules: Dict[Pair, List[str]],
    ordered: bool=False, allow_self: bool=False, model: str=MODEL_NAME,
    max_iters: int=50, depth: int=2, use_fallback: bool=True
) -> Tuple[List[str], List[Dict[str,Any]]]:
    """
    Args:
        use_fallback (bool): If False, do NOT fallback to empowerment calculation, only use LLM suggestion.
    """
    inventory: Set[str] = set(start_inventory)
    tried_combos: Set[Pair] = set()
    inventory_sizes: List[int] = [len(inventory)]
    history: List[Dict[str,Any]] = []
    for iteration in range(1, max_iters+1):
        # print(f"\nCrafting Step #{iteration}...")
        escape = is_flatline(inventory_sizes, window=FLATLINE_WINDOW)
        tabu_items: List[str] = []
        item_emp_hints = per_item_empowerment(list(inventory), tried_combos, rules, depth=depth)
        llm_pair = llm_propose_one_pair(list(inventory), tried_combos, item_emp_hints, tabu_items, escape, ordered, allow_self, model)

        chosen: Optional[Pair] = None
        mode = "llm"
        fallback_reason = None
        use_emp_fallback = False

        if llm_pair is not None:
            a, b = llm_pair
            key = _canon_pair(a, b, ordered)
            if (not allow_self and a==b) or (a not in inventory or b not in inventory):
                use_emp_fallback = True
                fallback_reason = "bad_output"
            elif key in tried_combos:
                use_emp_fallback = True
                fallback_reason = "repeat"
            else:
                chosen = key
        else:
            use_emp_fallback = True
            fallback_reason = "malformed"

        if escape:
            use_emp_fallback = True
            fallback_reason = "flatline" if fallback_reason is None else f"flatline+{fallback_reason}"

        # MAIN MODIFICATION for fallback toggle:
        if (use_emp_fallback or chosen is None):
            if use_fallback:
                top_list = top_pairs_by_empowerment(list(inventory), tried_combos, rules, top_k=FALLBACK_TOPK, depth=depth)
                if top_list:
                    chosen = top_list[0]
                    mode = "fallback"
                else:
                    history.append({"iter": iteration, "event": "no_choice"})
                    inventory_sizes.append(len(inventory))
                    continue
            else:
                # Don't fallback: just skip this iteration, nothing added
                history.append({"iter": iteration, "event": "skip_llm_failed"})
                inventory_sizes.append(len(inventory))
                continue

        tried_combos.add(chosen)
        emp_val = pair_empowerment(chosen, set(inventory), rules, depth=depth)
        ok = chosen in rules
        results = rules.get(chosen, [])
        new_items = 0
        if ok:
            for r in results:
                if r not in inventory:
                    inventory.add(r); new_items += 1
        history.append({
            "iter": iteration,
            "pair": list(chosen),
            "valid": bool(ok),
            "results": results if ok else [],
            "new_items": new_items,
            "inventory_size": len(inventory),
            "empowerment": emp_val,
            "mode": mode,
            "fallback_reason": (fallback_reason if mode=="fallback" else None)
        })
        inventory_sizes.append(len(inventory))
    return list(inventory), history, inventory_sizes

if __name__ == "__main__":
    rules = None
    try:
        if JSON_PATH and os.path.exists(JSON_PATH):
            rules = load_elements_rules(JSON_PATH, ordered=ORDERED)
        else:
            raise RuntimeError("elements.JSON not found")
    except Exception:
        print("Failed to load elements.JSON.")
        print(traceback.format_exc())
        rules = {}

    # 1. Run with fallback (use_fallback = True)
    all_inv_sizes_fallback = []
    for trial in range(1, TRIALS+1):
        print(f"\nTRIAL #{trial}... (with fallback)")
        inventory, history, inv_sizes = run_empowerment_loop(START_INVENTORY, rules, ordered=ORDERED, allow_self=ALLOW_SELF, model=MODEL_NAME, max_iters=MAX_ITERS, depth=EMPOWER_DEPTH, use_fallback=True)
        # Pad if short
        if len(inv_sizes) < MAX_ITERS+1:
            inv_sizes += [inv_sizes[-1]] * (MAX_ITERS+1 - len(inv_sizes))
        all_inv_sizes_fallback.append(inv_sizes)

    avg_inv_fallback = []
    for i in range(MAX_ITERS+1):
        avg = sum(trial[i] for trial in all_inv_sizes_fallback) / TRIALS
        avg_inv_fallback.append(avg)

    # 2. Run WITHOUT fallback (use_fallback = False)
    all_inv_sizes_no_fallback = []
    for trial in range(1, TRIALS+1):
        print(f"\nTRIAL #{trial}... (NO fallback)")
        inventory, history, inv_sizes = run_empowerment_loop(START_INVENTORY, rules, ordered=ORDERED, allow_self=ALLOW_SELF, model=MODEL_NAME, max_iters=MAX_ITERS, depth=EMPOWER_DEPTH, use_fallback=False)
        if len(inv_sizes) < MAX_ITERS+1:
            inv_sizes += [inv_sizes[-1]] * (MAX_ITERS+1 - len(inv_sizes))
        all_inv_sizes_no_fallback.append(inv_sizes)

    avg_inv_no_fallback = []
    for i in range(MAX_ITERS+1):
        avg = sum(trial[i] for trial in all_inv_sizes_no_fallback) / TRIALS
        avg_inv_no_fallback.append(avg)

    # ----------------- Plotting both lines -----------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, MAX_ITERS+1), avg_inv_fallback, label="With Fallback", marker='o')
    plt.plot(range(0, MAX_ITERS+1), avg_inv_no_fallback, label="No Fallback", marker='x')
    plt.xticks(range(0, MAX_ITERS+1, 5))
    plt.xlabel("Iteration")
    plt.ylabel("Average Inventory Size")
    plt.title(f"Average Inventory Size vs Iteration\n(With and Without Fallback, {TRIALS} Trials)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"avg_inventory_size_vs_iter_fallback_vs_nofallback.png", dpi=150, bbox_inches="tight")
    plt.show()
