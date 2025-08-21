from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Any
from collections import deque
from openai import OpenAI
from load_dotenv import load_dotenv
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
ORDERED = False
ALLOW_SELF = False
EMPOWER_DEPTH = 2
OUTPUT_HISTORY = "run_history.jsonl"
OUTPUT_SUMMARY = True
TRIED_CONTEXT_MAX = 700
OPENAI_TEMPERATURE = 2.0
OPENAI_PRESENCE_PENALTY = 0.9
OPENAI_FREQUENCY_PENALTY = 0.7
FLATLINE_WINDOW = 4
FALLBACK_TOPK = 6
LLM_MAX_RETRIES = 1
TRIALS = 1
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 1.0
TEMPERATURE_STEP_UP = 0.1
TEMPERATURE_STEP_DOWN = 0.1

TEMP_ADJUST_BLOCK = 5 
PAIR_RE = re.compile(
    r"""
    [\(\[]\s*
    ["']([^"'\\n]+)["']\s*,\s*
    ["']([^"'\\n]+)["']\s*,?\s*
    [\)\]]
    """,
    re.VERBOSE,
)

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
    s = text.strip()
    try:
        data = json.loads(s)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return (str(data[0]).strip(), str(data[1]).strip())
    except Exception:
        pass
    if s.startswith("```") and s.endswith("```"):
        inner = s[3:-3].strip()
        inner = re.sub(r"^[A-Za-z0-9_+\-.\s]*\n", "", inner, count=1)
        s = inner.strip()
    m = PAIR_RE.search(s)
    if m:
        return (m.group(1).strip(), m.group(2).strip())
    m2 = re.search(r'["\']([^"\'\n]+)["\']\s*,\s*["\']([^"\'\n]+)["\']', s)
    if m2:
        return (m2.group(1).strip(), m2.group(2).strip())
    return None

from typing import Sequence, Set, Tuple, List, Union, Iterable

Pair = Tuple[str, str]
LabeledPair = Tuple[str, str, bool]

TRIED_CONTEXT_MAX = 120

def _stringify_pairs(pairs: Iterable[Pair], cap: int) -> str:
    lst = list(pairs)[-cap:]
    return ', '.join(f"('{a}','{b}')" for (a, b) in lst)

def _tally_items(pairs: Iterable[Pair], cap: int = 20) -> List[Tuple[str, int]]:
    counts = {}
    for a, b in pairs:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:cap]

def build_prompt(
    inventory: Sequence[str],
    tried_combos: Union[Set[Pair], Sequence[Union[Pair, LabeledPair]]],
    item_emp_hints: List[Tuple[str, int]],
    taboo_items: Sequence[str],
    escape_mode: bool,
    ordered: bool,
    allow_self: bool,
) -> Tuple[str, str]:
    # --- stringify core lists ---
    items_list_str = ', '.join(f"'{x}'" for x in inventory)
    taboo_str = ', '.join(f"'{x}'" for x in taboo_items[:10])
    emp_hint_str = ', '.join(f"('{x}', {s})" for x, s in item_emp_hints[:12])

    # --- normalize tried_combos into: blacklist, valid_hist, invalid_hist ---
    blacklist_pairs: List[Pair] = []
    valid_hist: List[Pair] = []
    invalid_hist: List[Pair] = []

    # Accept: Set[(a,b)], List[(a,b)], or List[(a,b,bool)]
    for entry in list(tried_combos):
        if isinstance(entry, tuple) and len(entry) == 3:
            a, b, was_valid = entry  # labeled
            blacklist_pairs.append((a, b))
            (valid_hist if was_valid else invalid_hist).append((a, b))
        else:
            # unlabeled pair → only blacklist
            a, b = entry  # type: ignore
            blacklist_pairs.append((a, b))

    tried_str = _stringify_pairs(blacklist_pairs, TRIED_CONTEXT_MAX)

    # Histories (cap to keep prompt compact)
    VALID_HISTORY_MAX = 80
    INVALID_HISTORY_MAX = 80
    valid_str = _stringify_pairs(valid_hist, VALID_HISTORY_MAX)
    invalid_str = _stringify_pairs(invalid_hist, INVALID_HISTORY_MAX)

    # Per-item priors (only if we actually have labels)
    valid_item_counts_str = ''
    invalid_item_counts_str = ''
    if valid_hist or invalid_hist:
        valid_item_counts = _tally_items(valid_hist)
        invalid_item_counts = _tally_items(invalid_hist)
        valid_item_counts_str = ', '.join(f"('{x}', {c})" for x, c in valid_item_counts)
        invalid_item_counts_str = ', '.join(f"('{x}', {c})" for x, c in invalid_item_counts)

    # --- system & task framing ---
    sys_msg = "Return exactly one JSON array of two strings like [\"a\",\"b\"]. Output only that JSON."

    stage = (
        "Two-item crafting game. Empowerment is the tendency for a pair to unlock many new items over the next few steps. "
        "Use a loss-averse utility: repeated/malformed/non-inventory/self-combo is very negative; invalid is negative; "
        "valid & novel with high-empowerment items is strongly positive."
    )

    # Hidden reasoning (no leakage)
    reasoning_instructions = (
        "Think step by step silently (do not reveal reasoning). "
        "1) From the inventory, enumerate promising candidate pairs. "
        "2) Filter out blacklist, taboo, disallowed self-combos, and non-inventory. "
        "3) Use tried history: prefer motifs/items that succeeded; avoid motifs/items overrepresented in failures; "
        "   avoid near-duplicates of failed pairs; consider analogies to past successes with high-emp items. "
        "4) Score by empowerment hints, novelty vs. tried list, success-leaning priors, usage balance, and ordering rules. "
        "5) Select the single best pair. Do NOT explain; only output the JSON array."
    )

    checklist = (
        "Checklist before output: JSON array only; both items in inventory; not in blacklist; not taboo; "
        f"{'self-combos allowed' if allow_self else 'no self-combo'}; prefer higher empowerment items; "
        "avoid items/patterns frequent in invalid history; favor successful motifs from valid history; "
        "avoid pairs repeated often."
    )

    escape_note = (
        "Escape mode: recent growth stalled; favor high-empowerment, underused directions and analogies to past successes."
        if escape_mode else ""
    )

    # --- assemble user message ---
    history_blocks = []
    if valid_hist:
        history_blocks.append(f"History — VALID pairs (use as motifs): [{valid_str}]")
    if invalid_hist:
        history_blocks.append(f"History — INVALID pairs (avoid/penalize): [{invalid_str}]")
    if valid_item_counts_str or invalid_item_counts_str:
        history_blocks.append(f"History signals — item successes: [{valid_item_counts_str}]")
        history_blocks.append(f"History signals — item failures: [{invalid_item_counts_str}]")

    user_msg = (
        f"{stage}\n"
        f"{reasoning_instructions}\n"
        f"Inventory: [{items_list_str}]\n"
        f"Blacklist (do not repeat): [{tried_str}]\n"
        f"Taboo items (must not use): [{taboo_str}]\n"
        f"Empowerment hints: [{emp_hint_str}]\n"
        + ("\n".join(history_blocks) + "\n" if history_blocks else "")
        + f"Rules: {'ordered allowed' if ordered else 'order does not matter'}\n"
        f"{escape_note}\n"
        f"{checklist}\n"
        "Output only one JSON array like [\"a\",\"b\"]."
    )

    return sys_msg, user_msg


def llm_propose_one_pair(inventory: Sequence[str], tried_combos: Set[Pair], item_emp_hints: List[Tuple[str,int]], taboo_items: Sequence[str], escape_mode: bool, ordered: bool, allow_self: bool, model: str, temperature: float) -> Optional[Pair]:
    sys_msg, user_msg = build_prompt(inventory, tried_combos, item_emp_hints, taboo_items, escape_mode, ordered, allow_self)
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            presence_penalty=OPENAI_PRESENCE_PENALTY,
            frequency_penalty=OPENAI_FREQUENCY_PENALTY,
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
        )
        return parse_single_tuple((resp.choices[0].message.content or "").strip())
    except Exception:
        print(f"[red]Error in LLM response: {traceback.format_exc()}[/red]")
        return None

def run_empowerment_loop(
    start_inventory: Iterable[str],
    rules: Dict[Pair, List[str]],
    ordered: bool=False,
    allow_self: bool=False,
    model: str=MODEL_NAME,
    max_iters: int=50,
    depth: int=2
) -> Tuple[List[str], List[Dict[str,Any]]]:
    inventory: Set[str] = set(start_inventory)
    tried_combos: Set[Pair] = set()
    inventory_sizes: List[int] = [len(inventory)]
    history: List[Dict[str,Any]] = []

    temperature = max(TEMPERATURE_MIN, min(TEMPERATURE_MAX, OPENAI_TEMPERATURE))

    block_valids: deque[bool] = deque(maxlen=TEMP_ADJUST_BLOCK)

    for iteration in range(1, max_iters+1):
        escape = is_flatline(inventory_sizes, window=FLATLINE_WINDOW)
        tabu_items: List[str] = []
        item_emp_hints = per_item_empowerment(list(inventory), tried_combos, rules, depth=depth)
        chosen: Optional[Pair] = None
        reject_reasons: List[str] = []

        for _ in range(LLM_MAX_RETRIES):
            llm_pair = llm_propose_one_pair(
                list(inventory), tried_combos, item_emp_hints, tabu_items,
                escape, ordered, allow_self, model, temperature
            )
            if llm_pair is None:
                reject_reasons.append("malformed")
                continue
            a, b = llm_pair
            if (not allow_self and a == b) or (a not in inventory or b not in inventory):
                reject_reasons.append("bad_output")
                continue
            key = _canon_pair(a, b, ordered)
            if key in tried_combos:
                reject_reasons.append("repeat")
                continue
            chosen = key
            break

        # Default outcome values
        ok = False
        results: List[str] = []
        new_items = 0
        emp_val = 0

        if chosen is None:
            # Treat as invalid attempt for block accounting
            block_valids.append(False)
            history.append({
                "iter": iteration,
                "pair": [],
                "valid": False,
                "results": [],
                "new_items": 0,
                "event": "no_llm_choice",
                "reasons": reject_reasons,
                "temperature": temperature
            })
            inventory_sizes.append(len(inventory))
        else:
            tried_combos.add(chosen)
            emp_val = pair_empowerment(chosen, set(inventory), rules, depth=depth)
            ok = chosen in rules
            results = rules.get(chosen, [])
            if ok:
                for r in results:
                    if r not in inventory:
                        inventory.add(r)
                        new_items += 1
                block_valids.append(True)
            else:
                block_valids.append(False)

            history.append({
                "iter": iteration,
                "pair": list(chosen),
                "valid": bool(ok),
                "results": results if ok else [],
                "new_items": new_items,
                "inventory_size": len(inventory),
                "empowerment": emp_val,
                "mode": "llm",
                "fallback_reason": None,
                "temperature": temperature
            })
            inventory_sizes.append(len(inventory))

        temp_adjust_info = None
        if iteration % TEMP_ADJUST_BLOCK == 0 and len(block_valids) == TEMP_ADJUST_BLOCK:
            num_invalid = TEMP_ADJUST_BLOCK - sum(1 for v in block_valids if v)
  
            if num_invalid == 3:
                temperature = max(TEMPERATURE_MIN, temperature - TEMPERATURE_STEP_DOWN)
                rule = "decrease_all_invalid"
            elif num_invalid == 1:
                temperature = max(TEMPERATURE_MIN, temperature - TEMPERATURE_STEP_DOWN)
                rule = "decrease_one_invalid_consistency"
            else:
                temperature = min(TEMPERATURE_MAX, temperature + TEMPERATURE_STEP_UP)
                rule = "increase_explore"

            temp_adjust_info = {
                "block_valids": list(block_valids),
                "num_invalid": num_invalid,
                "rule": rule,
                "new_temperature": temperature,
            }
            block_valids.clear()  # start a fresh block

            # Attach adjustment info to the latest history row for visibility
            if history:
                history[-1]["temp_adjust_after_block"] = temp_adjust_info
                history[-1]["temperature"] = temperature

    return list(inventory), history

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

    all_inv_sizes = []
    fallback_attempts = []
    for trial in range(1, TRIALS+1):
        print(f"\nTRIAL #{trial}...")
        print(f"\nRunning Empowerment Loop...")
        inventory, history = run_empowerment_loop(START_INVENTORY, rules, ordered=ORDERED, allow_self=ALLOW_SELF, model=MODEL_NAME, max_iters=MAX_ITERS, depth=EMPOWER_DEPTH)
        attempts = sum(1 for h in history if "pair" in h or h.get("event") in {"no_choice"})
        valids   = sum(1 for h in history if h.get("valid") and h.get("mode")!= "fallback")
        avg_emp_all = sum(float(h.get("empowerment",0) or 0) for h in history if "empowerment" in h)
        avg_emp_all = (avg_emp_all / max(1, sum(1 for h in history if "empowerment" in h)))
        avg_emp_valid = sum(float(h.get("empowerment",0) or 0) for h in history if h.get("valid") and h.get("mode")!= "fallback")
        avg_emp_valid = (avg_emp_valid / max(1, sum(1 for h in history  if h.get("valid") and h.get("mode")!= "fallback")))
        with open(OUTPUT_HISTORY, "w", encoding="utf-8") as f:
            for h in history:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")

    def mode_stats(mode_name: str):
        rows = [h for h in history if h.get("mode")==mode_name]
        A = len(rows)
        V = sum(1 for h in rows if h.get("valid"))
        Eall = [float(h.get("empowerment",0) or 0) for h in rows if "empowerment" in h]
        Eval = [float(h.get("empowerment",0) or 0) for h in rows if h.get("valid")]
        NI = sum(int(h.get("new_items",0) or 0) for h in rows if h.get("valid"))
        return {
            "attempts": A,
            "valid": V,
            "valid_rate": (V/max(1,A)),
            "avg_emp_all": (sum(Eall)/max(1,len(Eall))),
            "avg_emp_valid": (sum(Eval)/max(1,len(Eval))),
            "new_items_total": NI,
        }
    os.makedirs("./without", exist_ok=True)
    OUTPUT_HISTORY = f"./without/trial_{trial}_history.jsonl"
    with open(OUTPUT_HISTORY, "w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
            
    m_llm = mode_stats("llm")
    m_fb  = mode_stats("fallback")
    fb_reasons = {}
    for h in history:
        if h.get("mode")=="fallback":
            r = h.get("fallback_reason") or "unknown"
            fb_reasons[r] = fb_reasons.get(r,0)+1

        if OUTPUT_SUMMARY:
            print(f"\n=== TRIAL #{trial} SUMMARY ===")
            print(f"Attempts: {attempts} | Valid: {valids} | Valid rate: {(valids/attempts*100 if attempts else 0):.1f}%")
            print(f"Final inventory size: {len(inventory)}")
            print(f"Avg empowerment (all): {avg_emp_all:.3f} | (valid only): {avg_emp_valid:.3f}")
            print("\n--- By mode ---")
            print(f"LLM -> attempts {m_llm['attempts']}, valid {m_llm['valid']} ({m_llm['valid_rate']*100:.1f}%), avg_emp_all {m_llm['avg_emp_all']:.3f}, avg_emp_valid {m_llm['avg_emp_valid']:.3f}, new_items {m_llm['new_items_total']}")
            print(f"Fallback -> attempts {m_fb['attempts']}, valid {m_fb['valid']} ({m_fb['valid_rate']*100:.1f}%), avg_emp_all {m_fb['avg_emp_all']:.3f}, avg_emp_valid {m_fb['avg_emp_valid']:.3f}, new_items {m_fb['new_items_total']}")
            if fb_reasons:
                print("Fallback reasons:", ", ".join(f"{k}:{v}" for k,v in fb_reasons.items()))

            inv_sizes = [h.get("inventory_size") for h in history if "inventory_size" in h and h.get("mode") != "fallback"]

            if len(inv_sizes) < MAX_ITERS:
                inv_sizes += [inv_sizes[-1]] * (MAX_ITERS - len(inv_sizes))
            all_inv_sizes.append(inv_sizes)
            fallback_attempts.append(m_fb['attempts'])

            iters = list(range(1, len(inv_sizes) + 1))

            print(f"\n==============================")

            plt.figure(figsize=(8, 5))
            plt.plot(iters, inv_sizes, marker="o")
            plt.xticks(range(0, len(inv_sizes) + 1, 5))
            plt.yticks(range(0, max(inv_sizes) + 1, 2))
            plt.xlabel("Iteration")
            plt.ylabel("Inventory Size")
            plt.title("Inventory Size vs Iteration")
            plt.grid(True)
            plt.show()

    avg_inv = []
    for i in range(MAX_ITERS):
        avg = sum(trial[i] for trial in all_inv_sizes) / TRIALS
        avg_inv.append(avg)
    with open("avg_inventory_sizes.json", "w", encoding="utf-8") as f:
        json.dump(avg_inv, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, MAX_ITERS + 1), avg_inv, marker="o")
    plt.xticks(range(0, MAX_ITERS + 1, 5))
    plt.xlabel("Iteration")
    plt.ylabel("Average Inventory Size")
    plt.title(f"Average Inventory Size vs Iteration (over {TRIALS} Trials)")
    plt.grid(True)
    plt.savefig(f"{MAX_ITERS}iters_{TRIALS}trials_fallback.png", dpi=150, bbox_inches="tight")
