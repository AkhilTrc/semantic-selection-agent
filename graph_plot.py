import os
import json
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

DIR_WITHOUT = "./without"
DIR_WITH    = "./with"
FILENAME_SUFFIX = "_history.jsonl"
start_iter_at = 1
MAX_TRIALS = 10


def load_trial_histories(data_dir, max_trials=None):
    if not os.path.isdir(data_dir):
        return [], 0
    files = [f for f in os.listdir(data_dir) if f.endswith(FILENAME_SUFFIX)]
    if max_trials is not None:
        files = files[:max_trials]
    trial_iter_size = []
    for file in files:
        path = os.path.join(data_dir, file)
        it2size = {}
        running_size = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "iter" not in rec:
                    continue
                it = int(rec["iter"])
                if "inventory_size" in rec and rec["inventory_size"] is not None:
                    inv = int(rec["inventory_size"])
                    it2size[it] = inv
                    running_size = inv
                else:
                    ni = int(rec.get("new_items", 0) or 0)
                    if running_size is None:
                        running_size = 0
                    running_size += ni
                    it2size[it] = running_size
        if it2size:
            sorted_iters = sorted(it2size.keys())
            filled = {}
            last_size = None
            for k in range(sorted_iters[0], sorted_iters[-1] + 1):
                if k in it2size:
                    last_size = it2size[k]
                if last_size is not None:
                    filled[k] = last_size
            trial_iter_size.append(OrderedDict(sorted(filled.items())))
    if not trial_iter_size:
        return [], 0
    dir_max_iter = max(d and max(d.keys()) or 0 for d in trial_iter_size)
    return trial_iter_size, dir_max_iter

def pad_trials(trials, max_iters, start_iter_at=1):
    padded = []
    for d in trials:
        if not d:
            padded.append([0] * max_iters)
            continue
        first_iter = min(d.keys())
        last_iter = max(d.keys())
        series = []
        carry = None
        for it in range(start_iter_at, max_iters + 1):
            if it in d:
                carry = d[it]
            elif carry is None and it < first_iter:
                carry = 0
            series.append(carry if carry is not None else 0)
        if series:
            last_known = series[last_iter - start_iter_at] if last_iter >= start_iter_at else 0
            for i in range(last_iter - start_iter_at + 1, max_iters - start_iter_at + 1):
                if 0 <= i < len(series):
                    series[i] = last_known
        padded.append(series)
    return padded

def mean_and_std(padded_trials):
    if not padded_trials:
        return [], []
    n = len(padded_trials)
    L = len(padded_trials[0])
    avg, std = [], []
    for i in range(L):
        vals = [trial[i] for trial in padded_trials]
        m = sum(vals) / n
        avg.append(m)
        var = sum((v - m) ** 2 for v in vals) / n
        std.append(math.sqrt(var))
    return avg, std

trials_without, max_wo = load_trial_histories(DIR_WITHOUT, MAX_TRIALS)
trials_with,    max_wi = load_trial_histories(DIR_WITH, MAX_TRIALS)

if not trials_without and not trials_with:
    raise SystemExit(f"No usable data found in {DIR_WITHOUT!r} or {DIR_WITH!r} ending with {FILENAME_SUFFIX!r}.")

OVERALL_MAX_ITERS = max(max_wo, max_wi)
if OVERALL_MAX_ITERS == 0:
    raise SystemExit("No iterations found.")

padded_without = pad_trials(trials_without, OVERALL_MAX_ITERS, start_iter_at=start_iter_at)
padded_with    = pad_trials(trials_with,    OVERALL_MAX_ITERS, start_iter_at=start_iter_at)

avg_wo, std_wo = mean_and_std(padded_without) if padded_without else ([], [])
avg_wi, std_wi = mean_and_std(padded_with)    if padded_with    else ([], [])

# Extra data from other experiments
avg_flair = pd.read_csv("avg_inventory_sizes_50iters_10trials_fliar.csv")["avg_iter"].tolist()[:50]
avg_base = pd.read_csv("avg_inventory_sizes_50iters_10trials_base.csv")["avg_iter"].tolist()[:50]
avg_rec = pd.read_csv("avg_inventory_sizes_50iters_10trials_rec.csv")["avg_iter"].tolist()[:50]
avg_trueemp = pd.read_csv("avg_inventory_sizes_50iters_10trials_trueemp.csv")["avg_iter"].tolist()[:50]
iters = list(range(start_iter_at, OVERALL_MAX_ITERS + 1))

if avg_wo:
    with open("avg_inventory_sizes_without.json", "w", encoding="utf-8") as f:
        json.dump({iters[i]: avg_wo[i] for i in range(len(iters))}, f, ensure_ascii=False, indent=2)
if avg_wi:
    with open("avg_inventory_sizes_with.json", "w", encoding="utf-8") as f:
        json.dump({iters[i]: avg_wi[i] for i in range(len(iters))}, f, ensure_ascii=False, indent=2)

plt.figure(figsize=(11, 6.5))
for series in padded_without:
    plt.plot(iters, series, alpha=0.08, linewidth=1, label="_nolegend_")
for series in padded_with:
    plt.plot(iters, series, alpha=0.08, linewidth=1, label="_nolegend_")

if avg_wo:
    plt.plot(iters, avg_wo, marker=None, linewidth=2.2, label=f"Dynamic Temp (avg, n={len(padded_without)})")
if avg_wi:
    plt.plot(iters, avg_wi, marker=None, linewidth=2.2, label=f"Fallback (avg, n={len(padded_with)})")
if avg_flair:
    plt.plot(iters, avg_flair, marker=None, linewidth=2.2, label="FLAIR (avg, n=10)")
if avg_base:
    plt.plot(iters, avg_base, marker=None, linewidth=2.2, label="Base (avg, n=10)")
if avg_rec:
    plt.plot(iters, avg_rec, marker=None, linewidth=2.2, label="Rec (avg, n=10)")
if avg_trueemp:
    plt.plot(iters, avg_trueemp, marker=None, linewidth=2.2, label="TrueEmp (avg, n=10)")


if avg_wo and std_wo:
    upper = [m + s for m, s in zip(avg_wo, std_wo)]
    lower = [m - s for m, s in zip(avg_wo, std_wo)]
    plt.fill_between(iters, lower, upper, alpha=0.18, label="Dynamic Temp ±1σ")
if avg_wi and std_wi:
    upper = [m + s for m, s in zip(avg_wi, std_wi)]
    lower = [m - s for m, s in zip(avg_wi, std_wi)]
    plt.fill_between(iters, lower, upper, alpha=0.18, label="Fallback ±1σ")

title_bits = []
if padded_without:
    title_bits.append(f"without: {len(padded_without)} trials")
if padded_with:
    title_bits.append(f"with: {len(padded_with)} trials")

plt.title(f"Inventory Size vs Iteration ({', '.join(title_bits)})")
plt.xlabel("Iteration")
plt.ylabel("Inventory Size")
plt.grid(True, linestyle="--", alpha=0.6)
xtick_step = max(1, (OVERALL_MAX_ITERS - start_iter_at + 1) // 10)
plt.xticks(range(start_iter_at, OVERALL_MAX_ITERS + 1, xtick_step))
plt.legend()
plt.tight_layout()
plt.savefig(f"inventory_with_vs_without_{OVERALL_MAX_ITERS}iters.png", dpi=200, bbox_inches="tight")
plt.show()
