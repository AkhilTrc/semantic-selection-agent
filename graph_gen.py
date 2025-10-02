import json
import matplotlib.pyplot as plt

path = "run_history.jsonl"

iters, sizes = [], []
last_size = None

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "iter" not in rec:
            continue
        iters.append(int(rec["iter"]))
        sz = rec.get("inventory_size")
        if sz is None or rec.get("mode") == "fallback":
            sz = last_size if last_size is not None else 0
        sizes.append(sz)
        last_size = sz

pairs = sorted(zip(iters, sizes), key=lambda t: t[0])
x = [p[0] for p in pairs]
y = [p[1] for p in pairs]

plt.figure()
plt.plot(x, y, marker="o")
# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("Inventory size")
plt.title("Inventory size vs iteration")
plt.tight_layout()
plt.savefig("inventory_vs_iteration.png", dpi=200)
plt.show()