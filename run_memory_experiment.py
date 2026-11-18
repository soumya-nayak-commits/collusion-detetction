"""Experiment 3: collusion that depends on the previous round (memory)."""
import os
import numpy as np, pandas as pd
from collusion.detectors import pair_scores, baseline_scores
from collusion.conditional import (simulate_memory, conditional_pair_scores,
                                   conditional_baseline)

os.makedirs("results", exist_ok=True)
methods = {"flat net influence": pair_scores,
           "conditional net influence": conditional_pair_scores,
           "flat Cramer's V": baseline_scores,
           "conditional Cramer's V": conditional_baseline}
rows = []
for p in [0.0, 0.2, 0.4, 0.8]:
    hits = {name: 0 for name in methods}
    for seed in range(100):
        data = simulate_memory(1000, p, seed)
        for name, fn in methods.items():
            s = fn(data)
            hits[name] += max(s, key=s.get) == (2, 3)
    rows.append({"p": p, **{k: v / 100 for k, v in hits.items()}})
pd.DataFrame(rows).to_csv("results/memory_experiment.csv", index=False)
print(pd.DataFrame(rows))
