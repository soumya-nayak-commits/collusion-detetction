"""Experiment 2: false alarms, detection at 5% false alarms, ROC figure."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collusion.simulators import simulate_rps
from collusion.null_simulators import simulate_null, simulate_hub_only, simulate_hidden_only
from collusion.detectors import pair_scores
from collusion.scoring import (all_scores, rates, best_detection_at_fpr,
                               flagged_pairs, mi_scores)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
N = 200
negs = {
    "hub only (q=0.7)": all_scores(lambda s: simulate_hub_only(500, 0.7, s), range(2000, 2000 + N)),
    "hidden only (q=0.7)": all_scores(lambda s: simulate_hidden_only(500, 0.7, s), range(3000, 3000 + N)),
}

rows = []
for p in [0.2, 0.3, 0.4, 0.5, 0.6]:
    pos = all_scores(lambda s: simulate_rps(500, p, s), range(1000, 1000 + N))
    for name, neg in negs.items():
        row = {"p": p, "scenario": name}
        for key in ("MI", "NI", "CV"):
            row[key] = round(best_detection_at_fpr(pos[key], neg[key]), 2)
        rows.append(row)
pd.DataFrame(rows).to_csv("results/detection_at_5pct_false_alarm.csv", index=False)

games = {"null": lambda s: simulate_null(500, s),
         "hub only (q=0.7)": lambda s: simulate_hub_only(500, 0.7, s),
         "hidden only (q=0.7)": lambda s: simulate_hidden_only(500, 0.7, s)}
fa = []
for name, g in games.items():
    mi_h = ni_h = 0
    for seed in range(N):
        d = g(seed)
        mi_h += len(flagged_pairs(mi_scores(d))) > 0
        ni_h += len(flagged_pairs(pair_scores(d))) > 0
    fa.append({"scenario": name, "MI": mi_h / N, "NI": ni_h / N})
pd.DataFrame(fa).to_csv("results/false_alarm_rate_alpha005.csv", index=False)

pos = all_scores(lambda s: simulate_rps(500, 0.6, s), range(1000, 1000 + N))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, neg) in zip(axes, negs.items()):
    for key in ("MI", "NI", "CV"):
        taus = np.linspace(0, max(pos[key].max(), neg[key].max()), 400)
        tpr, _ = rates(pos[key], taus)
        _, fpr = rates(neg[key], taus)
        ax.plot(fpr, tpr, label=key)
    ax.set_title(name); ax.set_xlabel("False alarm rate"); ax.legend(); ax.grid(alpha=0.3)
axes[0].set_ylabel("Correct detection rate")
plt.tight_layout(); plt.savefig("figures/ROC.png", dpi=150)
print("done")
