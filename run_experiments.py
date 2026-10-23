"""Reproduces all results and figures. Run: python run_experiments.py"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collusion.simulators import simulate_rps, simulate_hub, simulate_chain, simulate_separate
from collusion.evaluation import run_trials, wilson_interval, classify

N_TRIALS = 200
Q = 0.5
P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# ---- Experiment 1: accuracy vs collusion strength, four scenarios ----
scenarios = {
    "Basic (no decoy)": lambda n, p, s: simulate_rps(n, p, s),
    "Hub (q=0.5)": lambda n, p, s: simulate_hub(n, p, Q, s),
    "Chain (q=0.5)": lambda n, p, s: simulate_chain(n, p, Q, s),
    "Separate decoy (q=0.5)": lambda n, p, s: simulate_separate(n, p, Q, s),
}
rows, disagree = [], {}
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
for ax, (name, sim) in zip(axes.ravel(), scenarios.items()):
    ni_res, cv_res, dis = [], [], 0
    for p in P_VALUES:
        h_ni, h_cv, tops = run_trials(sim, 500, p, N_TRIALS)
        dis += sum(a != b for a, b in tops)
        ni_res.append((h_ni / N_TRIALS, *wilson_interval(h_ni, N_TRIALS)))
        cv_res.append((h_cv / N_TRIALS, *wilson_interval(h_cv, N_TRIALS)))
        rows.append([name, p, h_ni / N_TRIALS, h_cv / N_TRIALS])
    disagree[name] = dis
    print(f"{name}: methods disagreed in {dis} of {N_TRIALS * len(P_VALUES)} trials")
    for res, color, style, label in [(cv_res, "tab:orange", "s--", "Cramér's V"),
                                     (ni_res, "tab:blue", "o-", "Net influence")]:
        mid, lo, hi = zip(*res)
        ax.fill_between(P_VALUES, lo, hi, color=color, alpha=0.2)
        ax.plot(P_VALUES, mid, style, color=color, label=label)
    ax.axhline(1 / 6, color="gray", linestyle=":", label="Chance (1/6)")
    ax.set_title(name)
    ax.set_xlabel("Collusion strength p")
    ax.grid(alpha=0.3)
axes[0, 0].set_ylabel("Fraction ranking real pair first")
axes[1, 0].set_ylabel("Fraction ranking real pair first")
axes[0, 0].legend(loc="lower right")
fig.suptitle("Accuracy vs collusion strength (500 rounds, 200 trials, shaded = 95% CI)")
plt.tight_layout()
plt.savefig("figures/fig_scenarios.png", dpi=150)
plt.close()
with open("results/accuracy_vs_p.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "p", "net_influence_hit_rate", "cramers_v_hit_rate"])
    w.writerows(rows)
print("Total disagreements:", sum(disagree.values()), "of", N_TRIALS * len(P_VALUES) * 4)

# ---- Experiment 2: accuracy vs number of rounds (separate decoy, p=0.6, q=0.5) ----
rounds_list = [100, 250, 500, 1000, 2000]
res = {"NI": [], "CV": []}
round_rows = []
for n_rounds in rounds_list:
    h_ni, h_cv, tops = run_trials(lambda n, p, s: simulate_separate(n, p, 0.5, s), n_rounds, 0.6, N_TRIALS)
    d = sum(a != b for a, b in tops)
    print(f"rounds={n_rounds:<5} NI: {h_ni/N_TRIALS:.2f}   CV: {h_cv/N_TRIALS:.2f}   disagree: {d}/{N_TRIALS}")
    res["NI"].append((h_ni / N_TRIALS, *wilson_interval(h_ni, N_TRIALS)))
    res["CV"].append((h_cv / N_TRIALS, *wilson_interval(h_cv, N_TRIALS)))
    round_rows.append([n_rounds, h_ni / N_TRIALS, h_cv / N_TRIALS, d])
fig, ax = plt.subplots(figsize=(6, 4))
for key, color, style, label in [("CV", "tab:orange", "s--", "Cramér's V"),
                                 ("NI", "tab:blue", "o-", "Net influence")]:
    mid, lo, hi = zip(*res[key])
    ax.fill_between(rounds_list, lo, hi, color=color, alpha=0.2)
    ax.plot(rounds_list, mid, style, color=color, label=label)
ax.axhline(1 / 6, color="gray", linestyle=":", label="Chance (1/6)")
ax.set_xscale("log")
ax.set_xlabel("Rounds per game")
ax.set_ylabel("Fraction ranking the real pair first")
ax.set_title("Accuracy vs game length (p=0.6, q=0.5, 200 trials)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig_rounds.png", dpi=150)
plt.close()
with open("results/accuracy_vs_rounds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rounds", "net_influence_hit_rate", "cramers_v_hit_rate", "disagreements_of_200"])
    w.writerows(round_rows)

# ---- Experiment 3: who was right when the methods disagreed ----
settings = [("Hub q=0.5", lambda n, p, s: simulate_hub(n, p, 0.5, s)),
            ("Chain q=0.5", lambda n, p, s: simulate_chain(n, p, 0.5, s)),
            ("Hub q=0.3", lambda n, p, s: simulate_hub(n, p, 0.3, s)),
            ("Hub q=0.7", lambda n, p, s: simulate_hub(n, p, 0.7, s))]
cls_rows = []
for name, sim in settings:
    all_tops = []
    for p in P_VALUES:
        all_tops += run_trials(sim, 500, p, N_TRIALS)[2]
    a, b, c = classify(all_tops)
    print(f"{name}: NI right only: {a} | CV right only: {b} | both wrong: {c}")
    cls_rows.append([name, a, b, c])
with open("results/disagreements.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["setting", "ni_right_only", "cv_right_only", "both_wrong"])
    w.writerows(cls_rows)
