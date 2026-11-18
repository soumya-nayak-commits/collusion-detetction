"""Raw MI scores, the threshold rule, and detection-vs-false-alarm rates."""
import numpy as np
from collusion.detectors import mutual_information, pair_scores, baseline_scores

ALPHA = 0.05
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
TRUE_IDX = PAIRS.index((2, 3))


def mi_scores(data):
    """Raw mutual information per pair: net influence WITHOUT the subtraction."""
    n = data.shape[1]
    return {(i, j): mutual_information(data[:, i], data[:, j])
            for i in range(n) for j in range(i + 1, n)}


def flagged_pairs(scores, alpha=ALPHA):
    """Pairs whose score is at or above the cutoff."""
    return {pair for pair, s in scores.items() if s >= alpha}


def all_scores(game, seeds):
    """Scores of every pair, for each method, over many games."""
    out = {"MI": [], "NI": [], "CV": []}
    for s in seeds:
        d = game(s)
        for key, fn in (("MI", mi_scores), ("NI", pair_scores), ("CV", baseline_scores)):
            sc = fn(d)
            out[key].append([sc[p] for p in PAIRS])
    return {k: np.array(v) for k, v in out.items()}


def rates(S, taus):
    """Per cutoff: share of games where only (2,3) is flagged, and share where any pair is flagged."""
    flag = S[:, :, None] >= taus[None, None, :]
    exact = flag[:, TRUE_IDX, :] & (flag.sum(1) == 1)
    return exact.mean(0), (flag.sum(1) > 0).mean(0)


def best_detection_at_fpr(pos_scores, neg_scores, max_fpr=0.05, n_cutoffs=400):
    """Best correct-detection rate achievable with false-alarm rate <= max_fpr."""
    hi = max(pos_scores.max(), neg_scores.max())
    taus = np.linspace(0, hi, n_cutoffs)
    tpr, _ = rates(pos_scores, taus)
    _, fpr = rates(neg_scores, taus)
    return tpr[fpr <= max_fpr].max() if (fpr <= max_fpr).any() else 0.0
