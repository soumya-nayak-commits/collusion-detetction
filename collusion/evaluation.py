"""Evaluation helpers: hit rates, Wilson intervals, disagreement classification."""
import numpy as np
from .detectors import pair_scores, baseline_scores

TRUE_PAIR = (2, 3)


def wilson_interval(hits, n, z=1.96):
    """95% Wilson score interval for a hit rate hits/n."""
    phat = hits / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def run_trials(sim_fn, n_rounds, p, n_trials=200):
    """Return hits for net influence, hits for Cramer's V, and a list of (top_ni, top_cv)."""
    hits_ni = hits_cv = 0
    tops = []
    for seed in range(n_trials):
        data = sim_fn(n_rounds, p, seed)
        ni, cv = pair_scores(data), baseline_scores(data)
        top_ni, top_cv = max(ni, key=ni.get), max(cv, key=cv.get)
        hits_ni += top_ni == TRUE_PAIR
        hits_cv += top_cv == TRUE_PAIR
        tops.append((top_ni, top_cv))
    return hits_ni, hits_cv, tops


def classify(tops):
    """Count trials where the methods disagreed: (NI only right, CV only right, both wrong)."""
    ni_only = cv_only = both_wrong = 0
    for top_ni, top_cv in tops:
        if top_ni != top_cv:
            if top_ni == TRUE_PAIR:
                ni_only += 1
            elif top_cv == TRUE_PAIR:
                cv_only += 1
            else:
                both_wrong += 1
    return ni_only, cv_only, both_wrong
