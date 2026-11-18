"""Scenarios with no real collusion, used to measure false alarms."""
import numpy as np


def simulate_null(n_rounds, seed=0):
    """Nobody colludes, no links at all: every agent plays randomly."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(n_rounds, 4))


def simulate_hub_only(n_rounds, q, seed=0):
    """No collusion. Agents 1 and 2 both copy agent 0 with probability q,
    so (1, 2) look linked through agent 0 without any direct link."""
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 3, size=(n_rounds, 4))
    for a in (1, 2):
        m[:, a] = np.where(rng.random(n_rounds) < q, m[:, 0], m[:, a])
    return m


def simulate_hidden_only(n_rounds, q, seed=0):
    """No collusion. A hidden driver (not logged) is copied by agents 0 and 1
    with probability q, so (0, 1) look linked through something we cannot see."""
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 3, size=(n_rounds, 4))
    hidden = rng.integers(0, 3, size=n_rounds)
    for a in (0, 1):
        m[:, a] = np.where(rng.random(n_rounds) < q, hidden, m[:, a])
    return m
