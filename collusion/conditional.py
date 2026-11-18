"""Conditional MI / Cramér's V: score a pair given both agents' previous-round moves."""
import numpy as np
from collusion.detectors import mutual_information, net_influence, cramers_v


def simulate_memory(n_rounds, p, seed=0):
    """Agent 3 plays (agent 2's move now + agent 2's previous move) % 3 with prob p."""
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 3, size=(n_rounds, 4))
    follow = rng.random(n_rounds) < p
    shifted = (m[1:, 2] + m[:-1, 2]) % 3
    m[1:, 3] = np.where(follow[1:], shifted, m[1:, 3])
    return m


def conditional_mi(x, y, state):
    """I(X;Y | S) = sum over states s of p(s) * I(X;Y | S=s)."""
    total = 0.0
    for s in np.unique(state):
        mask = state == s
        total += mask.mean() * mutual_information(x[mask], y[mask])
    return total


def conditional_gamma(data):
    """gamma[i,j] = I(a_i; a_j | previous-round moves of i and j)."""
    n = data.shape[1]
    g = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                state = data[:-1, i] * 3 + data[:-1, j]
                g[i, j] = conditional_mi(data[1:, i], data[1:, j], state)
    return g


def conditional_pair_scores(data):
    G = net_influence(conditional_gamma(data))
    n = G.shape[0]
    return {(i, j): min(G[i, j], G[j, i]) for i in range(n) for j in range(i + 1, n)}


def conditional_cramers(x, y, state):
    """Cramér's V computed inside each state, averaged by state size."""
    return sum((state == s).mean() * cramers_v(x[state == s], y[state == s])
               for s in np.unique(state))


def conditional_baseline(data):
    n = data.shape[1]
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            state = data[:-1, i] * 3 + data[:-1, j]
            out[(i, j)] = conditional_cramers(data[1:, i], data[1:, j], state)
    return out
