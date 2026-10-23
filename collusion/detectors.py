"""Pairwise net influence (Bonjour et al., 2022, fully observable case) and Cramer's V baseline."""
import numpy as np


def mutual_information(x, y):
    """Mutual information (bits) between two move sequences with values 0,1,2."""
    joint = np.zeros((3, 3))
    np.add.at(joint, (x, y), 1)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return np.sum(joint[mask] * np.log2(joint[mask] / (px @ py)[mask]))


def influence_matrix(data):
    """gamma[i, j] = mutual information between agents i and j."""
    n = data.shape[1]
    gamma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                gamma[i, j] = mutual_information(data[:, i], data[:, j])
    return gamma


def net_influence(gamma):
    """Gamma[i, j] = gamma[i, j] - max over k (k != i, j) of gamma[k, j]."""
    n = gamma.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                others = [k for k in range(n) if k != i and k != j]
                G[i, j] = gamma[i, j] - max(gamma[k, j] for k in others)
    return G


def pair_scores(data):
    """Score each pair as min(Gamma[i, j], Gamma[j, i])."""
    G = net_influence(influence_matrix(data))
    n = G.shape[0]
    return {(i, j): min(G[i, j], G[j, i]) for i in range(n) for j in range(i + 1, n)}


def cramers_v(x, y):
    """Plain correlation for categories (0 = independent, 1 = perfectly tied)."""
    n = len(x)
    obs = np.zeros((3, 3))
    np.add.at(obs, (x, y), 1)
    exp = obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True) / n
    mask = exp > 0
    chi2 = np.sum((obs[mask] - exp[mask]) ** 2 / exp[mask])
    return np.sqrt(chi2 / (n * 2))


def baseline_scores(data):
    n = data.shape[1]
    return {(i, j): cramers_v(data[:, i], data[:, j])
            for i in range(n) for j in range(i + 1, n)}
