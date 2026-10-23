"""Four-agent Rock-Paper-Scissors simulators. Moves: 0=rock, 1=paper, 2=scissors.
(m + 1) % 3 is the move that beats m. The real colluding pair is always (2, 3)."""
import numpy as np

MOVE_NAMES = {0: "rock", 1: "paper", 2: "scissors"}


def simulate_rps(n_rounds, p, seed=0):
    """Basic: agent 3 plays the move that beats agent 2's move with prob p."""
    rng = np.random.default_rng(seed)
    moves = rng.integers(0, 3, size=(n_rounds, 4))
    beats = (moves[:, 2] + 1) % 3
    follow = rng.random(n_rounds) < p
    moves[:, 3] = np.where(follow, beats, moves[:, 3])
    return moves


def simulate_hub(n_rounds, p, q, seed=0):
    """Hub decoy: agents 1 and 2 copy agent 0 with prob q; real pair (2, 3) with prob p."""
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 3, size=(n_rounds, 4))
    for a in (1, 2):
        m[:, a] = np.where(rng.random(n_rounds) < q, m[:, 0], m[:, a])
    m[:, 3] = np.where(rng.random(n_rounds) < p, (m[:, 2] + 1) % 3, m[:, 3])
    return m


def simulate_chain(n_rounds, p, q, seed=0):
    """Chain decoy: 1 copies 0, 2 copies 1 (each with prob q); real pair (2, 3) with prob p."""
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 3, size=(n_rounds, 4))
    m[:, 1] = np.where(rng.random(n_rounds) < q, m[:, 0], m[:, 1])
    m[:, 2] = np.where(rng.random(n_rounds) < q, m[:, 1], m[:, 2])
    m[:, 3] = np.where(rng.random(n_rounds) < p, (m[:, 2] + 1) % 3, m[:, 3])
    return m


def simulate_separate(n_rounds, p, q, seed=0):
    """Separate decoy: pair (0, 1) with strength q, real pair (2, 3) with strength p."""
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 3, size=(n_rounds, 4))
    m[:, 1] = np.where(rng.random(n_rounds) < q, (m[:, 0] + 1) % 3, m[:, 1])
    m[:, 3] = np.where(rng.random(n_rounds) < p, (m[:, 2] + 1) % 3, m[:, 3])
    return m
