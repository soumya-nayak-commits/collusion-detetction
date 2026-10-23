import numpy as np
from collusion.simulators import simulate_rps
from collusion.detectors import mutual_information, cramers_v, pair_scores, baseline_scores


def test_simulator_shape_and_values():
    d = simulate_rps(1000, 0.5, seed=42)
    assert d.shape == (1000, 4)
    assert set(np.unique(d)) <= {0, 1, 2}


def test_simulator_collusion_rate_extremes():
    d1 = simulate_rps(1000, 1.0, seed=1)
    assert np.mean(d1[:, 3] == (d1[:, 2] + 1) % 3) == 1.0
    d0 = simulate_rps(10000, 0.0, seed=1)
    assert abs(np.mean(d0[:, 3] == (d0[:, 2] + 1) % 3) - 1 / 3) < 0.03


def test_mutual_information_independent_is_near_zero():
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 3, 20000), rng.integers(0, 3, 20000)
    assert mutual_information(x, y) < 0.01


def test_mutual_information_identical_is_log2_3():
    x = np.tile([0, 1, 2], 1000)
    assert abs(mutual_information(x, x) - np.log2(3)) < 1e-9


def test_cramers_v_bounds():
    x = np.tile([0, 1, 2], 1000)
    assert abs(cramers_v(x, x) - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    assert cramers_v(rng.integers(0, 3, 20000), rng.integers(0, 3, 20000)) < 0.05


def test_both_methods_find_planted_pair():
    d = simulate_rps(5000, 0.8, seed=7)
    ni, cv = pair_scores(d), baseline_scores(d)
    assert max(ni, key=ni.get) == (2, 3)
    assert max(cv, key=cv.get) == (2, 3)
