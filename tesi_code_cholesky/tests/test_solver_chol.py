# test_solver_chol.py
# Checks of nw_solver_chol.py and nw_loop_levels_chol.py.
# Run (from the main folder, in the Miniforge Prompt):
#   python tests/test_solver_chol.py
# 1. Hand example: chain S - M1 - M2 - D, all g = 1, V_S = 1.
#    M2-D goes from 1 to 3  ->  voltages M1 = 4/7, M2 = 1/7.
#    S-M1 goes from 1 to 3  ->  voltages M1 = 6/7, M2 = 3/7.
# 2. Random changes on a real network: at every step the voltages are equal to
#    nw_solver.solve_node_voltages (difference < 1e-9 of V). With a small
#    maxupd the refactorisation is tested too.
# 3. Whole loop on 10 networks (K = 1 and K = 10) against the fast LU loop:
#    number of switches within 1%. (The sequences can differ on ties.)

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import nanowire_network as nn
from nw_solver import solve_node_voltages
from nw_solver_chol import CholeskyKirchhoff
from nw_loop_levels_fast import run_loop_levels_fast
from nw_loop_levels_chol import run_loop_levels_chol


def test_hand_example():
    junctions = np.array([[0, 1], [1, 2], [2, 3]])
    ck = CholeskyKirchhoff(4, junctions, [0], [3], 1.0, np.array([1.0, 1.0, 1.0]), maxupd=5)
    ck.update_junction(2, 3.0)
    v = ck.solve()
    assert abs(v[1] - 4.0 / 7.0) < 1e-12
    assert abs(v[2] - 1.0 / 7.0) < 1e-12
    ck2 = CholeskyKirchhoff(4, junctions, [0], [3], 1.0, np.array([1.0, 1.0, 1.0]), maxupd=5)
    ck2.update_junction(0, 3.0)
    v2 = ck2.solve()
    assert abs(v2[1] - 6.0 / 7.0) < 1e-12
    assert abs(v2[2] - 3.0 / 7.0) < 1e-12
    print("test 1 (hand example): OK")


def test_random_changes():
    net = nn.generate_network(n_wires=100, side=30.0, seed=3)
    s, d = nn.point_electrodes(net)
    m = net.n_junctions()
    n = net.n_wires()
    K = 10
    level_g = np.zeros(K + 1)
    for k in range(K + 1):
        level_g[k] = 1.0e-3 * (1.0e3) ** (float(k) / K)
    level = np.zeros(m, dtype=int)
    ck = CholeskyKirchhoff(n, net.junctions, s, d, 0.2, level_g[level], maxupd=10)
    rng = np.random.default_rng(0)
    worst = 0.0
    for step in range(500):
        j = int(rng.integers(0, m))
        if level[j] < K:
            level[j] = level[j] + 1
            ck.update_junction(j, level_g[level[j]])
        v_c = ck.solve()
        v_ref = solve_node_voltages(n, net.junctions, level_g[level], s, d, 0.2)
        worst = max(worst, float(np.max(np.abs(v_c - v_ref))) / 0.2)
    assert worst < 1.0e-9
    print("test 2 (random changes): OK, max error %.1e of V, refactorisations: %d"
          % (worst, ck.n_refactor))


def test_loop_statistics():
    for K in (1, 10):
        n_a = 0
        n_b = 0
        for seed in range(10):
            net = nn.generate_network(seed=seed)
            s, d = nn.point_electrodes(net)
            for v in (0.08, 0.15, 0.3):
                a = run_loop_levels_fast(net.n_wires(), net.junctions, s, d, v,
                                         0.01, 0.01, 1.0e-3, 1.0, K)
                b = run_loop_levels_chol(net.n_wires(), net.junctions, s, d, v,
                                         0.01, 0.01, 1.0e-3, 1.0, K)
                n_a = n_a + a["event_junction"].shape[0]
                n_b = n_b + b["event_junction"].shape[0]
        rel = abs(n_a - n_b) / max(n_a, 1)
        print("test 3 (loop) K=%d: switches LU %d, Cholesky %d, difference %.3f%%"
              % (K, n_a, n_b, 100 * rel))
        assert rel < 0.01


if __name__ == "__main__":
    test_hand_example()
    test_random_changes()
    test_loop_statistics()
    print("ALL TESTS OK")
