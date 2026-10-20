# test_solver_avalanches_tau.py
# Checks for nw_solver.py, nw_avalanches.py and nw_tau.py.
# Run: python tests/test_solver_avalanches_tau.py
# (the tests of the multi-level loop are in test_loop_levels.py, those of the
# network generator in test_nanowire_network.py)

# The library files (nanowire_network.py, nw_solver.py, ...) are in the main
# folder, one level up: add it to the search path.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import nanowire_network as nn
from nw_solver import (solve_node_voltages, junction_drops,
                       total_current, total_current_into_drains)
from nw_loop_levels import run_loop_levels
from nw_avalanches import find_avalanches
from nw_tau import fit_tau

# Parameters of Hochstetter et al. (Methods). Units: G0 = 1/(12.9 kOhm), so
# Goff = 1e-3 and Gon = 1 (ratio 1000 = Roff/Rf in the paper).
V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0


def test_solver_chain():
    # Hand example: V_M = 1, drops 3 and 1, current 3.
    junctions = np.array([[0, 1], [1, 2]])
    g = np.array([1.0, 3.0])
    v = solve_node_voltages(3, junctions, g, [0], [2], 4.0)
    assert np.allclose(v, [4.0, 1.0, 0.0]), v
    drops = junction_drops(junctions, v)
    assert np.allclose(drops, [3.0, 1.0])
    assert abs(total_current(3, junctions, g, v, [0]) - 3.0) < 1e-12
    print("OK test 1: hand chain, V_M = 1, drops 3 and 1, current 3")


def test_solver_vs_dense():
    # Compare with a dense matrix solved by numpy, on a real random network.
    net = nn.generate_network(seed=11)
    s, d = nn.strip_electrodes(net, 0.15)
    n = net.n_wires()
    # Mixed conductances: every third junction ON, the others OFF.
    g = np.where(np.arange(net.n_junctions()) % 3 == 0, G_ON, G_OFF)
    v = solve_node_voltages(n, net.junctions, g, s, d, 0.2)
    A = np.zeros((n, n))
    for k in range(net.n_junctions()):
        i, j = net.junctions[k]
        A[i, i] += g[k]
        A[j, j] += g[k]
        A[i, j] -= g[k]
        A[j, i] -= g[k]
    fixed = list(s) + list(d)
    vf = np.zeros(n)
    for i in s:
        vf[i] = 0.2
    free = [i for i in range(n) if i not in fixed]
    x = np.linalg.solve(A[np.ix_(free, free)],
                        -A[np.ix_(free, fixed)] @ vf[fixed])
    v_dense = vf.copy()
    v_dense[free] = x
    assert np.allclose(v, v_dense, atol=1e-10)
    # All voltages between 0 and V_ext (maximum principle: a free node sits
    # at a weighted average of its neighbours, so it cannot exceed the electrodes)
    assert v.min() >= -1e-12 and v.max() <= 0.2 + 1e-12
    # Current out of sources = current into drains
    i_out = total_current(n, net.junctions, g, v, s)
    i_in = total_current_into_drains(n, net.junctions, g, v, d)
    assert abs(i_out - i_in) < 1e-10 * max(1.0, abs(i_out))
    print("OK test 2: solver = dense matrix; V in [0,Vext]; I_out = I_in =",
          round(i_out, 6))


def test_avalanches_hand_example():
    # Example in nw_avalanches.py: counts [2,2,0,2,0,0,1],
    # S = [4,2,1], T = [2,1,1].
    out = find_avalanches([0, 0.5, 1.0, 1.1, 3.0, 3.2, 5.0])
    assert out["counts"].tolist() == [2, 2, 0, 2, 0, 0, 1], out["counts"]
    assert out["S"].tolist() == [4, 2, 1], out["S"]
    assert out["T"].tolist() == [2, 1, 1], out["T"]
    assert abs(out["frame_width"] - 5.0 / 6) < 1e-12
    # edge cases: no events, one event, all events at the same clock
    assert find_avalanches([]).get("S").shape[0] == 0
    assert find_avalanches([2.0])["S"].tolist() == [1]
    assert find_avalanches([1.0, 1.0, 1.0])["S"].tolist() == [3]
    # on a real run (K = 1), the sizes add up to the number of switches
    net = nn.generate_network(seed=7)
    s, d = nn.point_electrodes(net)
    res = run_loop_levels(net.n_wires(), net.junctions, s, d, 0.15,
                          V_SET, LAM_CRIT, G_OFF, G_ON, 1)
    out = find_avalanches(res["event_time"])
    assert int(out["S"].sum()) == len(res["event_time"])
    assert int(out["counts"].sum()) == len(res["event_time"])
    print("OK test 3: avalanches, hand example + edge cases; seed 7 at 0.15 V:",
          len(out["S"]), "avalanches, largest S =", int(out["S"].max()))


def test_tau_synthetic():
    # Tau estimator on synthetic data with known exponent (numpy zipf draws
    # P(k) ~ k^-a exactly): the fit must come back close to a.
    rng = np.random.default_rng(1)
    x = rng.zipf(2.2, size=50000)
    tau = fit_tau(x, 1)
    assert abs(tau - 2.2) < 0.03, tau
    print("OK test 4: tau fit on synthetic data (true 2.2, fitted",
          round(tau, 3), ")")


if __name__ == "__main__":
    test_solver_chain()
    test_solver_vs_dense()
    test_avalanches_hand_example()
    test_tau_synthetic()
