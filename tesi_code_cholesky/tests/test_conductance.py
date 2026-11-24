# test_conductance.py
# Checks that the loops save the conductance of the network correctly.
# Run (from the main folder):  python tests/test_conductance.py
# 1. Hand example: chain S - M1 - M2 - D, all g = 1, V_S = 1 -> I = 1/3.
# 2. On a real network, the current saved inside the loop is equal to the one
#    obtained by "replaying" the switches and solving Kirchhoff after each of
#    them (the method used before in plot_model_overview.py, with
#    nw_solver.total_current). Done for the three loops (original, fast LU,
#    Cholesky).
# 3. Physical property: if the conductances only increase, the conductance of
#    the network cannot decrease (G non-decreasing).
# 4. Lengths: one value of G for each state (n_switches + 1) and
#    state_time[0] = 0.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import nanowire_network as nn
from nw_current import make_current_meter
from nw_solver import solve_node_voltages, total_current
from nw_loop_levels import run_loop_levels
from nw_loop_levels_fast import run_loop_levels_fast
from nw_loop_levels_chol import run_loop_levels_chol

V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0


def test_hand_example():
    junctions = np.array([[0, 1], [1, 2], [2, 3]])
    voltage = np.array([1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0])
    g = np.array([1.0, 1.0, 1.0])
    meter = make_current_meter(4, junctions, [0])
    assert abs(meter(voltage, g) - 1.0 / 3.0) < 1e-12
    print("test 1 (hand example): OK, I = 1/3")


def replay_currents(net, s, d, v, res, K, n_check):
    # Current after each switch, replaying the switches (original solver).
    n = net.n_wires()
    level_g = np.zeros(K + 1)
    for k in range(K + 1):
        level_g[k] = G_OFF * (G_ON / G_OFF) ** (float(k) / K)
    level = np.zeros(net.n_junctions(), dtype=int)
    out = []
    volt = solve_node_voltages(n, net.junctions, level_g[level], s, d, v)
    out.append(total_current(n, net.junctions, level_g[level], volt, s))
    for e in range(n_check):
        level[res["event_junction"][e]] = res["event_level"][e]
        volt = solve_node_voltages(n, net.junctions, level_g[level], s, d, v)
        out.append(total_current(n, net.junctions, level_g[level], volt, s))
    return np.array(out)


def test_loops():
    net = nn.generate_network(n_wires=100, side=30.0, seed=2)
    s, d = nn.point_electrodes(net)
    n = net.n_wires()
    loops = [("original", run_loop_levels), ("fast LU", run_loop_levels_fast),
             ("Cholesky", run_loop_levels_chol)]
    for K in (1, 10):
        for name, fn in loops:
            v = 0.12
            res = fn(n, net.junctions, s, d, v, V_SET, LAM_CRIT, G_OFF, G_ON, K)
            n_ev = res["event_junction"].shape[0]
            # (4) lengths
            assert res["state_current"].shape[0] == n_ev + 1
            assert res["state_conductance"].shape[0] == n_ev + 1
            assert res["state_time"].shape[0] == n_ev + 1
            assert res["state_time"][0] == 0.0
            assert abs(res["state_time"][-1] - res["clock"]) < 1e-12
            # (3) G non-decreasing
            dg = np.diff(res["state_conductance"])
            assert np.min(dg) > -1e-10 * np.max(res["state_conductance"])
            # (2) comparison with the replay (first 300 switches)
            n_check = min(n_ev, 300)
            ref = replay_currents(net, s, d, v, res, K, n_check)
            rel = np.max(np.abs(res["state_current"][:n_check + 1] - ref) / ref)
            assert rel < 1e-8
            print("test 2-4: K=%2d, loop %-10s: %5d switches, G from %.3e to %.3e, "
                  "max relative error %.1e: OK"
                  % (K, name, n_ev, res["state_conductance"][0],
                     res["state_conductance"][-1], rel))


if __name__ == "__main__":
    test_hand_example()
    test_loops()
    print("ALL TESTS OK")
