# test_loop_model_two_state.py
# Checks for the two-state loop (nw_loop_model.py) and the voltage ramp
# (nw_ramp.py). Run: python test_loop_model_two_state.py
# (the tests of the solver, of the avalanches and of the tau fit are in
# test_solver_avalanches_tau.py in the main folder)

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import nanowire_network as nn
from nw_loop_model import run_loop
from nw_ramp import ramp_avalanches

# Parameters of Hochstetter et al. (Methods). Units: G0 = 1/(12.9 kOhm),
# so Goff = 1e-3 and Gon = 1 (ratio 1000 = Roff/Rf in the paper).
V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0


def test_loop_hand_example():
    # j1 switches at 0.5, j0 at 0.6669 (example in nw_loop_model.py).
    junctions = np.array([[0, 1], [1, 2]])
    res = run_loop(3, junctions, [0], [2], 0.04, V_SET, LAM_CRIT,
                   G_OFF, G_ON, lam_start=np.array([0.0, 0.005]))
    assert list(res["event_junction"]) == [1, 0], res["event_junction"]
    drop0 = 0.04 * G_ON / (G_ON + G_OFF)
    t0 = 0.5 + (LAM_CRIT - 0.005) / (drop0 - V_SET)
    assert abs(res["event_time"][0] - 0.5) < 1e-12
    assert abs(res["event_time"][1] - t0) < 1e-12, res["event_time"]
    assert "all junctions" in res["stop_reason"]
    print("OK test 1: hand example, times", np.round(res["event_time"], 4),
          "(expected 0.5 and", round(t0, 4), ")")


def test_loop_stops_immediately():
    # Drops = 0.0075 < Vset = 0.01 -> all times infinite, no switch.
    junctions = np.array([[0, 1], [1, 2]])
    res = run_loop(3, junctions, [0], [2], 0.015, V_SET, LAM_CRIT, G_OFF, G_ON)
    assert res["event_junction"].shape[0] == 0
    assert "steady state" in res["stop_reason"]
    print("OK test 2: V_ext = 0.015 -> no switch, stop")


def test_loop_properties():
    # General properties on real networks.
    for seed in range(5):
        net = nn.generate_network(seed=seed)
        s, d = nn.point_electrodes(net)
        res = run_loop(net.n_wires(), net.junctions, s, d, 0.3,
                       V_SET, LAM_CRIT, G_OFF, G_ON)
        et = res["event_time"]
        ej = res["event_junction"]
        # times never decrease
        assert np.all(np.diff(et) >= 0.0)
        # no junction switches twice
        assert len(set(ej.tolist())) == len(ej)
        # number of ON junctions = number of events
        assert int(res["is_on"].sum()) == len(ej)
        # total current never decreases (Rayleigh monotonicity)
        ch = res["current_history"]
        assert np.all(np.diff(ch) >= -1e-12), "current decreased"
        # Lambda never above Lcrit (up to rounding)
        assert res["lam"].max() <= LAM_CRIT * (1 + 1e-9)
    print("OK test 3: 5 real networks, times non-decreasing, no double switch,"
          " current non-decreasing, Lambda <= Lcrit")


def test_ramp():
    # Ramp, hand example: 0.015 V gives nothing, 0.04 V gives one avalanche S = 2.
    junctions = np.array([[0, 1], [1, 2]])
    out = ramp_avalanches(3, junctions, [0], [2], [0.015, 0.04],
                          V_SET, LAM_CRIT, G_OFF, G_ON)
    assert out["S"].tolist() == [2], out["S"]
    # A ramp with a single step must give the same number of switches as a plain run.
    net = nn.generate_network(seed=7)
    s, d = nn.point_electrodes(net)
    one = ramp_avalanches(net.n_wires(), net.junctions, s, d, [0.15],
                          V_SET, LAM_CRIT, G_OFF, G_ON)
    plain = run_loop(net.n_wires(), net.junctions, s, d, 0.15,
                     V_SET, LAM_CRIT, G_OFF, G_ON)
    assert one["S"].tolist() == [len(plain["event_junction"])]
    print("OK test 4: ramp hand example, one-step ramp = plain run")


def sweep():
    # Not pass/fail: numbers to look at.
    print("\nSweep in V_ext, 100-wire network, point electrodes, 10 networks")
    print("  V_ext    mean switches   (min-max)    mean final clock")
    for v_ext in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
        n_ev = []
        clk = []
        for seed in range(10):
            net = nn.generate_network(seed=seed)
            s, d = nn.point_electrodes(net)
            res = run_loop(net.n_wires(), net.junctions, s, d, v_ext,
                           V_SET, LAM_CRIT, G_OFF, G_ON)
            n_ev.append(len(res["event_junction"]))
            clk.append(res["clock"])
        print("  %5.2f    %8.1f      (%d-%d)       %.3f" %
              (v_ext, np.mean(n_ev), min(n_ev), max(n_ev), np.mean(clk)))


if __name__ == "__main__":
    test_loop_hand_example()
    test_loop_stops_immediately()
    test_loop_properties()
    test_ramp()
    sweep()
