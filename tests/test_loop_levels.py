# test_loop_levels.py
# Checks for the multi-level loop (nw_loop_levels.py). Run: python tests/test_loop_levels.py
#
#   1. hand chain, K = 1 
#   2. hand chain, K = 2 (times, order, levels)
#   3. no switch below Vset
#   4. K = 1 gives the same switches as the two-state loop
#      (legacy_two_state/nw_loop_model.py) on real networks
#   5. general properties for K = 3 and K = 10 on real networks
#   6. number of junctions that started <= number of episodes <= number of steps
#   7. reference numbers for seed 7 (to compare two computers)

import os
import sys

import numpy as np

# The library files (nanowire_network.py, nw_solver.py, ...) are in the main
# folder, one level up: add it to the search path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nanowire_network as nn
from nw_loop_levels import run_loop_levels

# The two-state loop is kept in the legacy folder; it is the reference for K = 1.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "legacy_two_state"))
from nw_loop_model import run_loop

V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0

CHAIN = np.array([[0, 1], [1, 2]])      # source 0 - wire 1 - drain 2


def run_chain(v, K):
    return run_loop_levels(3, CHAIN, [0], [2], v, V_SET, LAM_CRIT,
                           G_OFF, G_ON, K)


def test_chain_k1():
    # Two identical junctions, V = 0.04: drops 0.02 each, rate 0.01, time
    # (Lcrit - 0) / 0.01 = 1 s for both. Tie -> junction 0 first, then junction
    # 1 at the same clock (its bucket is already full).
    res = run_chain(0.04, 1)
    assert res["event_junction"].tolist() == [0, 1], res["event_junction"]
    assert np.allclose(res["event_time"], [1.0, 1.0])
    assert res["is_on"].all()
    print("OK test 1: chain K = 1, times [1, 1], order [0, 1]")


def test_chain_k2():
    # K = 2: a level step needs Lcrit / 2 = 0.005 of Lambda.
    # V = 0.025: drop 0.0125, rate 0.0025 -> 0.005 / 0.0025 = 2 s per level.
    res = run_chain(0.025, 2)
    assert np.allclose(res["event_time"], [2.0, 2.0, 4.0, 4.0]), res["event_time"]
    assert res["event_junction"].tolist() == [0, 1, 0, 1]
    assert res["event_level"].tolist() == [1, 1, 2, 2]
    # V = 0.04: rate 0.01 -> 0.5 s per level.
    res = run_chain(0.04, 2)
    assert np.allclose(res["event_time"], [0.5, 0.5, 1.0, 1.0]), res["event_time"]
    assert res["level"].tolist() == [2, 2]
    # Growth-episode flags. Both junctions grow without a real pause (the
    # two simultaneous junctions are processed one after the other, but no
    # time passes between them), so each junction makes ONE episode:
    # [True, True, False, False].
    flags = res["event_new_episode"].tolist()
    assert flags == [True, True, False, False], flags
    res2 = run_chain(0.025, 2)
    assert res2["event_new_episode"].tolist() == [True, True, False, False]
    print("OK test 2: chain K = 2, times, order, levels and episode flags as by hand")


def test_no_switch():
    # V = 0.015: drops 0.0075 < Vset, nothing happens.
    res = run_chain(0.015, 2)
    assert res["event_junction"].shape[0] == 0
    assert res["level"].tolist() == [0, 0]
    print("OK test 3: V_ext = 0.015 -> no switch")


def test_k1_equals_two_state():
    # K = 1 must give the same switches as the two-state loop.
    count = 0
    for seed in range(10):
        net = nn.generate_network(seed=seed)
        s, d = nn.point_electrodes(net)
        for v in (0.05, 0.15, 0.3):
            a = run_loop(net.n_wires(), net.junctions, s, d, v,
                         V_SET, LAM_CRIT, G_OFF, G_ON)
            b = run_loop_levels(net.n_wires(), net.junctions, s, d, v,
                                V_SET, LAM_CRIT, G_OFF, G_ON, 1)
            assert a["event_junction"].tolist() == b["event_junction"].tolist(), \
                "seed %d, V %g: different order" % (seed, v)
            assert np.allclose(a["event_time"], b["event_time"], rtol=1e-9, atol=0.0)
            assert (a["is_on"] == b["is_on"]).all()
            assert abs(a["clock"] - b["clock"]) <= 1e-9 * max(1.0, a["clock"])
            count = count + len(a["event_junction"])
    print("OK test 4: K = 1 equals the two-state loop (10 networks x 3 voltages,",
          count, "switches compared)")


def test_properties():
    for K in (3, 10):
        step = LAM_CRIT / K
        for seed in range(5):
            net = nn.generate_network(seed=seed)
            s, d = nn.point_electrodes(net)
            res = run_loop_levels(net.n_wires(), net.junctions, s, d, 0.2,
                                  V_SET, LAM_CRIT, G_OFF, G_ON, K)
            et = res["event_time"]
            ej = res["event_junction"]
            el = res["event_level"]
            level = res["level"]
            lam = res["lam"]
            # times never decrease
            assert np.all(np.diff(et) >= 0.0)
            # levels between 0 and K; is_on means top level
            assert level.min() >= 0 and level.max() <= K
            assert (res["is_on"] == (level == K)).all()
            # every junction goes up ONE level at a time: 1, 2, 3, ...
            for j in np.nonzero(level > 0)[0]:
                assert el[ej == j].tolist() == list(range(1, int(level[j]) + 1))
            # one event per level step
            assert len(ej) == int(level.sum())
            # Lambda: top junctions sit exactly at Lcrit; the others are
            # between their level and the next one
            top = (level == K)
            assert np.allclose(lam[top], LAM_CRIT)
            rest = ~top
            assert np.all(lam[rest] >= level[rest] * step - 1e-15)
            assert np.all(lam[rest] <= (level[rest] + 1) * step * (1 + 1e-9) + 1e-15)
    print("OK test 5: K = 3 and 10, 5 networks each: times non-decreasing, one"
          " level at a time, Lambda between its limits")


def test_started_episodes_steps():
    for K in (3, 10):
        for seed in range(5):
            net = nn.generate_network(seed=seed)
            s, d = nn.point_electrodes(net)
            res = run_loop_levels(net.n_wires(), net.junctions, s, d, 0.2,
                                  V_SET, LAM_CRIT, G_OFF, G_ON, K)
            started = int(np.sum(res["level"] > 0))
            episodes = int(np.sum(res["event_new_episode"]))
            steps = len(res["event_junction"])
            assert started <= episodes <= steps, (started, episodes, steps)
            # the first step of every junction always opens an episode
            first_steps = res["event_level"] == 1
            assert res["event_new_episode"][first_steps].all()
    print("OK test 6: junctions started <= episodes <= steps (K = 3, 10)")


def test_reference_numbers():
    # Seed 7, point electrodes, V = 0.15 V. Numbers computed by the author;
    # if they differ on another computer, something depends on the machine.
    net = nn.generate_network(seed=7)
    s, d = nn.point_electrodes(net)
    reference = {1: (102, 12.8774), 3: (338, 34.7836), 10: (1015, 68.8279)}
    line = ""
    for K in (1, 3, 10):
        res = run_loop_levels(net.n_wires(), net.junctions, s, d, 0.15,
                              V_SET, LAM_CRIT, G_OFF, G_ON, K)
        n_steps, clock = reference[K]
        assert len(res["event_time"]) == n_steps, (K, len(res["event_time"]))
        assert abs(res["clock"] - clock) < 5.0e-5, (K, res["clock"])
        line = line + " K=%d: %d steps, clock %.4f;" % (K, n_steps, res["clock"])
    print("OK test 7: reference numbers, seed 7 at 0.15 V:" + line)


if __name__ == "__main__":
    test_chain_k1()
    test_chain_k2()
    test_no_switch()
    test_k1_equals_two_state()
    test_properties()
    test_started_episodes_steps()
    test_reference_numbers()
