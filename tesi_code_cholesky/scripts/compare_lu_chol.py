# compare_lu_chol.py
# SPEED (and result) comparison between:
#   - the current fast solver: LU + Woodbury formula, refactorises every 50
#     changed junctions (nw_loop_levels_fast.py)
#   - the solver with update of the Cholesky factor (nw_loop_levels_chol.py),
#     tried with several values of maxupd (how often it refactorises).
#
# For every network and every V* = factor x Vth the two loops are run at the
# SAME voltage, one after the other in the same process, and timed.
# It prints: number of switches (they must be equal or almost), total time of
# each method and the ratio (time LU / time Cholesky; > 1 means that Cholesky
# is faster).
#
# How to run (from the main folder, in the Miniforge Prompt):
#   python scripts/compare_lu_chol.py 100:30:5 400:60:3
# Every argument is  wires:side_um:number_of_networks.
# Optional environment variables (PowerShell: $env:K = 10  /  cmd: set K=10):
#   K (default 10), FACTORS (default 1.05,1.8), CHOL_MAXUPDS (default 50,200,1000)

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nanowire_network as nn
import nw_threshold as thr
from nw_loop_levels_fast import run_loop_levels_fast as run_lu
from nw_loop_levels_chol import run_loop_levels_chol as run_chol

V_SET = 0.01
LAM_CRIT = 0.01
G_ON = 1.0
G_OFF = G_ON / 1000.0
K = int(os.environ.get("K", "10"))
FACTORS = [float(x) for x in os.environ.get("FACTORS", "1.05,1.8").split(",")]
MAXUPDS = [int(x) for x in os.environ.get("CHOL_MAXUPDS", "50,200,1000").split(",")]

if len(sys.argv) > 1:
    SIZE_SPECS = sys.argv[1:]
else:
    SIZE_SPECS = ["100:30:5"]

for spec in SIZE_SPECS:
    parts = spec.split(":")
    n_wires = int(parts[0])
    side = float(parts[1])
    n_nets = int(parts[2])
    print("=" * 70)
    print("%d wires, side %.0f, %d networks, K = %d" % (n_wires, side, n_nets, K))
    # for every factor: accumulated times and switches
    t_lu = {}
    t_ch = {}
    ev_lu = {}
    ev_ch = {}
    for f in FACTORS:
        t_lu[f] = 0.0
        ev_lu[f] = 0
        for mu in MAXUPDS:
            t_ch[(f, mu)] = 0.0
            ev_ch[(f, mu)] = 0
    for seed in range(n_nets):
        net = nn.generate_network(n_wires=n_wires, side=side, seed=seed)
        s, d = nn.point_electrodes(net)
        vth = thr.find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
        if np.isnan(vth):
            print("  network %d: no path, skipped" % seed, flush=True)
            continue
        for f in FACTORS:
            v = f * vth
            t0 = time.perf_counter()
            a = run_lu(net.n_wires(), net.junctions, s, d, v,
                       V_SET, LAM_CRIT, G_OFF, G_ON, K)
            t_lu[f] = t_lu[f] + (time.perf_counter() - t0)
            ev_lu[f] = ev_lu[f] + a["event_junction"].shape[0]
            for mu in MAXUPDS:
                t0 = time.perf_counter()
                b = run_chol(net.n_wires(), net.junctions, s, d, v,
                             V_SET, LAM_CRIT, G_OFF, G_ON, K, maxupd=mu)
                t_ch[(f, mu)] = t_ch[(f, mu)] + (time.perf_counter() - t0)
                ev_ch[(f, mu)] = ev_ch[(f, mu)] + b["event_junction"].shape[0]
        print("  network %d done" % seed, flush=True)
    for f in FACTORS:
        print("-" * 70)
        print("V* = %.2f   LU+Woodbury: %d switches, %.2f s" % (f, ev_lu[f], t_lu[f]))
        for mu in MAXUPDS:
            if t_ch[(f, mu)] > 0.0:
                ratio = t_lu[f] / t_ch[(f, mu)]
            else:
                ratio = float("nan")
            print("   Cholesky maxupd=%5d: %d switches, %.2f s, LU/Cholesky = %.2f"
                  % (mu, ev_ch[(f, mu)], t_ch[(f, mu)], ratio))
