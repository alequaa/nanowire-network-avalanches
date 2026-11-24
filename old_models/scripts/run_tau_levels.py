# run_tau_levels.py
# Avalanche exponent tau with the multi-level loop (nw_loop_levels.py).
# Each network is run at a voltage proportional to its own threshold:
# V = V* x Vth, with V* = 0.7, 1.0, 1.05, 1.8. K = 1 is the two-state
# junction. Network: default of generate_network (100 wires, 30 x 30 um),
# point electrodes.
#
# Ways to count an "event" (one junction):
#   all    : every level step of every junction
#   first  : only the step 0 -> 1 (the junction starts to conduct)
#   last   : only the last step K-1 -> K (the junction becomes fully ON)
#   episode: one event per growth episode of a junction (like Hochstetter):
#            starts at the first level step, ends when the drop reaches Vset
#
# Run: python scripts/run_tau_levels.py [ratio Gon/Goff] [K ...]
#   python scripts/run_tau_levels.py             ratio 1000, K = 1, 3, 10
#   python scripts/run_tau_levels.py 4 10        ratio 4,    only K = 10
# Optional: environment variable NNET = number of networks (default 200).
#   PowerShell:  $env:NNET = 10
# About 20-30 minutes for the default run (2 processes). The results are
# saved in cache_*.pkl files: delete the file to repeat a run.

import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np

# The library files (nanowire_network.py, nw_solver.py, ...) are in the main
# folder, one level up: add it to the search path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nanowire_network as nn
from nw_loop_levels import run_loop_levels
from nw_avalanches import find_avalanches
from nw_tau import fit_tau, bootstrap_tau
from nw_threshold import find_threshold

V_SET = 0.01
LAM_CRIT = 0.01
G_ON = 1.0
# Ratio Gon/Goff from the first command-line argument (default 1000).
# This is read at the top of the file so that the worker processes
# (Windows starts them from scratch) read it too.
if len(sys.argv) > 1:
    RATIO = float(sys.argv[1])
else:
    RATIO = 1000.0
G_OFF = G_ON / RATIO
FACTORS = [0.7, 1.0, 1.05, 1.8]           # V* values
if len(sys.argv) > 2:
    K_LIST = [int(x) for x in sys.argv[2:]]
else:
    K_LIST = [1, 3, 10]
N_NETWORKS = int(os.environ.get("NNET", "200"))
EVENT_TYPES = ["all", "first", "last", "episode"]
FOLDER = os.path.dirname(os.path.abspath(__file__))


def one_network(args):
    # Worker: one network (seed) and one K. Returns (seed, Vth, sizes) where
    # sizes[(V*, event type)] = array of avalanche sizes.
    seed, K = args
    net = nn.generate_network(seed=seed)
    s, d = nn.point_electrodes(net)
    vth = find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
    out = {}
    if np.isnan(vth):
        return seed, vth, out
    for f in FACTORS:
        res = run_loop_levels(net.n_wires(), net.junctions, s, d, f * vth,
                              V_SET, LAM_CRIT, G_OFF, G_ON, K)
        et = res["event_time"]
        lv = res["event_level"]
        times = {}
        times["all"] = et
        times["first"] = et[lv == 1]
        times["last"] = et[lv == K]
        times["episode"] = et[res["event_new_episode"]]
        for name in EVENT_TYPES:
            av = find_avalanches(times[name])
            out[(f, name)] = np.asarray(av["S"], dtype=int)
    return seed, vth, out


def get_results(K):
    # Results for all networks at this K (read from the cache file if present).
    cache = os.path.join(FOLDER, "cache_Dlev3_r%g_K%d_%d.pkl" % (RATIO, K, N_NETWORKS))
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    with Pool(2) as pool:
        results = pool.map(one_network, [(seed, K) for seed in range(N_NETWORKS)],
                           chunksize=2)
    with open(cache, "wb") as fh:
        pickle.dump(results, fh)
    return results


if __name__ == "__main__":
    for K in K_LIST:
        results = get_results(K)
        good = [r for r in results if not np.isnan(r[1])]
        vth = np.array([r[1] for r in good]) / V_SET
        print("\n=== ratio %g, K = %d: %d networks, Vth/Vset mean %.2f (min %.1f, max %.1f)"
              % (RATIO, K, len(good), vth.mean(), vth.min(), vth.max()), flush=True)
        for name in EVENT_TYPES:
            if K == 1 and name != "all":
                continue
            print(" events counted: %s" % name)
            print("   V*     avalanches  mean S  max S  frac(S=1) |  tau smin=1, 2, 3 (+- boot)")
            for f in FACTORS:
                per_net = [r[2][(f, name)] for r in good if r[2][(f, name)].shape[0] > 0]
                if len(per_net) == 0:
                    print("   %.2f   none" % f)
                    continue
                pooled = np.concatenate(per_net)
                line = "   %.2f   %9d  %6.2f  %5d  %8.3f |" % (
                    f, pooled.shape[0], pooled.mean(), pooled.max(), np.mean(pooled == 1))
                if pooled.shape[0] >= 100:
                    smax = int(pooled.max())
                    for smin in [1, 2, 3]:
                        t = fit_tau(pooled, smin, smax)
                        m, sd = bootstrap_tau(per_net, smin, smax, 100)
                        line = line + "  %.2f+-%.2f" % (t, sd)
                else:
                    line = line + "  too few"
                print(line, flush=True)
