# run_average_size.py
# Average of the avalanche exponent tau over many random networks, for
# several system sizes at CONSTANT wire density.
#
# Why groups: one network has only ~30 avalanches, too few for a stable fit.
# So the networks are cut into groups of GROUP_SIZE (10). For every group we
# pool the avalanches of its 10 networks and fit tau once. Then we report the
# mean of tau over the groups and the spread (standard deviation) BETWEEN
# groups. That spread is the variation between random realisations.
# (nw_tau.bootstrap_tau is different: it resamples the same networks.)
#
# Example: 200 networks, GROUP_SIZE = 10 -> 20 groups -> 20 values of tau.
#   If they are 1.70, 1.82, ... the table shows their mean and their std.
#
# Constant density: 100 wires in 30 x 30 um -> 400 wires in 60 x 60 um
# (area x4, wires x4) -> 900 wires in 90 x 90 um (area x9, wires x9).
# The wire length stays 10 um, so only the system size changes.
#
# Each network is run at V = V* x Vth, with its own Vth (nw_threshold.py).
# Events are counted by growth episode (see nw_loop_levels.py).
#
# Run (from the main folder):
#   python scripts/run_average_size.py
#   python scripts/run_average_size.py 100:30:20 400:60:5
# Each argument is  n_wires:side_um:n_groups.  Default: 100:30:20 400:60:5 900:90:3
# Optional environment variables:
#   K (default 10), RATIO = Gon/Goff (default 1000),
#   PROCS = number of processes (default 2).
#   PowerShell:  $env:K = 3
# Time per network (one sample each): about 2 s (100 wires), 35 s (400),
# 75 s (900). Results are cached in cache_avg_*.pkl next to this script:
# delete the file to repeat a run.

import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np

# The library files are in the main folder, one level up.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nanowire_network as nn
from nw_loop_levels import run_loop_levels
from nw_avalanches import find_avalanches
from nw_tau import fit_tau
from nw_threshold import find_threshold

V_SET = 0.01
LAM_CRIT = 0.01
G_ON = 1.0
# Read at the top of the file so that worker processes (Windows starts them
# from scratch) read the same values.
RATIO = float(os.environ.get("RATIO", "1000"))
G_OFF = G_ON / RATIO
K = int(os.environ.get("K", "10"))
N_PROCS = int(os.environ.get("PROCS", "2"))
FACTORS = [1.05, 1.8]          # V* = V / Vth
GROUP_SIZE = 10                # networks per group
MIN_AVALANCHES = 30            # a group with fewer avalanches is skipped (my choice)
SMIN_LIST = [1, 2, 3]
FOLDER = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    SIZE_SPECS = sys.argv[1:]
else:
    SIZE_SPECS = ["100:30:20", "400:60:5", "900:90:3"]


def one_network(args):
    # Worker: one network. Returns (seed, Vth, sizes) where
    # sizes[V*] = array of avalanche sizes (episode events).
    seed, n_wires, side = args
    net = nn.generate_network(n_wires=n_wires, side=side, seed=seed)
    s, d = nn.point_electrodes(net)
    vth = find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
    out = {}
    if np.isnan(vth):
        return seed, vth, out
    for f in FACTORS:
        res = run_loop_levels(net.n_wires(), net.junctions, s, d, f * vth,
                              V_SET, LAM_CRIT, G_OFF, G_ON, K)
        times = res["event_time"][res["event_new_episode"]]
        av = find_avalanches(times)
        out[f] = np.asarray(av["S"], dtype=int)
    return seed, vth, out


def get_results(n_wires, side, n_networks):
    # Results for all networks of this size (read from the cache if present).
    name = "cache_avg_r%g_K%d_w%d_L%g_n%d.pkl" % (RATIO, K, n_wires, side, n_networks)
    cache = os.path.join(FOLDER, name)
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    # Reuse a smaller cache of the same run (same size, K, ratio): the seeds
    # 0 .. n_old-1 are the same networks, so only the new seeds are computed.
    # Example: cache with 30 networks, 100 requested -> only seeds 30..99.
    prefix = "cache_avg_r%g_K%d_w%d_L%g_n" % (RATIO, K, n_wires, side)
    best_old = 0
    best_file = None
    for fname in os.listdir(FOLDER):
        if fname.startswith(prefix) and fname.endswith(".pkl"):
            n_old = int(fname[len(prefix):-4])
            if n_old < n_networks and n_old > best_old:
                best_old = n_old
                best_file = fname
    old_results = []
    if best_file is not None:
        with open(os.path.join(FOLDER, best_file), "rb") as fh:
            old_results = pickle.load(fh)
        print("   reusing %d networks from %s" % (best_old, best_file), flush=True)

    jobs = [(seed, n_wires, side) for seed in range(best_old, n_networks)]
    with Pool(N_PROCS) as pool:
        new_results = pool.map(one_network, jobs, chunksize=1)
    results = list(old_results) + list(new_results)
    with open(cache, "wb") as fh:
        pickle.dump(results, fh)
    return results


def group_taus(results, factor, smin):
    # One tau per group of GROUP_SIZE consecutive networks (seeds).
    # Returns (list of tau, list of number of avalanches per group).
    taus = []
    counts = []
    n_groups = len(results) // GROUP_SIZE
    for g in range(n_groups):
        chunk = results[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
        arrays = []
        for r in chunk:
            if factor in r[2]:
                arrays.append(r[2][factor])
        if len(arrays) == 0:
            continue
        pooled = np.concatenate(arrays)
        counts.append(pooled.shape[0])
        if pooled.shape[0] < MIN_AVALANCHES:
            continue
        t = fit_tau(pooled, smin, int(pooled.max()))
        if not np.isnan(t):
            taus.append(t)
    return taus, counts


if __name__ == "__main__":
    print("K = %d, ratio Gon/Goff = %g, groups of %d networks, event = episode"
          % (K, RATIO, GROUP_SIZE))
    for spec in SIZE_SPECS:
        parts = spec.split(":")
        n_wires = int(parts[0])
        side = float(parts[1])
        n_networks = int(parts[2]) * GROUP_SIZE
        results = get_results(n_wires, side, n_networks)
        good = [r for r in results if not np.isnan(r[1])]
        vth = np.array([r[1] for r in good]) / V_SET
        print("\n=== %d wires in %g x %g um: %d networks (%d groups), "
              "Vth/Vset mean %.2f (min %.1f, max %.1f)"
              % (n_wires, side, side, len(good), len(results) // GROUP_SIZE,
                 vth.mean(), vth.min(), vth.max()), flush=True)
        print("   V*    aval/group  |  tau smin=1        smin=2        smin=3   "
              "(mean +- std between groups, [groups used])")
        for f in FACTORS:
            line = None
            for smin in SMIN_LIST:
                taus, counts = group_taus(results, f, smin)
                if line is None:
                    line = "   %.2f  %9.0f  |" % (f, np.mean(counts))
                if len(taus) >= 2:
                    line = line + "  %.2f+-%.2f [%d]" % (
                        np.mean(taus), np.std(taus, ddof=1), len(taus))
                else:
                    line = line + "  too few groups"
            print(line, flush=True)
