# run_tau_no_spanning.py
# Tau of the avalanche distribution WITH and WITHOUT the "spanning" avalanche.
#
# Spanning avalanche = the avalanche in which the first path of ON junctions
# between source and drain is formed (the moment the network becomes conductive).
# check_first_avalanche.py showed that this is the LARGEST avalanche of the run
# in 91-100% of the networks (K = 1, 100-900 wires, V* = 1.05 and 1.8).
# It plays the role of the percolating cluster, which is left out of the
# cluster-size statistics because it scales with the whole system.
#
# Rule: in every network we remove exactly that avalanche (one per network).
# Example: sizes S = [3, 1, 5, 2], the path forms at episode number 7:
#   cumulative sizes = [3, 4, 9, 11]; the first one >= 7 is 9 -> the third
#   avalanche (S = 5) is the spanning one. Without it: S = [3, 1, 2].
# A network with no ON path at the end of the run loses nothing (it is counted
# and printed).
#
# CAUTION: removing one avalanche per network lowers the number of avalanches
# and cuts the tail. With only 4-8 avalanches per network, removing the largest
# one raises tau by itself, even for a perfect power law (synthetic test: tau
# 1.8 -> 2.2 with 4-8 avalanches per network). So this table is a way to see how
# much tau depends on the spanning avalanche, not "the true tau". Always read
# the two columns together.
#
# Run (from the main folder):
#   python scripts/run_tau_no_spanning.py 100:30:20 400:60:20 900:90:10
# Each argument is n_wires:side_um:n_groups; networks = n_groups x GROUP.
# Environment variables: K (default 10), RATIO (default 1000), GROUP (networks
# per group, default 10), PROCS (default 2), SMIN (default 2).
#   cmd:  set K=1   then   set GROUP=50
# Results are cached in cache_nospan_*.pkl next to this script (a cache with
# fewer networks of the same size is reused; only the new seeds are computed).

import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nanowire_network as nn
from nw_loop_levels import run_loop_levels
from nw_avalanches import find_avalanches
from nw_threshold import find_threshold
from nw_tau import fit_tau
# first_path_event: event after which a source-drain path of ON junctions exists
from check_first_avalanche import first_path_event

V_SET = 0.01
LAM_CRIT = 0.01
G_ON = 1.0
# Read at the top of the file so that worker processes (Windows starts them
# from scratch) read the same values.
RATIO = float(os.environ.get("RATIO", "1000"))
G_OFF = G_ON / RATIO
K = int(os.environ.get("K", "10"))
N_PROCS = int(os.environ.get("PROCS", "2"))
GROUP_SIZE = int(os.environ.get("GROUP", "10"))
SMIN = int(os.environ.get("SMIN", "2"))
FACTORS = [0.5, 0.7, 0.9, 1.0, 1.05, 1.2, 1.4, 1.8, 2.5]
MIN_AVALANCHES = 30
SMAX_FIXED = 100
FOLDER = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    SIZE_SPECS = sys.argv[1:]
else:
    SIZE_SPECS = ["100:30:20", "400:60:20", "900:90:10"]


def one_network(args):
    # Worker: one network. Returns (seed, Vth, out) with
    # out[V*] = (S_all, S_without_spanning, has_path, S_spanning)
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
        is_new = res["event_new_episode"]
        times = res["event_time"][is_new]
        sizes = np.asarray(find_avalanches(times)["S"], dtype=int)
        if sizes.shape[0] == 0:
            continue
        e_path = first_path_event(net, s, d, res)
        if e_path is None:
            # no ON path at the end of the run: nothing to remove
            out[f] = (sizes, sizes, False, 0)
        else:
            rank = int(np.sum(is_new[:e_path]))
            cum = np.cumsum(sizes)
            index_av = int(np.searchsorted(cum, rank, side="left"))
            index_av = min(index_av, sizes.shape[0] - 1)
            without = np.delete(sizes, index_av)
            out[f] = (sizes, without, True, int(sizes[index_av]))
    return seed, vth, out


def get_results(n_wires, side, n_networks):
    # Same cache logic as run_average_size.py.
    name = "cache_nospan_r%g_K%d_w%d_L%g_n%d.pkl" % (RATIO, K, n_wires, side, n_networks)
    cache = os.path.join(FOLDER, name)
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    prefix = "cache_nospan_r%g_K%d_w%d_L%g_n" % (RATIO, K, n_wires, side)
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


def group_taus(results, factor, which, smax_mode):
    # One tau per group of GROUP_SIZE networks. which = 0 (all avalanches) or
    # 1 (without the spanning one). smax_mode = "observed" or a number.
    taus = []
    n_groups = len(results) // GROUP_SIZE
    for g in range(n_groups):
        chunk = results[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
        arrays = []
        for r in chunk:
            if factor in r[2]:
                arrays.append(r[2][factor][which])
        if len(arrays) == 0:
            continue
        pooled = np.concatenate(arrays)
        if pooled.shape[0] < MIN_AVALANCHES:
            continue
        if smax_mode == "observed":
            smax = int(pooled.max())
        else:
            smax = int(smax_mode)
        t = fit_tau(pooled, SMIN, smax)
        if not np.isnan(t):
            taus.append(t)
    return taus


def cell(taus):
    if len(taus) >= 2:
        return "%.2f+-%.2f [%d]" % (np.mean(taus), np.std(taus, ddof=1), len(taus))
    return "too few groups"


if __name__ == "__main__":
    print("K = %d, ratio %g, groups of %d networks, smin = %d, smax fixed = %d"
          % (K, RATIO, GROUP_SIZE, SMIN, SMAX_FIXED))
    for spec in SIZE_SPECS:
        parts = spec.split(":")
        n_wires = int(parts[0])
        side = float(parts[1])
        n_networks = int(parts[2]) * GROUP_SIZE
        results = get_results(n_wires, side, n_networks)
        print("\n=== %d wires in %g x %g um: %d networks" % (n_wires, side, side, len(results)))
        print("   V*   | smax=%d: all           without spanning | smax=observed: all   "
              "without spanning | spanning: median S, share of events, no path" % SMAX_FIXED)
        for f in FACTORS:
            span_sizes = []
            share = []
            no_path = 0
            used = 0
            for r in results:
                if f in r[2]:
                    used = used + 1
                    if r[2][f][2]:
                        span_sizes.append(r[2][f][3])
                        share.append(r[2][f][3] / float(r[2][f][0].sum()))
                    else:
                        no_path = no_path + 1
            line = "   %.2f |" % f
            line = line + "  %-16s %-16s |" % (
                cell(group_taus(results, f, 0, SMAX_FIXED)),
                cell(group_taus(results, f, 1, SMAX_FIXED)))
            line = line + "  %-16s %-16s |" % (
                cell(group_taus(results, f, 0, "observed")),
                cell(group_taus(results, f, 1, "observed")))
            line = line + "  %.0f, %.0f%%, %d of %d" % (
                np.median(span_sizes), 100.0 * np.median(share), no_path, used)
            print(line, flush=True)
