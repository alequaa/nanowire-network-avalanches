# check_smax.py
# Does tau depend on the upper cut smax used in the fit?
# run_average_size.py fits every group with smax = largest avalanche of that
# group. That maximum grows with the system size, so part of the growth of
# tau with size could come from the fit and not from the physics.
# Here the SAME cache files (no new simulation) are refitted with:
#   observed : smax = largest avalanche of the group (as in run_average_size)
#   a fixed smax (the same number for all sizes): avalanches larger than
#   smax are dropped from the fit. If smax is larger than the largest
#   avalanche, nothing is dropped, but the normalisation still sums up to smax.
# Example: group with sizes [1, 1, 2, 5, 400], fixed smax = 100 -> the 400 is
# dropped and the fit uses [1, 1, 2, 5].
#
# Run (from the main folder, after run_average_size.py has produced the caches):
#   python scripts/check_smax.py
# Optional environment variables: K (default 10), RATIO (default 1000),
# SMIN (default 2). PowerShell:  $env:SMIN = 3
# Fast: only refits, no simulation.

import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nw_tau import fit_tau

RATIO = float(os.environ.get("RATIO", "1000"))
K = int(os.environ.get("K", "10"))
SMIN = int(os.environ.get("SMIN", "2"))
FOLDER = os.path.dirname(os.path.abspath(__file__))
SIZES = [(100, 30.0), (400, 60.0), (900, 90.0)]    # (wires, side in um)
FACTORS = [1.05, 1.8]
GROUP_SIZE = 10
MIN_AVALANCHES = 30
SMAX_FIXED = [100, 300, 1000, 3000]


def find_cache(n_wires, side):
    # Largest cache file for this size (most networks).
    prefix = "cache_avg_r%g_K%d_w%d_L%g_n" % (RATIO, K, n_wires, side)
    best_n = 0
    best_file = None
    for fname in os.listdir(FOLDER):
        if fname.startswith(prefix) and fname.endswith(".pkl"):
            n = int(fname[len(prefix):-4])
            if n > best_n:
                best_n = n
                best_file = fname
    return best_file


def group_fits(results, factor, smax_mode):
    # One tau per group of GROUP_SIZE networks. smax_mode = "observed" or a
    # number. Returns the list of tau.
    taus = []
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


def mean_group_max(results, factor):
    # Mean over groups of the largest avalanche in the group.
    maxima = []
    n_groups = len(results) // GROUP_SIZE
    for g in range(n_groups):
        chunk = results[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
        arrays = []
        for r in chunk:
            if factor in r[2]:
                arrays.append(r[2][factor])
        if len(arrays) > 0:
            maxima.append(np.concatenate(arrays).max())
    return float(np.mean(maxima))


def cell(taus):
    if len(taus) >= 2:
        return "%.2f+-%.2f [%d]" % (np.mean(taus), np.std(taus, ddof=1), len(taus))
    return "too few groups"


if __name__ == "__main__":
    print("K = %d, ratio %g, smin = %d (mean +- std between groups [groups])"
          % (K, RATIO, SMIN))
    modes = ["observed"] + SMAX_FIXED
    for f in FACTORS:
        print("\nV* = %.2f" % f)
        header = "   wires  networks  mean max S |"
        for m in modes:
            header = header + "  smax=%-9s" % str(m)
        print(header)
        for n_wires, side in SIZES:
            fname = find_cache(n_wires, side)
            if fname is None:
                print("   %5d  (no cache file found)" % n_wires)
                continue
            with open(os.path.join(FOLDER, fname), "rb") as fh:
                results = pickle.load(fh)
            line = "   %5d  %8d  %10.0f |" % (n_wires, len(results),
                                                  mean_group_max(results, f))
            for m in modes:
                line = line + "  %-15s" % cell(group_fits(results, f, m))
            print(line, flush=True)
