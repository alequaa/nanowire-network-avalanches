# tau_without_largest.py
# Computes the exponent tau removing from every network the LARGEST avalanche
# (instead of the one in which the path forms). It runs no simulation: it reads
# the cache_nospan_*.pkl files, where for every network and every V* there is
# the list of all the avalanche sizes S.
#
# Numerical example: S = [3, 1, 14, 2]. The largest is 14. Without it: [3, 1, 2].
#
# As a check it also recomputes the column "all" and the column "without the
# path avalanche" with the same method (groups of networks, smin = 2).
#
# How to run (from the main folder, with the script placed in scripts/ next to
# the cache files):
#   python scripts/tau_without_largest.py
import os
import sys
import pickle

import numpy as np

# nw_tau.py is in the main folder of the project, i.e. ONE FOLDER ABOVE this
# script (which goes in scripts/, next to the cache_nospan_*.pkl files).
# The line below tells Python to look for modules in the folder above as well.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nw_tau import fit_tau

SMIN = 2
SMAX_FIXED = 100
MIN_AVALANCHES = 30
FACTORS = [0.5, 0.7, 0.9, 1.0, 1.05, 1.2, 1.4, 1.8, 2.5]
# (K, wires, side, n_networks, networks_per_group)
JOBS = [
    (1, 100, 30, 1000, 50), (1, 400, 60, 1000, 50), (1, 900, 90, 500, 50),
    (10, 100, 30, 1000, 50), (10, 400, 60, 250, 25), (10, 900, 90, 100, 20),
]


def sizes_for(rec, which):
    # rec = (S_all, S_without_path_avalanche, has_path, S_path_avalanche)
    # which = "all" | "largest" | "spanning"
    s_all = np.asarray(rec[0], dtype=int)
    if which == "all":
        return s_all
    if which == "spanning":
        return np.asarray(rec[1], dtype=int)
    # "largest": ONE element is removed, the one with the maximum value
    # (if two are equal, only one of them is removed)
    idx = int(np.argmax(s_all))
    return np.delete(s_all, idx)


def group_taus(results, f, which, smax_mode, group):
    taus = []
    for g in range(len(results) // group):
        chunk = results[g * group:(g + 1) * group]
        arrays = []
        for r in chunk:
            if f in r[2]:
                arrays.append(sizes_for(r[2][f], which))
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
    return "troppo pochi gruppi"


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for (K, w, side, n, group) in JOBS:
        name = "cache_nospan_r1000_K%d_w%d_L%d_n%d.pkl" % (K, w, side, n)
        with open(os.path.join(here, name), "rb") as fh:
            results = pickle.load(fh)
        print("\n=== K=%d, %d fili, %d reti, gruppi da %d" % (K, w, len(results), group))
        print("  V*  | smax=100: tutte | senza piu' grande | senza cammino | "
              "smax=osservato: tutte | senza piu' grande | senza cammino")
        for f in FACTORS:
            line = "  %.2f |" % f
            for mode in (100, "observed"):
                for which in ("all", "largest", "spanning"):
                    line = line + " %-18s" % cell(group_taus(results, f, which, mode, group))
                line = line + "|"
            print(line)
