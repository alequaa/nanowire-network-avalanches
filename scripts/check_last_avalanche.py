# check_last_avalanche.py
# Two checks on the avalanches already saved by run_average_size.py
# (no new simulation):
#
# 1. Is the bump made of the LAST avalanche of each network?
#    Idea: avalanches start small and grow while the loop goes on, so the last
#    one can be huge and scale with the whole system, like the percolating
#    cluster in percolation (it is left out of cluster statistics).
#    For every network we compare its last avalanche with its largest one,
#    and we refit tau with and without the last avalanche.
# 1b. The same for the FIRST avalanche of each network (the large avalanches
#    seem to be at the start of the run).
# 2. Do avalanches grow during the run?
#    S of every avalanche against its position in the run (0 = first,
#    1 = last), all networks of one size together.
#
# Example: network with avalanches S = [1, 2, 1, 3, 40]  (in time order)
#   last = 40 = largest -> "last is the largest". Without the last: [1, 2, 1, 3].
#
# Run (from the main folder, after run_average_size.py):
#   python scripts/check_last_avalanche.py
# Output: table on screen and images/last_avalanche.png, images/size_vs_order.png
# Optional environment variables: K (default 10), RATIO (default 1000),
# SMIN (default 2).

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nw_tau import fit_tau

RATIO = float(os.environ.get("RATIO", "1000"))
K = int(os.environ.get("K", "10"))
SMIN = int(os.environ.get("SMIN", "2"))
FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(FOLDER, "images")
SIZES = [(100, 30.0), (400, 60.0), (900, 90.0)]
FACTORS = [1.05, 1.8]
GROUP_SIZE = 10
MIN_AVALANCHES = 30
COLORS = ["#9ec5ee", "#3f7fc4", "#0f3d75"]


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


def network_arrays(results, f, drop_last, drop_first=False):
    # List of avalanche-size arrays (time order), one per network.
    out = []
    for r in results:
        if f in r[2]:
            arr = r[2][f]
            if arr.shape[0] == 0:
                continue
            if drop_last:
                arr = arr[:-1]
            if drop_first:
                arr = arr[1:]
            out.append(arr)
    return out


def group_tau(arrays, smax_mode):
    # Mean and std of tau over groups of GROUP_SIZE networks.
    taus = []
    for g in range(len(arrays) // GROUP_SIZE):
        pooled = np.concatenate(arrays[g * GROUP_SIZE:(g + 1) * GROUP_SIZE])
        if pooled.shape[0] < MIN_AVALANCHES:
            continue
        if smax_mode == "observed":
            smax = int(pooled.max())
        else:
            smax = int(smax_mode)
        t = fit_tau(pooled, SMIN, smax)
        if not np.isnan(t):
            taus.append(t)
    if len(taus) < 2:
        return "too few groups"
    return "%.2f+-%.2f" % (np.mean(taus), np.std(taus, ddof=1))


def log_binned(sizes):
    edges = np.unique(np.round(np.logspace(0.0, np.log10(sizes.max() + 1.0), 25))).astype(int)
    counts, _ = np.histogram(sizes, bins=edges)
    density = counts / np.diff(edges) / sizes.shape[0]
    centres = np.sqrt(edges[:-1] * (edges[1:] - 1.0 + 1e-9))
    keep = counts > 0
    return centres[keep], density[keep]


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    print("K = %d, ratio %g, smin = %d (tau: mean +- std between groups)" % (K, RATIO, SMIN))
    for fi, f in enumerate(FACTORS):
        print("\nV* = %.2f" % f)
        print("   wires  nets | last = largest | median S_last/S_max | "
              "tau all (obs)   tau no last (obs) | tau all (<=100)  tau no last (<=100)")
        for (n_wires, side), color in zip(SIZES, COLORS):
            fname = find_cache(n_wires, side)
            if fname is None:
                print("   %5d  (no cache file)" % n_wires)
                continue
            with open(os.path.join(FOLDER, fname), "rb") as fh:
                results = pickle.load(fh)
            full = network_arrays(results, f, False)
            no_last = network_arrays(results, f, True)
            is_largest = 0
            ratios = []
            for arr in full:
                if arr[-1] == arr.max():
                    is_largest = is_largest + 1
                ratios.append(arr[-1] / float(arr.max()))
            print("   %5d  %4d | %6.0f%%        | %10.2f          | %-15s %-17s | %-16s %s" % (
                n_wires, len(full), 100.0 * is_largest / len(full), np.median(ratios),
                group_tau(full, "observed"), group_tau(no_last, "observed"),
                group_tau(full, 100), group_tau(no_last, 100)), flush=True)

            # same check for the FIRST avalanche
            no_first = network_arrays(results, f, False, True)
            first_largest = 0
            first_ratio = []
            for arr in full:
                if arr[0] == arr.max():
                    first_largest = first_largest + 1
                first_ratio.append(arr[0] / float(arr.max()))
            print("          first: largest in %3.0f%% of networks, median S_first/S_max %.2f | "
                  "tau no first: %s (obs), %s (<=100)" % (
                      100.0 * first_largest / len(full), np.median(first_ratio),
                      group_tau(no_first, "observed"), group_tau(no_first, 100)), flush=True)

            # figure 1: P(S) with and without the last avalanche
            x, p = log_binned(np.concatenate(full))
            axes1[fi].plot(x, p, "-o", color=color, linewidth=2.0, markersize=4,
                           label="%d wires, all avalanches" % n_wires)
            x2, p2 = log_binned(np.concatenate(no_last))
            axes1[fi].plot(x2, p2, "--", color=color, linewidth=1.6,
                           label="%d wires, without last" % n_wires)

            # figure 2: S against position in the run
            xs = []
            ys = []
            for arr in full:
                n = arr.shape[0]
                if n < 2:
                    continue
                xs.append(np.arange(n) / float(n - 1))
                ys.append(arr)
            axes2[fi].scatter(np.concatenate(xs), np.concatenate(ys), s=5, alpha=0.25,
                              color=color, label="%d wires" % n_wires)
        for ax in (axes1[fi], axes2[fi]):
            ax.set_yscale("log")
            ax.set_title("V* = %.2f" % f)
            ax.grid(True, which="major", color="0.88", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes1[fi].set_xscale("log")
        axes1[fi].set_xlabel("avalanche size S")
        axes1[fi].legend(fontsize=7.5, frameon=False, loc="lower left")
        axes2[fi].set_xlabel("position in the run (0 = first avalanche, 1 = last)")
        axes2[fi].legend(fontsize=8, frameon=False, loc="upper left")
    axes1[0].set_ylabel("P(S)")
    axes2[0].set_ylabel("avalanche size S")
    fig1.tight_layout()
    fig1.savefig(os.path.join(IMAGES, "last_avalanche.png"), dpi=150)
    fig2.tight_layout()
    fig2.savefig(os.path.join(IMAGES, "size_vs_order.png"), dpi=150)
    print("\nsaved images/last_avalanche.png and images/size_vs_order.png")
