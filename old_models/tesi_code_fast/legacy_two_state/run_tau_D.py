# run_tau_D.py
# Definition D: avalanches at fixed voltage, V* = V / Vth, as in Hochstetter
# (DOI 10.1038/s41467-021-24260-z), with the two-state loop (nw_loop_model.py).
#   1. Per network, Vth = smallest fixed voltage at which, starting from all
#      junctions OFF and Lambda = 0, the ON junctions connect source and drain
#      (bisection; the loop is run to its steady state at each trial voltage).
#   2. Run the loop at V = f * Vth for f in FACTORS (V* = 0.7, 1, 1.05, 1.8).
#   3. Avalanches with find_avalanches (frames of mean inter-switch interval).
#   4. Pool the networks; fit tau (fixed smin, and Hochstetter's range rule).
# Run: python run_tau_D.py         

# The shared library files (nanowire_network.py, nw_solver.py, ...) are in the
# main folder, one level up: add it to the search path.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pickle
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nanowire_network as nn
from nw_loop_model import run_loop
from nw_ramp import on_path_exists
from nw_avalanches import find_avalanches
from nw_tau import fit_tau, bootstrap_tau
from nw_tau_extra import select_range, ks_pvalue

V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0

FACTORS = [0.7, 1.0, 1.05, 1.8]
TOL = 2.0e-4               # bisection precision on Vth (V)
V_LOW = 0.011              # just above Vset: below it nothing can switch
SMIN_FIXED = [1, 2, 3]
XMIN_LIST = [1, 2, 3, 4, 5, 6, 8, 10]
N_BOOT = 200

# name, number of networks, wires, side (um), electrodes
CASES = [
    ("small", 1000, 100, 30.0, "point"),
    ("large", 100, 1000, 100.0, "strip"),
]
COLORS = ["#9ac0e6", "#d9622b", "#3b7bbf", "#143d6b"]
FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES_FOLDER = os.path.join(FOLDER, "images")


def path_forms(net, s, d, v):
    # Fresh run (all OFF, Lambda = 0) at voltage v: do the ON junctions
    # connect a source wire to a drain wire?
    res = run_loop(net.n_wires(), net.junctions, s, d, v,
                   V_SET, LAM_CRIT, G_OFF, G_ON)
    on_list = np.nonzero(res["is_on"])[0]
    return on_path_exists(net.n_wires(), net.junctions, on_list, s, d)


def find_threshold(net, s, d):
    # Smallest voltage (to precision TOL) with a source-drain ON path.
    # Assumes: more voltage never destroys the path (bisection).
    lo = V_LOW
    v = 0.05
    found = False
    while v <= 0.8 and not found:
        if path_forms(net, s, d, v):
            found = True
        else:
            lo = v
            v = v * 2.0
    if not found:
        return float("nan")
    hi = v
    while hi - lo > TOL:
        mid = 0.5 * (lo + hi)
        if path_forms(net, s, d, mid):
            hi = mid
        else:
            lo = mid
    return hi


def one_network(args):
    # Worker: returns (seed, Vth, {factor: array of avalanche sizes}).
    seed, n_wires, side, electrodes = args
    net = nn.generate_network(n_wires=n_wires, side=side, seed=seed)
    if electrodes == "point":
        s, d = nn.point_electrodes(net)
    else:
        s, d = nn.strip_electrodes(net, 0.025)
    vth = find_threshold(net, s, d)
    out = {}
    if np.isnan(vth):
        return seed, vth, out
    for f in FACTORS:
        res = run_loop(net.n_wires(), net.junctions, s, d, f * vth,
                       V_SET, LAM_CRIT, G_OFF, G_ON)
        av = find_avalanches(res["event_time"])
        out[f] = np.asarray(av["S"], dtype=int)
    return seed, vth, out


def get_results(name, n_net, n_wires, side, electrodes):
    cache = os.path.join(FOLDER, "cache_D_%s_%d.pkl" % (name, n_net))
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    jobs = [(seed, n_wires, side, electrodes) for seed in range(n_net)]
    with Pool(2) as pool:
        results = pool.map(one_network, jobs, chunksize=4)
    with open(cache, "wb") as fh:
        pickle.dump(results, fh)
    return results


if __name__ == "__main__":
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for c in range(len(CASES)):
        name, n_net, n_wires, side, electrodes = CASES[c]
        results = get_results(name, n_net, n_wires, side, electrodes)
        good = [r for r in results if not np.isnan(r[1])]
        vth = np.array([r[1] for r in good])
        print("\n=== %s: %d of %d networks form a path; Vth/Vset: mean %.2f, "
              "std %.2f, min %.1f, max %.1f" %
              (name, len(good), n_net, vth.mean() / V_SET, vth.std() / V_SET,
               vth.min() / V_SET, vth.max() / V_SET))
        ax = axes[c]
        for k in range(len(FACTORS)):
            f = FACTORS[k]
            per_net = [r[2][f] for r in good if r[2][f].shape[0] > 0]
            if len(per_net) == 0:
                print("\n V* = %.2f: no avalanches" % f)
                continue
            pooled = np.concatenate(per_net)
            print("\n V* = %.2f: %d avalanches (%d networks with events), "
                  "mean S %.2f, max S %d, frac(S=1) %.3f" %
                  (f, pooled.shape[0], len(per_net), pooled.mean(),
                   pooled.max(), np.mean(pooled == 1)))
            values, counts = np.unique(pooled, return_counts=True)
            ax.loglog(values, counts / counts.sum(), "o", markersize=3.5,
                      color=COLORS[k], label="V* = " + str(f))
            if pooled.shape[0] < 100:
                print("   too few avalanches for a fit")
                continue
            smax = int(pooled.max())
            line = "   fixed smin (tau +- boot std):"
            for smin in SMIN_FIXED:
                t = fit_tau(pooled, smin, smax)
                m, sd = bootstrap_tau(per_net, smin, smax, N_BOOT)
                line = line + "  smin=%d: %.3f +- %.3f;" % (smin, t, sd)
            print(line)
            if f in (1.0, 1.05):
                r = select_range(pooled, XMIN_LIST)
                if r is None:
                    print("   Hochstetter rule: no valid range")
                else:
                    decades = np.log10(r["xmax"] / r["xmin"])
                    p_strict = ks_pvalue(pooled, r["xmin"], r["xmax"], r["tau"],
                                         n_synth=200, refit=True)
                    print("   Hochstetter rule: range %d-%d (%.2f decades), tau = "
                          "%.3f, n = %d, p = %.2f (%s), p with refit = %.2f" %
                          (r["xmin"], r["xmax"], decades, r["tau"], r["n"],
                           r["p"], "passed" if r["passed"] else "NOT passed",
                           p_strict))
        ss = np.array([1.0, 100.0])
        ax.loglog(ss, 0.5 * ss ** (-1.95), "k--", linewidth=1,
                  label="slope -1.95")
        ax.set_xlabel("avalanche size S")
        ax.set_ylabel("P(S)")
        ax.set_title(name + ": " + str(len(good)) + " networks, " + electrodes)
        ax.grid(alpha=0.25, which="both")
        ax.legend()
    path = os.path.join(IMAGES_FOLDER, "tau_definition_D.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:", path)
