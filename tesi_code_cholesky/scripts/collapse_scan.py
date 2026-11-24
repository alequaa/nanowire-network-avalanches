# collapse_scan.py
# Data collapse of the avalanche size distribution P(S, V*) (no simulation:
# it only reads the cache_nospan_*.pkl files made by run_tau_no_spanning.py).
#
# Scaling form asked for:   P(S, V*) = S^(-tau) * f( S * |V* - Vc|^a )
# If it holds, the curves  y = S^tau * P(S)  of different V* fall on ONE curve
# when plotted against  x = S * |V* - Vc|^a.   Unknowns: tau, Vc, a.
#
# Numerical example (invented): Vc = 1, a = 2. At V* = 1.2 -> |V*-Vc|^a = 0.04,
# so S = 100 sits at x = 4. At V* = 1.4 -> 0.16, so S = 25 sits at x = 4 too:
# the two points land on top of each other, if the form is right.
#
# How "good collapse" is measured (cost):
#   1. each curve is drawn in log10(x), log10(y);
#   2. on a common grid of log10(x) the curves are interpolated;
#   3. at every grid point where at least 2 curves exist, we take the variance
#      of log10(y) between the curves; the cost is the average of these variances.
#   Cost = 0 means perfect overlap. A fit is only meaningful if the curves
#   overlap on a decent part of the x axis (see MIN_OVERLAP).
#
# Run (from the main folder, script placed in scripts/ next to the cache files):
#   python scripts/collapse_scan.py
# Optional environment variables:
#   K (default 1), WIRES (default 400), SIDE (default 60 -> must match WIRES:
#   100 -> 30, 400 -> 60, 900 -> 90), WHICH = all | largest | spanning
#   (default all), VSTARS (default 1.05,1.2,1.4,1.8,2.5), RATIO (default 1000)
# Output: a table on the screen and an image in images/ (the name contains K,
# wires, WHICH, number of networks, the Vc mode and the SIDES mode).
#
# NEW in this version (18 Nov):
#  1. V* BELOW 1 can be used (for example VSTARS=0.7,0.9,1.05,1.2,1.4,1.8,2.5).
#     Vc is now searched over the whole range 0 .. VC_MAX, also INSIDE the range
#     of V*. A V* too close to Vc (|V* - Vc| < MIN_DISTANCE) is refused: do not
#     put 1.0 in VSTARS if you want to test Vc = 1.
#  2. VC_FIXED: if set (for example VC_FIXED=1.0), Vc is NOT a free parameter,
#     only tau and a are fitted.
#  3. SIDES = one (default) -> ONE scaling function f for the curves on both
#     sides of Vc. SIDES = two -> two functions: the curves with V* < Vc must
#     overlap among themselves, the curves with V* > Vc must overlap among
#     themselves (tau, Vc and a are shared). The cost is the average of the
#     costs of the two groups. A group with only one curve cannot be judged
#     and is left out.
#  4. The image name contains the number of networks (NMAX) so that runs do not
#     overwrite each other.
#  5. A table "best cost for each Vc" shows how well Vc is fixed.

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

RATIO = float(os.environ.get("RATIO", "1000"))
K = int(os.environ.get("K", "1"))
WIRES = int(os.environ.get("WIRES", "400"))
SIDE = float(os.environ.get("SIDE", "60"))
WHICH = os.environ.get("WHICH", "all")
VSTARS = [float(v) for v in os.environ.get("VSTARS", "1.05,1.2,1.4,1.8,2.5").split(",")]
# VC_FIXED: empty = Vc free; a number (for example 1.0) = Vc fixed to that value.
VC_FIXED_TEXT = os.environ.get("VC_FIXED", "").strip()
if VC_FIXED_TEXT == "":
    VC_FIXED = None
else:
    VC_FIXED = float(VC_FIXED_TEXT)
# SIDES: "one" = one scaling function for both sides of Vc, "two" = one per side.
SIDES = os.environ.get("SIDES", "one").strip()
if SIDES != "one" and SIDES != "two":
    raise SystemExit("SIDES must be one or two")
VC_MAX = float(os.environ.get("VC_MAX", "2.0"))   # largest Vc searched
# NMAX = use only the first NMAX networks of the cache (0 = all). It serves to
# test how much the result depends on the number of networks.
NMAX = int(os.environ.get("NMAX", "0"))
FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(FOLDER, "images")

MIN_COUNTS = 5        # a bin with fewer avalanches is too noisy: not used
MIN_OVERLAP = 0.3     # at least 30% of the x grid must be covered by >= 2 curves
N_GRID = 60           # points of the common log10(x) grid
MIN_DISTANCE = 0.02   # |V* - Vc| below this is refused (x would go to 0)


def find_cache():
    # Cache file with the most networks for this size (same logic as the
    # other scripts). Searches in the script folder and in a "data" subfolder.
    prefix = "cache_nospan_r%g_K%d_w%d_L%g_n" % (RATIO, K, WIRES, SIDE)
    best_n = 0
    best_path = None
    for folder in (FOLDER, os.path.join(FOLDER, "data")):
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.startswith(prefix) and fname.endswith(".pkl"):
                n = int(fname[len(prefix):-4])
                if n > best_n:
                    best_n = n
                    best_path = os.path.join(folder, fname)
    return best_path


def sizes_of(rec):
    # rec = (S_all, S_without_path_avalanche, has_path, S_path_avalanche)
    # WHICH decides which avalanches are used.
    s_all = np.asarray(rec[0], dtype=int)
    if WHICH == "all":
        return s_all
    if WHICH == "spanning":
        return np.asarray(rec[1], dtype=int)
    # "largest": remove ONE avalanche, the biggest (same rule as
    # tau_without_largest.py)
    idx = int(np.argmax(s_all))
    return np.delete(s_all, idx)


def make_edges(results):
    # ONE common set of bin edges for all V*: same bins -> curves comparable.
    # Same recipe as log_binned in the other scripts (24 log bins, width 1
    # at small S).
    s_max = 1
    for r in results:
        for f in VSTARS:
            if f in r[2]:
                s = sizes_of(r[2][f])
                if s.shape[0] > 0:
                    s_max = max(s_max, int(s.max()))
    edges = np.unique(np.round(np.logspace(0.0, np.log10(s_max + 1.0), 25))).astype(int)
    return edges


def build_curves(results, edges):
    # For every V*: (bin centres, P(S), counts), only bins with >= MIN_COUNTS.
    # P(S) = avalanches in the bin / bin width / total avalanches.
    curves = {}
    centres_all = np.sqrt(edges[:-1] * (edges[1:] - 1.0 + 1e-9))
    widths = np.diff(edges)
    for f in VSTARS:
        arrays = []
        for r in results:
            if f in r[2]:
                s = sizes_of(r[2][f])
                if s.shape[0] > 0:
                    arrays.append(s)
        if len(arrays) == 0:
            continue
        pooled = np.concatenate(arrays)
        counts, _ = np.histogram(pooled, bins=edges)
        density = counts / widths / float(pooled.shape[0])
        keep = counts >= MIN_COUNTS
        curves[f] = (centres_all[keep], density[keep], counts[keep])
    return curves


def group_variance(lxs, lys):
    # Average variance (between curves) of log10(y) on a common grid of log10(x),
    # for ONE group of curves. Returns (cost, overlap fraction); (inf, overlap)
    # if the curves hardly overlap. Needs at least 2 curves.
    if len(lxs) < 2:
        return np.inf, 0.0
    lo = min(lx.min() for lx in lxs)
    hi = max(lx.max() for lx in lxs)
    if hi - lo < 1e-9:
        return np.inf, 0.0
    grid = np.linspace(lo, hi, N_GRID)
    table = np.full((len(lxs), N_GRID), np.nan)
    for i in range(len(lxs)):
        order = np.argsort(lxs[i])
        inside = (grid >= lxs[i].min()) & (grid <= lxs[i].max())
        table[i, inside] = np.interp(grid[inside], lxs[i][order], lys[i][order])
    n_curves_here = np.sum(~np.isnan(table), axis=0)
    ok = n_curves_here >= 2
    overlap = float(np.sum(ok)) / N_GRID
    if overlap < MIN_OVERLAP:
        return np.inf, overlap
    variances = np.nanvar(table[:, ok], axis=0)
    return float(np.mean(variances)), overlap


def collapse_cost(curves, tau, vc, a):
    # Cost of the collapse for the parameters (tau, Vc, a).
    # Every curve is moved to  x = S |V* - Vc|^a ,  y = S^tau P(S)  (log10).
    # SIDES = "one": all the curves are in one group.
    # SIDES = "two": curves with V* < Vc form a group, curves with V* > Vc form
    # another; the cost is the mean of the costs of the groups that have >= 2
    # curves (if no group has 2 curves the cost is infinite).
    below_x = []
    below_y = []
    above_x = []
    above_y = []
    for f in curves:
        dist = abs(f - vc)
        if dist < MIN_DISTANCE:
            return np.inf, 0.0
        s, p, c = curves[f]
        x = np.log10(s * dist ** a)
        y = np.log10(s ** tau * p)
        if f < vc:
            below_x.append(x)
            below_y.append(y)
        else:
            above_x.append(x)
            above_y.append(y)
    if SIDES == "one":
        return group_variance(below_x + above_x, below_y + above_y)
    costs = []
    overlaps = []
    for lxs, lys in ((below_x, below_y), (above_x, above_y)):
        if len(lxs) >= 2:
            cost, ov = group_variance(lxs, lys)
            if not np.isfinite(cost):
                return np.inf, ov
            costs.append(cost)
            overlaps.append(ov)
    if len(costs) == 0:
        return np.inf, 0.0
    return float(np.mean(costs)), float(np.mean(overlaps))


def scan(curves):
    # Coarse grid over (tau, Vc, a), then a local refinement of the best point.
    taus = np.arange(1.0, 3.01, 0.1)
    if VC_FIXED is None:
        vcs = np.arange(0.0, VC_MAX + 1e-9, 0.05)
    else:
        vcs = np.array([VC_FIXED])
    avals = np.arange(-2.0, 2.01, 0.1)
    best = None
    all_points = []
    for tau in taus:
        for vc in vcs:
            for a in avals:
                cost, ov = collapse_cost(curves, tau, vc, a)
                if np.isfinite(cost):
                    all_points.append((cost, tau, vc, a))
                    if best is None:
                        best = (cost, tau, vc, a)
                    else:
                        if cost < best[0]:
                            best = (cost, tau, vc, a)
    if best is None:
        return None, []

    def fun(p):
        # p = [tau, Vc, a] (Vc free) or [tau, a] (Vc fixed). Outside the allowed
        # ranges the cost is made huge, so the refinement cannot run away to a
        # meaningless minimum.
        if VC_FIXED is None:
            tau_p = p[0]
            vc_p = p[1]
            a_p = p[2]
        else:
            tau_p = p[0]
            vc_p = VC_FIXED
            a_p = p[1]
        if vc_p < 0.0 or vc_p > VC_MAX:
            return 1.0e6
        if tau_p < 1.0 or tau_p > 3.0:
            return 1.0e6
        cost, ov = collapse_cost(curves, tau_p, vc_p, a_p)
        if not np.isfinite(cost):
            return 1.0e6
        return cost

    if VC_FIXED is None:
        start = np.array([best[1], best[2], best[3]])
    else:
        start = np.array([best[1], best[3]])
    out = minimize(fun, start, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 600})
    if VC_FIXED is None:
        refined = (float(out.fun), float(out.x[0]), float(out.x[1]), float(out.x[2]))
    else:
        refined = (float(out.fun), float(out.x[0]), VC_FIXED, float(out.x[1]))
    if refined[0] < best[0]:
        best = refined
    return best, all_points


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    path = find_cache()
    if path is None:
        print("no cache file found for K=%d, %d wires, side %g" % (K, WIRES, SIDE))
        sys.exit(1)
    with open(path, "rb") as fh:
        results = pickle.load(fh)
    if NMAX > 0:
        results = results[:NMAX]
    edges = make_edges(results)
    curves = build_curves(results, edges)
    print("K = %d, %d wires, %d networks, avalanches used: %s" % (K, WIRES, len(results), WHICH))
    if VC_FIXED is None:
        print("Vc free in [0, %.2f]; SIDES = %s" % (VC_MAX, SIDES))
    else:
        print("Vc FIXED = %.2f; SIDES = %s" % (VC_FIXED, SIDES))
    for f in curves:
        print("  V* = %.2f: %d bins used, S from %d to %d" % (
            f, curves[f][0].shape[0], curves[f][0].min(), curves[f][0].max()))
    if len(curves) < 3:
        print("fewer than 3 curves: collapse not meaningful")
        sys.exit(1)

    # baseline: no scaling at all (a = 0) and tau free -> how bad are raw curves?
    base = None
    for tau in np.arange(1.0, 3.01, 0.05):
        cost, ov = collapse_cost(curves, tau, 0.0, 0.0)
        if np.isfinite(cost):
            if base is None:
                base = (cost, tau)
            else:
                if cost < base[0]:
                    base = (cost, tau)
    best, points = scan(curves)
    if best is None:
        print("no admissible parameters found")
        sys.exit(1)
    print("\nbaseline (a = 0, curves just rescaled by S^tau): cost %.4f at tau = %.2f" % (base[0], base[1]))
    print("best collapse: cost %.4f   tau = %.2f   Vc = %.2f   a = %.2f" % (best[0], best[1], best[2], best[3]))

    # how well is each parameter fixed? range among points with cost <= 1.5 * best
    near = [p for p in points if p[0] <= 1.5 * best[0]]
    print("points of the grid with cost <= 1.5 x best: %d (of %d)" % (len(near), len(points)))
    if len(near) > 0:
        arr = np.array(near)
        print("   tau in [%.2f, %.2f]   Vc in [%.2f, %.2f]   a in [%.2f, %.2f]" % (
            arr[:, 1].min(), arr[:, 1].max(), arr[:, 2].min(), arr[:, 2].max(),
            arr[:, 3].min(), arr[:, 3].max()))

    # best cost for each Vc (every 0.1): how well is Vc fixed by the data?
    # For each Vc we keep the best (tau, a). A flat column of costs = Vc not fixed.
    if VC_FIXED is None:
        print("\nbest cost for each Vc (minimum over tau and a, grid points):")
        print("   (Vc equal to one of the V* used is refused, so those rows are missing)")
        print("   Vc    cost     tau    a")
        vc_values = sorted(set(round(p[2], 2) for p in points))
        for vc_value in vc_values:
            if abs(vc_value * 10.0 - round(vc_value * 10.0)) > 1e-6:
                continue
            same = [p for p in points if abs(p[2] - vc_value) < 1e-6]
            top = min(same, key=lambda p: p[0])
            print("  %4.2f  %.4f  %5.2f  %5.2f" % (vc_value, top[0], top[1], top[3]))

    # figure: raw curves (left) and collapsed curves (right)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(curves)))
    k = 0
    for f in curves:
        s, p, c = curves[f]
        # dashed line = curve with V* below Vc, solid = above
        if f < best[2]:
            style = "--o"
        else:
            style = "-o"
        axes[0].plot(s, p, style, color=colors[k], markersize=3, label="V* = %.2f" % f)
        x = s * abs(f - best[2]) ** best[3]
        y = s ** best[1] * p
        axes[1].plot(x, y, style, color=colors[k], markersize=3, label="V* = %.2f" % f)
        k = k + 1
    axes[0].set_xlabel("S")
    axes[0].set_ylabel("P(S)")
    axes[0].set_title("raw distributions")
    axes[1].set_xlabel("S |V* - Vc|^a")
    axes[1].set_ylabel("S^tau P(S)")
    axes[1].set_title("collapse: tau = %.2f, Vc = %.2f, a = %.2f" % (best[1], best[2], best[3]))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("K = %d, %d wires, %s avalanches" % (K, WIRES, WHICH), fontsize=10)
    fig.tight_layout()
    if VC_FIXED is None:
        vc_text = "vcfree"
    else:
        vc_text = "vc%g" % VC_FIXED
    name = "collapse_K%d_w%d_%s_n%d_%s_%s.png" % (K, WIRES, WHICH, len(results), vc_text, SIDES)
    fig.savefig(os.path.join(IMAGES, name), dpi=140)
    print("saved images/%s" % name)
