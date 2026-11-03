# plot_without_largest.py   (version for the scan over many V*)
#
# Reads the cache_nospan_*.pkl files created by run_tau_no_spanning.py
# (same K, same ratio) and runs NO simulation.
#
# For every size (100, 400, 900 wires) and every V* of the scan it draws the
# distribution P(S) of the avalanche sizes:
#   solid line  = all avalanches
#   dashed line = all avalanches except ONE per network (the "removed" one)
# It also draws a summary figure as a function of V*.
#
# Which avalanche is removed (environment variable MODE):
#   largest  (default) : the LARGEST avalanche of every network
#   spanning           : the avalanche in which the first ON path between
#                        source and drain forms (the two are the same in
#                        91-100% of the networks with K = 1 from V* = 1;
#                        the script prints how often)
#
# Numerical example: a network has avalanches S = [3, 1, 20, 2], 26 events in
# total.
#   MODE=largest -> 20 is removed, [3, 1, 2] is left.
#   The "share of events in the removed avalanche" is 20 / 26 = 77%.
#
# Content of the cache: one entry per network (seed, vth, out) with
#   out[V*] = (S_all, S_without_spanning, has_path, S_spanning)
# If a network has no avalanche at a given V*, out does not contain that V*.
#
# How to run (from the main folder, in cmd):
#   set K=1
#   python scripts/plot_without_largest.py
# Optional variables: K (default 1), RATIO (default 1000), MODE (default largest).
#
# Files produced in scripts/images:
#   without_<MODE>_K<K>_w<wires>.png : one figure per size, one panel per V*
#   summary_<MODE>_K<K>.png          : share of events in the removed avalanche,
#                                      its median size, fraction of networks
#                                      with at least one avalanche, all vs V*

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATIO = float(os.environ.get("RATIO", "1000"))
K = int(os.environ.get("K", "1"))
MODE = os.environ.get("MODE", "largest")
MIN_AVALANCHES = 30     # below this number of avalanches the panel is not drawn
FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(FOLDER, "images")
SIZES = [(100, 30.0), (400, 60.0), (900, 90.0)]   # (wires, side in um)
COLORS = ["#9ec5ee", "#3f7fc4", "#0f3d75"]        # light -> dark with the size


def find_cache(n_wires, side):
    # Looks for the cache file with the most networks for this size.
    # Returns the file name, or None if there is none.
    prefix = "cache_nospan_r%g_K%d_w%d_L%g_n" % (RATIO, K, n_wires, side)
    best_n = 0
    best_file = None
    for fname in os.listdir(FOLDER):
        if fname.startswith(prefix) and fname.endswith(".pkl"):
            n = int(fname[len(prefix):-4])
            if n > best_n:
                best_n = n
                best_file = fname
    return best_file


def log_binned(sizes):
    # Histogram with logarithmic bins (about 24 bins). Returns (centres, P(S)).
    # P(S) = avalanches in the bin / bin width / total avalanches.
    # Example: 3 avalanches in the bin S = 10..13 (width 4), 300 in total
    # -> P = 3 / 4 / 300 = 0.0025.
    edges = np.unique(np.round(np.logspace(0.0, np.log10(sizes.max() + 1.0), 25))).astype(int)
    counts, _ = np.histogram(sizes, bins=edges)
    density = counts / np.diff(edges) / sizes.shape[0]
    centres = np.sqrt(edges[:-1] * (edges[1:] - 1.0 + 1e-9))
    keep = counts > 0
    return centres[keep], density[keep]


def split_network(out, factor):
    # For one network and one V* returns (sizes_all, sizes_without, removed, share, same)
    #   sizes_all     : all the avalanches of the network
    #   sizes_without : the same without the removed one (None if nothing is removed)
    #   removed       : size of the removed avalanche (0 if nothing is removed)
    #   share         : removed / total events of the network
    #   same          : True if the largest is also the one in which the path forms
    # Returns None if the network has no avalanche at this V*.
    if factor not in out:
        return None
    sizes_all = np.asarray(out[factor][0], dtype=int)
    has_path = out[factor][2]
    span_size = int(out[factor][3])
    if sizes_all.shape[0] == 0:
        return None
    same = False
    if has_path:
        if span_size == int(sizes_all.max()):
            same = True
    if MODE == "spanning":
        if has_path:
            sizes_without = np.asarray(out[factor][1], dtype=int)
            removed = span_size
        else:
            # no path: there is nothing to remove
            sizes_without = sizes_all
            removed = 0
    else:
        index_largest = int(np.argmax(sizes_all))
        sizes_without = np.delete(sizes_all, index_largest)
        removed = int(sizes_all[index_largest])
    share = removed / float(sizes_all.sum())
    return sizes_all, sizes_without, removed, share, same


def load_size(n_wires, side):
    fname = find_cache(n_wires, side)
    if fname is None:
        return None
    with open(os.path.join(FOLDER, fname), "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    if MODE != "largest":
        if MODE != "spanning":
            print("MODE deve essere largest oppure spanning")
            sys.exit(1)
    os.makedirs(IMAGES, exist_ok=True)
    print("K = %d, rapporto %g, MODE = %s" % (K, RATIO, MODE))

    # the V* values are read from the cache itself: union of the V* found in the networks
    summary = {}      # summary[n_wires][V*] = dict with the statistics
    all_factors = set()
    loaded = {}
    for (n_wires, side) in SIZES:
        results = load_size(n_wires, side)
        if results is None:
            print("taglia %d: nessun file cache, salto" % n_wires)
            continue
        loaded[n_wires] = results
        for r in results:
            for f in r[2].keys():
                all_factors.add(f)
    factors = sorted(all_factors)
    if len(factors) == 0:
        print("nessun dato trovato")
        sys.exit(1)

    n_cols = 3
    n_rows = int(np.ceil(len(factors) / float(n_cols)))

    for (n_wires, side) in SIZES:
        if n_wires not in loaded:
            continue
        results = loaded[n_wires]
        n_nets = len(results)
        summary[n_wires] = {}
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.1 * n_rows),
                                 sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        print("\n=== %d fili, %d reti" % (n_wires, n_nets))
        print("  V*   | reti con valanghe | valanghe tutte -> senza | "
              "tolta: mediana  10-90%% | quota eventi (mediana) | piu' grande = cammino"
              .replace("%%", "%"))
        for i, f in enumerate(factors):
            ax = axes[i // n_cols][i % n_cols]
            all_list = []
            without_list = []
            removed_list = []
            share_list = []
            same_count = 0
            has_count = 0
            for r in results:
                one = split_network(r[2], f)
                if one is None:
                    continue
                has_count = has_count + 1
                all_list.append(one[0])
                without_list.append(one[1])
                if one[2] > 0:
                    removed_list.append(one[2])
                    share_list.append(one[3])
                if one[4]:
                    same_count = same_count + 1
            n_all = 0
            if len(all_list) > 0:
                pooled_all = np.concatenate(all_list)
                pooled_without = np.concatenate(without_list)
                n_all = pooled_all.shape[0]
            stats = {}
            stats["frac_with_avalanches"] = has_count / float(n_nets)
            stats["n_removed"] = len(removed_list)
            stats["median_removed"] = np.nan
            stats["median_share"] = np.nan
            stats["frac_same"] = np.nan
            if len(removed_list) > 0:
                stats["median_removed"] = float(np.median(removed_list))
                stats["median_share"] = float(np.median(share_list))
                stats["frac_same"] = same_count / float(has_count)
            summary[n_wires][f] = stats

            if n_all >= MIN_AVALANCHES and len(removed_list) == 0:
                # there are avalanches but no network has the ON path (spanning
                # mode): there is no avalanche to remove, so the statistics
                # line cannot be computed. All the avalanches are drawn anyway.
                print("  %.2f | %4d su %4d      | %6d valanghe, nessuna rete con il cammino: "
                      "niente da togliere" % (f, has_count, n_nets, n_all))
                x, p = log_binned(pooled_all)
                ax.plot(x, p, "-o", color=COLORS[1], linewidth=1.8, markersize=3.5,
                        label="tutte")
                ax.legend(fontsize=7, frameon=False, loc="lower left")
            elif n_all >= MIN_AVALANCHES:
                rem = np.array(removed_list)
                print("  %.2f | %4d su %4d      | %6d -> %6d        | %5.0f  %3.0f-%3.0f      "
                      "| %5.0f%%                 | %4.0f%%" % (
                          f, has_count, n_nets, n_all, pooled_without.shape[0],
                          np.median(rem), np.percentile(rem, 10), np.percentile(rem, 90),
                          100.0 * stats["median_share"], 100.0 * stats["frac_same"]))
                x, p = log_binned(pooled_all)
                ax.plot(x, p, "-o", color=COLORS[1], linewidth=1.8, markersize=3.5,
                        label="tutte")
                if pooled_without.shape[0] > 0:
                    x2, p2 = log_binned(pooled_without)
                    ax.plot(x2, p2, "--s", color="#c0392b", linewidth=1.4, markersize=3,
                            label="senza la tolta")
                ax.legend(fontsize=7, frameon=False, loc="lower left")
            else:
                print("  %.2f | %4d su %4d      | %6d valanghe: troppo poche, non disegno"
                      % (f, has_count, n_nets, n_all))
                ax.text(0.5, 0.5, "troppo poche valanghe\n(%d reti su %d)" % (has_count, n_nets),
                        ha="center", va="center", transform=ax.transAxes, fontsize=8,
                        color="0.4")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("V* = %.2f" % f, fontsize=9)
            ax.grid(True, which="major", color="0.9", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        # empty panels at the bottom (if the V* values do not fill the grid)
        for j in range(len(factors), n_rows * n_cols):
            axes[j // n_cols][j % n_cols].axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("dimensione S")
        for row in axes:
            row[0].set_ylabel("P(S)")
        fig.suptitle("K = %d, %d fili, %d reti, modo: %s" % (K, n_wires, n_nets, MODE),
                     fontsize=10)
        fig.tight_layout()
        name = "without_%s_K%d_w%d.png" % (MODE, K, n_wires)
        fig.savefig(os.path.join(IMAGES, name), dpi=140)
        plt.close(fig)
        print("salvata images/%s" % name)

    # summary figure: three panels against V*
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    titles = ["quota di eventi nella valanga tolta (mediana)",
              "dimensione mediana della valanga tolta",
              "frazione di reti con almeno una valanga"]
    keys = ["median_share", "median_removed", "frac_with_avalanches"]
    for (n_wires, side), color in zip(SIZES, COLORS):
        if n_wires not in summary:
            continue
        xs = []
        ys = [[], [], []]
        for f in factors:
            st = summary[n_wires][f]
            # a point is drawn only if at least 20 networks have a removed avalanche
            # (otherwise the median is too unstable); the fraction of networks is
            # always drawn
            xs.append(f)
            for k in range(3):
                value = st[keys[k]]
                if k < 2:
                    if st["n_removed"] < 20:
                        value = np.nan
                    # below V* = 1 almost no network forms a path: the "largest"
                    # avalanche is an isolated event of 1-3 junctions, not a
                    # lightning. A 100% share there would be an artefact: not drawn.
                    if f < 1.0:
                        value = np.nan
                ys[k].append(value)
        for k in range(3):
            axes[k].plot(xs, ys[k], "-o", color=color, linewidth=2.0, markersize=4,
                         label="%d fili" % n_wires)
    for k in range(3):
        axes[k].set_title(titles[k], fontsize=9)
        axes[k].set_xlabel("V*")
        axes[k].grid(True, color="0.9", linewidth=0.6)
        axes[k].spines["top"].set_visible(False)
        axes[k].spines["right"].set_visible(False)
    axes[1].set_yscale("log")
    axes[0].set_ylim(0, 1)
    axes[2].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("K = %d, modo: %s (primi due pannelli: solo V* >= 1, dove si forma il cammino)"
                 % (K, MODE), fontsize=9)
    fig.tight_layout()
    name = "summary_%s_K%d.png" % (MODE, K)
    fig.savefig(os.path.join(IMAGES, name), dpi=140)
    print("\nsalvata images/%s" % name)
