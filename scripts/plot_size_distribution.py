# plot_size_distribution.py
# Distribution of avalanche sizes P(S) for the three system sizes, at
# V* = 1.05 and V* = 1.8 (log-log). It reads the cache files made by
# run_average_size.py: no simulation, it takes a few seconds.
#
# All avalanches of all networks of one size are pooled. S is cut into
# logarithmic bins (width 1 for small S, wider for large S); in each bin
#   P(S) = (avalanches in the bin) / (bin width) / (total avalanches)
# Example: 3 avalanches in the bin S = 10..13 (width 4), 300 avalanches in
# total -> P = 3 / 4 / 300 = 0.0025.
# Bins with no avalanche are not drawn.
#
# Run (from the main folder):
#   python scripts/plot_size_distribution.py
# Output: images/size_distribution.png (inside scripts/).
# Optional environment variables: K (default 10), RATIO (default 1000).

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATIO = float(os.environ.get("RATIO", "1000"))
K = int(os.environ.get("K", "10"))
FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(FOLDER, "images")
SIZES = [(100, 30.0), (400, 60.0), (900, 90.0)]    # (wires, side in um)
FACTORS = [1.05, 1.8]
# one blue hue, light -> dark with the system size (magnitude, not identity)
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


def log_binned(sizes):
    # Returns (bin centres, P(S)) for the pooled avalanche sizes.
    s_max = int(sizes.max())
    # about 24 geometric edges; rounding to integers gives width 1 at small S
    edges = np.unique(np.round(np.logspace(0.0, np.log10(s_max + 1.0), 25)))
    edges = edges.astype(int)
    counts, _ = np.histogram(sizes, bins=edges)
    widths = np.diff(edges)
    density = counts / widths / sizes.shape[0]
    centres = np.sqrt(edges[:-1] * (edges[1:] - 1.0 + 1e-9))
    keep = counts > 0
    return centres[keep], density[keep]


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, f in zip(axes, FACTORS):
        for (n_wires, side), color in zip(SIZES, COLORS):
            fname = find_cache(n_wires, side)
            if fname is None:
                print("no cache file for %d wires (skipped)" % n_wires)
                continue
            with open(os.path.join(FOLDER, fname), "rb") as fh:
                results = pickle.load(fh)
            arrays = []
            for r in results:
                if f in r[2]:
                    arrays.append(r[2][f])
            sizes = np.concatenate(arrays)
            x, p = log_binned(sizes)
            label = "%d wires (%d networks, %d avalanches)" % (
                n_wires, len(results), sizes.shape[0])
            ax.plot(x, p, "-o", color=color, linewidth=2.0, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("V* = %.2f" % f)
        ax.set_xlabel("avalanche size S")
        ax.grid(True, which="major", color="0.88", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    axes[0].set_ylabel("P(S)")
    fig.suptitle("Avalanche size distribution, K = %d, Gon/Goff = %g, events = growth episodes"
                 % (K, RATIO), fontsize=11)
    fig.tight_layout()
    out = os.path.join(IMAGES, "size_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)
