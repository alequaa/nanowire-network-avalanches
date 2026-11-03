# plot_lightning_hist.py
# Histogram of the size S of the LARGEST avalanche of every network (the
# "lightning"), on a LINEAR scale, for the three sizes, at chosen V*.
# It serves to see whether the shape is roughly Gaussian and moves with the
# network size. It also prints mean, standard deviation, median and skewness.
#
# Reads the cache_nospan_*.pkl files (no simulation).
# How to run (from the main folder, in cmd):
#   set K=1
#   python scripts/plot_lightning_hist.py
# Optional variables: K (default 1), RATIO (default 1000),
# VSTARS (comma-separated list, default 1.2,1.4,1.8,2.5).
# Image produced: scripts/images/lightning_hist_K<K>.png
#
# Caution: the "largest avalanche" of a network is a lightning only if the
# network has the ON path. Below V* = 1 almost no network has it, which is why
# the default V* values start at 1.2.
#
# Numerical example of skewness: if the mean is 42 and the median 36, the
# distribution has a tail on the right (positive skewness): it is not
# symmetric, so it is not a Gaussian.

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATIO = float(os.environ.get("RATIO", "1000"))
K = int(os.environ.get("K", "1"))
VSTARS = [float(v) for v in os.environ.get("VSTARS", "1.2,1.4,1.8,2.5").split(",")]
FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(FOLDER, "images")
SIZES = [(100, 30.0), (400, 60.0), (900, 90.0)]
COLORS = ["#9ec5ee", "#3f7fc4", "#0f3d75"]


def find_cache(n_wires, side):
    # Looks for the cache file with the most networks for this size.
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


def skewness(x):
    # Skewness: mean of ((x - mean)/std)^3. Zero for a symmetric distribution.
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    fig, axes = plt.subplots(1, len(VSTARS), figsize=(3.4 * len(VSTARS), 3.6))
    axes = np.atleast_1d(axes)
    print("K = %d" % K)
    print("  fili   V*  | reti | media  std  mediana  skewness")
    for (n_wires, side), color in zip(SIZES, COLORS):
        fname = find_cache(n_wires, side)
        if fname is None:
            print("  %4d  nessuna cache, salto" % n_wires)
            continue
        with open(os.path.join(FOLDER, fname), "rb") as fh:
            results = pickle.load(fh)
        for i, f in enumerate(VSTARS):
            largest = []
            for r in results:
                out = r[2]
                if f not in out:
                    continue
                # only networks that have the ON path (otherwise it is not a lightning)
                if not out[f][2]:
                    continue
                largest.append(int(np.max(out[f][0])))
            x = np.array(largest, dtype=float)
            if x.shape[0] < 20:
                print("  %4d  %.2f | %4d | troppo poche reti" % (n_wires, f, x.shape[0]))
                continue
            print("  %4d  %.2f | %4d | %5.1f %5.1f %6.0f   %6.2f" % (
                n_wires, f, x.shape[0], x.mean(), x.std(), np.median(x), skewness(x)))
            axes[i].hist(x, bins=25, density=True, histtype="step", linewidth=2.0,
                         color=color, label="%d fili" % n_wires)
    for i, f in enumerate(VSTARS):
        axes[i].set_title("V* = %.2f" % f, fontsize=9)
        axes[i].set_xlabel("S della valanga piu' grande")
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
    axes[0].set_ylabel("densita'")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("K = %d: dimensione della valanga piu' grande (solo reti con cammino acceso)" % K,
                 fontsize=9)
    fig.tight_layout()
    name = "lightning_hist_K%d.png" % K
    fig.savefig(os.path.join(IMAGES, name), dpi=140)
    print("salvata images/%s" % name)
