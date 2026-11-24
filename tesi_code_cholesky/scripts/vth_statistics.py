# vth_statistics.py
# Statistics of the activation threshold Vth of the networks, for every size
# (number of wires) and every K, read from the cache files made by
# run_conductance_curves.py (no simulation is run here).
#
# Output:
#   1. a table printed on the screen and saved as scripts/vth_table.csv, with
#      for every (K, wires): number of networks, number of networks without a
#      threshold (nan), mean, standard deviation, standard error of the mean,
#      minimum, median, maximum, and the ratio std / mean (relative scatter)
#   2. one picture scripts/images/vth_histograms.png: one row per K, one
#      column per size, each panel is the histogram of Vth over the networks
#
# Numerical example: 4 networks with Vth = 0.20, 0.22, 0.18, 0.20 V:
#   mean = 0.20, standard deviation = 0.0163 (it divides by n - 1 = 3),
#   standard error = 0.0163 / sqrt(4) = 0.0082, std / mean = 0.082.
#
# How to run (from the main folder):  python scripts/vth_statistics.py
# Environment variables: KS (list of K separated by commas, default "1,10"),
# RATIO (default 1000).

import os
import sys
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RATIO = int(os.environ.get("RATIO", "1000"))
KS_TEXT = os.environ.get("KS", "1,10")
KS = []
for piece in KS_TEXT.split(","):
    KS.append(int(piece))
IMAGES = os.path.join(HERE, "images")


def find_caches(k_value):
    # Returns {n_wires: list of Vth, one number per network}: for every size
    # the cache file with the most networks is used.
    pattern = os.path.join(HERE, "conductance_r%d_K%d_w*_L*_f*.pkl" % (RATIO, k_value))
    best_n = {}
    best_vth = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        n_wires = int(name.split("_w")[1].split("_")[0])
        with open(path, "rb") as fh:
            content = pickle.load(fh)
        vth_list = []
        for seed in sorted(content["nets"].keys()):
            vth_list.append(float(content["nets"][seed]["vth"]))
        n_nets = len(vth_list)
        keep = False
        if n_wires not in best_n:
            keep = True
        else:
            if n_nets > best_n[n_wires]:
                keep = True
        if keep:
            best_n[n_wires] = n_nets
            best_vth[n_wires] = vth_list
    return best_vth


def describe(vth_list):
    # Statistics of a list of thresholds; the nan values (networks where no
    # path forms up to the maximum voltage) are counted and then left out.
    arr = np.asarray(vth_list, dtype=float)
    n_total = arr.shape[0]
    good = arr[~np.isnan(arr)]
    n_good = good.shape[0]
    n_nan = n_total - n_good
    out = {"n_total": n_total, "n_nan": n_nan, "n_good": n_good}
    if n_good == 0:
        out["mean"] = float("nan")
        out["std"] = float("nan")
        out["sem"] = float("nan")
        out["min"] = float("nan")
        out["median"] = float("nan")
        out["max"] = float("nan")
        out["rel"] = float("nan")
        return out
    out["mean"] = float(np.mean(good))
    out["min"] = float(np.min(good))
    out["median"] = float(np.median(good))
    out["max"] = float(np.max(good))
    if n_good >= 2:
        out["std"] = float(np.std(good, ddof=1))
        out["sem"] = out["std"] / float(np.sqrt(n_good))
    else:
        out["std"] = float("nan")
        out["sem"] = float("nan")
    out["rel"] = out["std"] / out["mean"]
    return out


if __name__ == "__main__":
    data = {}                              # data[K][n_wires] = list of Vth
    for k_value in KS:
        found = find_caches(k_value)
        if len(found) > 0:
            data[k_value] = found
    if len(data) == 0:
        print("No cache found. Run run_conductance_curves.py first.")
        sys.exit(1)

    # ---------------- table ----------------
    header = ["K", "wires", "networks", "no_Vth", "mean_V", "std_V", "sem_V",
              "min_V", "median_V", "max_V", "std/mean"]
    rows = []
    for k_value in sorted(data.keys()):
        for n_wires in sorted(data[k_value].keys()):
            st = describe(data[k_value][n_wires])
            rows.append([k_value, n_wires, st["n_total"], st["n_nan"],
                         st["mean"], st["std"], st["sem"], st["min"],
                         st["median"], st["max"], st["rel"]])
    print("")
    print("%3s %6s %9s %7s %9s %9s %9s %9s %9s %9s %9s" % tuple(header))
    for r in rows:
        print("%3d %6d %9d %7d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.3f" % tuple(r))
    csv_path = os.path.join(HERE, "vth_table.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join([str(x) for x in r]) + "\n")
    print("")
    print("table saved in", csv_path)

    # ---------------- histograms ----------------
    os.makedirs(IMAGES, exist_ok=True)
    k_list = sorted(data.keys())
    sizes = set()
    for k_value in k_list:
        for n_wires in data[k_value].keys():
            sizes.add(n_wires)
    sizes = sorted(sizes)
    fig, axes = plt.subplots(len(k_list), len(sizes),
                             figsize=(4.2 * len(sizes), 3.2 * len(k_list)),
                             squeeze=False)
    for i, k_value in enumerate(k_list):
        for j, n_wires in enumerate(sizes):
            ax = axes[i][j]
            if n_wires not in data[k_value]:
                ax.axis("off")
                continue
            arr = np.asarray(data[k_value][n_wires], dtype=float)
            good = arr[~np.isnan(arr)]
            if good.shape[0] == 0:
                ax.axis("off")
                continue
            ax.hist(good, bins=20, color="#2a788e", edgecolor="white")
            mean = float(np.mean(good))
            ax.axvline(mean, color="black", linewidth=1.2, linestyle="--")
            ax.set_title("K = %d, %d wires (%d networks)\nmean = %.3f V"
                         % (k_value, n_wires, good.shape[0], mean), fontsize=9)
            ax.set_xlabel("Vth (V)")
            ax.set_ylabel("number of networks")
            ax.grid(alpha=0.25)
    fig.suptitle("Distribution of the threshold Vth over the networks "
                 "(dashed line = mean)", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(IMAGES, "vth_histograms.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("histograms saved in", out_path)
