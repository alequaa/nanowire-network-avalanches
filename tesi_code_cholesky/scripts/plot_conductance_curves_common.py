# plot_conductance_curves.py
# THIS COPY of plot_conductance_curves.py uses by DEFAULT the mode "common"
# (files conductance_common_*.pkl, images ending with _common).
# Draws the curves of the conductance of the network from the cache files made
# by run_conductance_curves.py (no simulation is run here).
#
# Figure 1 (one per size): conductance G of ONE network versus time, one curve
#   for every V* (log-log). The dot marks the switch that closes the first
#   path of ON junctions. G is in units of Gon.
# Figure 2 (one per K, all sizes together): versus V*, one curve per size:
#   (a) fraction of networks where a path of ON junctions forms
#   (b) jump of G in the single switch that closes the path
#   (c) jump of G over the whole avalanche that contains that switch
#   (d) the largest jump of G over one avalanche (any avalanche)
#   (e) final conductance G_end
# In (b), (c), (d) the value is the mean over ALL the networks (a network
# without a path counts as 0 in (b) and (c)); the bars are the standard error
# of the mean (standard deviation / sqrt(number of networks)).
# Which definition of the jump is the right one has to be decided; all of them
# are drawn.
#
# Numerical example of the standard error: 4 networks with jumps 0.1, 0.2,
# 0.3, 0.4: mean 0.25, standard deviation 0.129, error 0.129 / 2 = 0.065.
#
# How to run (from the main folder):  python scripts/plot_conductance_curves.py
# Environment variables: K (default 10), RATIO (1000), NET_SEED (network drawn
# in figure 1, default 0), VTH_MODE ("own" default, or "common": reads the
# files conductance_common_*.pkl, where V = V* x Vref(L) with the SAME Vref for
# all the networks of one size). The images are saved in scripts/images; with
# VTH_MODE=common their names end with _common.

import os
import sys
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
K = int(os.environ.get("K", "10"))
RATIO = int(os.environ.get("RATIO", "1000"))
NET_SEED = int(os.environ.get("NET_SEED", "0"))
VTH_MODE = os.environ.get("VTH_MODE", "common")
IMAGES = os.path.join(HERE, "images")
PREFIX = "conductance"
SUFFIX = ""
V_LABEL = "V* = V / Vth (Vth of each network)"
V_NAME = "Vth"
if VTH_MODE == "common":
    PREFIX = "conductance_common"
    SUFFIX = "_common"
    V_LABEL = "V* = V / Vref(L) (same Vref for all the networks of one size)"
    V_NAME = "Vref"


def find_caches():
    # Returns {n_wires: (path, content)}: for every size the cache with the
    # most networks.
    pattern = os.path.join(HERE, "%s_r%d_K%d_w*_L*_f*.pkl" % (PREFIX, RATIO, K))
    best = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        n_wires = int(name.split("_w")[1].split("_")[0])
        with open(path, "rb") as fh:
            content = pickle.load(fh)
        n_nets = len(content["nets"])
        if n_wires not in best:
            best[n_wires] = (path, content)
        else:
            if n_nets > len(best[n_wires][1]["nets"]):
                best[n_wires] = (path, content)
    return best


def figure_time(n_wires, content):
    # Figure 1: G(t) of one network at all the V*.
    nets = content["nets"]
    factors = content["factors"]
    seed = NET_SEED
    if seed not in nets:
        seed = sorted(nets.keys())[0]
    if np.isnan(nets[seed]["vth"]):
        print("network", seed, "has no threshold: figure 1 skipped for", n_wires, "wires")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(factors):
        run = nets[seed]["runs"][f]
        color = cmap(float(i) / max(len(factors) - 1, 1))
        # state 0 is at time 0, which cannot be drawn on a log axis: start from state 1
        ax.plot(run["state_time"][1:], run["state_conductance"][1:],
                color=color, linewidth=1.3, label="V* = %.2f" % f)
        k = run["path_event"]
        if k >= 0:
            ax.plot(run["state_time"][k + 1], run["state_conductance"][k + 1],
                    "o", color=color, markersize=5, markeredgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time (s, log scale)")
    ax.set_ylabel("network conductance G (units of Gon, log scale)")
    ax.set_title("%d wires, K = %d, network %d (%s = %.3f V)\n"
                 "dot = switch that closes the first ON path"
                 % (n_wires, K, seed, V_NAME, nets[seed]["vth"]))
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(IMAGES, "conductance_time_K%d_w%d%s.png" % (K, n_wires, SUFFIX))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def mean_and_error(values):
    # Mean and standard error of a list of numbers (nan if fewer than 2).
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    err = float("nan")
    if n >= 2:
        err = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mean, err


def figure_jump(caches):
    # Figure 2: quantities versus V*, one curve per size.
    titles = ["(a) fraction of networks with an ON path",
              "(b) jump of G in the single switch that closes the path",
              "(c) jump of G over the avalanche that closes the path",
              "(d) largest jump of G over one avalanche",
              "(e) final conductance G_end"]
    keys = [None, "jump_path", "jump_path_aval", "dg_aval_max", "G_end"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes_flat = axes.flatten()
    cmap = plt.get_cmap("viridis")
    sizes = sorted(caches.keys())
    for si, n_wires in enumerate(sizes):
        content = caches[n_wires][1]
        factors = content["factors"]
        color = cmap(float(si) / max(len(sizes) - 1, 1) * 0.85)
        good = []
        for seed in content["nets"]:
            if not np.isnan(content["nets"][seed]["vth"]):
                good.append(seed)
        label = "%d wires (%d networks)" % (n_wires, len(good))
        for p in range(5):
            xs = []
            ys = []
            es = []
            for f in factors:
                vals = []
                for seed in good:
                    run = content["nets"][seed]["runs"][f]
                    if keys[p] is None:
                        if run["path_event"] >= 0:
                            vals.append(1.0)
                        else:
                            vals.append(0.0)
                    else:
                        vals.append(run[keys[p]])
                m, e = mean_and_error(vals)
                xs.append(f)
                ys.append(m)
                es.append(e)
            axes_flat[p].errorbar(xs, ys, yerr=es, color=color, marker="o",
                                  markersize=4, linewidth=1.2, capsize=2,
                                  label=label)
    for p in range(5):
        axes_flat[p].set_title(titles[p], fontsize=9)
        axes_flat[p].set_xlabel("V* = V / %s" % V_NAME)
        axes_flat[p].grid(alpha=0.25)
    axes_flat[0].set_ylabel("fraction")
    axes_flat[1].set_ylabel("mean jump (units of Gon)")
    axes_flat[2].set_ylabel("mean jump (units of Gon)")
    axes_flat[3].set_ylabel("mean jump (units of Gon)")
    axes_flat[4].set_ylabel("mean G_end (units of Gon)")
    axes_flat[0].legend(fontsize=8, frameon=False)
    axes_flat[5].axis("off")
    fig.suptitle("Jump of the network conductance versus V*, K = %d\n%s" % (K, V_LABEL), fontsize=10)
    fig.tight_layout()
    out = os.path.join(IMAGES, "conductance_jump_K%d%s.png" % (K, SUFFIX))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    caches = find_caches()
    if len(caches) == 0:
        print("No cache found for K = %d. Run run_conductance_curves.py first." % K)
        sys.exit(1)
    for n_wires in sorted(caches.keys()):
        figure_time(n_wires, caches[n_wires][1])
    figure_jump(caches)
