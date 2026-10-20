# plot_model_overview.py
# ONE figure with three panels that explain the model (saved as
# images/model_overview.png, in the folder of this script).
# Gon/Goff = 1000, default network (100 wires, 30 x 30 um), point electrodes.
#
#   (a) One network after the loop at V* = 1.05: every junction is coloured
#       by its conductance level (0 = OFF, K = fully ON). Source and drain are
#       the thick black wires. The orange junctions that touch each other
#       form the conducting path.
#   (b) Same network: total current after each switch, versus the clock.
#       The dashed line marks the moment when the ON junctions (level K)
#       connect source and drain for the first time.
#   (c) Avalanche size distribution P(S) pooled over many networks, at three
#       values of V* = V / Vth (Vth is computed for each network). The dashed
#       line has slope -1.95 (exponent of Hochstetter et al., DOI
#       10.1038/s41467-021-24260-z): it is only a reference to compare the
#       slope by eye, NOT a fit of these data.
#
# Run: python scripts/plot_model_overview.py
# Optional: environment variable NNET = number of networks for panel (c)
# (default 40). PowerShell:  $env:NNET = 10
# Panel (c) needs the threshold of every network, so it is the slow part
# (about a few minutes for 40 networks, not timed on every computer). The
# results are saved in a cache_*.pkl file: delete it to repeat the run.

import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The library files (nanowire_network.py, nw_solver.py, ...) are in the main
# folder, one level up: add it to the search path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nanowire_network as nn
from nw_solver import solve_node_voltages, total_current
from nw_loop_levels import run_loop_levels
from nw_avalanches import find_avalanches
from nw_threshold import find_threshold, on_path_exists

# ---------------------------------------------------------------- parameters
V_SET = 0.01                # switching threshold of one junction (V)
LAM_CRIT = 0.01             # Lambda needed to reach the top level (V s)
G_ON = 1.0                  # units of G0
G_OFF = G_ON / 1000.0       # ratio Gon/Goff = 1000
K = 10                      # number of level steps of one junction

SEED_EXAMPLE = 7            # network drawn in panels (a) and (b)
V_STAR_EXAMPLE = 1.05       # V* used for panels (a) and (b)

V_STAR_LIST = [0.7, 1.0, 1.8]            # V* values of panel (c)
V_COLORS = ["#9ac0e6", "#d9622b", "#143d6b"]
N_NETWORKS = int(os.environ.get("NNET", "40"))

COLOR_WIRE = "#b5b5b5"
COLOR_OFF = "#3b7bbf"
COLOR_ON = "#d9622b"

FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES_FOLDER = os.path.join(FOLDER, "images")


# ----------------------------------------------------------- helper functions
def make_network(seed):
    # Network + point electrodes (source = highest wire centre,
    # drain = lowest wire centre).
    net = nn.generate_network(seed=seed)
    s, d = nn.point_electrodes(net)
    return net, s, d


def current_after_each_event(net, s, d, v_ext, res):
    # The loop returns the list of switches but not the current. Here we
    # "replay" the switches one by one: start with every junction at level
    # 0, apply the first switch, solve Kirchhoff, read the total current,
    # apply the second switch, and so on.
    # Returns (current before any switch, array: current after each switch).
    m = net.junctions.shape[0]
    n_nodes = net.n_wires()

    # conductance of every level k = 0..K:  G_k = Goff * (Gon/Goff)^(k/K)
    level_g = np.zeros(K + 1, dtype=float)
    for k in range(K + 1):
        level_g[k] = G_OFF * (G_ON / G_OFF) ** (float(k) / K)

    level = np.zeros(m, dtype=int)
    voltage = solve_node_voltages(n_nodes, net.junctions, level_g[level],
                                  s, d, v_ext)
    current_start = total_current(n_nodes, net.junctions, level_g[level],
                                  voltage, s)

    currents = []
    for j in res["event_junction"]:
        level[j] = level[j] + 1             # this junction goes up one level
        voltage = solve_node_voltages(n_nodes, net.junctions, level_g[level],
                                      s, d, v_ext)
        currents.append(total_current(n_nodes, net.junctions, level_g[level],
                                      voltage, s))
    return current_start, np.array(currents, dtype=float)


def first_path_event(net, s, d, res):
    # Index of the first switch after which the junctions at level K (ON)
    # connect a source wire to a drain wire. Returns -1 if it never happens.
    # Replay again, but now we only follow the levels (no Kirchhoff).
    m = net.junctions.shape[0]
    level = np.zeros(m, dtype=int)
    answer = -1
    k = 0
    while k < len(res["event_junction"]) and answer < 0:
        j = res["event_junction"][k]
        level[j] = level[j] + 1
        # junctions that are fully ON right now
        on_list = np.nonzero(level == K)[0]
        if on_path_exists(net.n_wires(), net.junctions, on_list, s, d):
            answer = k
        k = k + 1
    return answer


def one_network(seed):
    # Worker for panel (c): threshold of this network, then one loop run at
    # each V*, counting one event per growth episode. Returns
    # (seed, Vth, {V*: array of avalanche sizes}).
    net, s, d = make_network(seed)
    vth = find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
    out = {}
    if np.isnan(vth):
        return seed, vth, out
    for f in V_STAR_LIST:
        res = run_loop_levels(net.n_wires(), net.junctions, s, d, f * vth,
                              V_SET, LAM_CRIT, G_OFF, G_ON, K)
        times = res["event_time"][res["event_new_episode"]]
        av = find_avalanches(times)
        out[f] = np.asarray(av["S"], dtype=int)
    return seed, vth, out


def get_results():
    # Results for all networks (read from the cache file if it exists).
    cache = os.path.join(FOLDER, "cache_overview_K%d_%d.pkl" % (K, N_NETWORKS))
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    with Pool(2) as pool:
        results = pool.map(one_network, list(range(N_NETWORKS)), chunksize=2)
    with open(cache, "wb") as fh:
        pickle.dump(results, fh)
    return results


# ------------------------------------------------------------------- panels
def panel_network(ax, net, s, d, res, vth):
    # (a) wires in grey, junctions coloured by level (blue = 0, orange = K).
    cmap = LinearSegmentedColormap.from_list("levels", [COLOR_OFF, COLOR_ON])
    for i in range(net.n_wires()):
        ax.plot([net.p1[i, 0], net.p2[i, 0]], [net.p1[i, 1], net.p2[i, 1]],
                color=COLOR_WIRE, linewidth=0.8, zorder=1)
    level = res["level"]
    # draw the low levels first, so the orange junctions stay on top
    order = np.argsort(level)
    points = ax.scatter(net.junction_xy[order, 0], net.junction_xy[order, 1],
                        c=level[order], cmap=cmap, vmin=0, vmax=K,
                        s=14 + 4 * level[order], edgecolor="white",
                        linewidth=0.4, zorder=3)
    for label, wires in (("S", s), ("D", d)):
        for i in wires:
            ax.plot([net.p1[i, 0], net.p2[i, 0]], [net.p1[i, 1], net.p2[i, 1]],
                    color="black", linewidth=2.4, zorder=4)
            ax.text(net.centers[i, 0] + 1.0, net.centers[i, 1] + 1.0, label,
                    fontsize=12, fontweight="bold", zorder=5)
    ax.plot([0, net.side, net.side, 0, 0], [0, 0, net.side, net.side, 0],
            color="black", linewidth=0.5, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x (micrometres)")
    ax.set_ylabel("y (micrometres)")
    n_top = int(res["is_on"].sum())
    ax.set_title("(a) One network at V* = " + str(V_STAR_EXAMPLE) +
                 " (V = %.3f V)\n%d of %d junctions fully ON" %
                 (V_STAR_EXAMPLE * vth, n_top, net.n_junctions()), fontsize=10)
    bar = plt.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("conductance level (0 = OFF, " + str(K) + " = ON)")


def panel_current(ax, net, s, d, res, v_ext):
    # (b) current after each switch versus the clock (log-log).
    i_start, currents = current_after_each_event(net, s, d, v_ext, res)
    times = res["event_time"]
    k_path = first_path_event(net, s, d, res)
    # one point before the first switch, drawn a bit earlier on the time axis
    x = np.concatenate([[times[0] / 2.0], times])
    y = np.concatenate([[i_start], currents])
    ax.step(x, y, where="post", color=V_COLORS[2], linewidth=1.6)
    if k_path >= 0:
        t_path = times[k_path]
        ax.axvline(t_path, color=COLOR_ON, linestyle="--", linewidth=1.2)
        ax.text(t_path, y.min() * 1.5, " first ON path\n source-drain",
                color=COLOR_ON, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clock (s, log scale)")
    ax.set_ylabel("total current (G0 x V, log scale)")
    ax.set_title("(b) Same network: total current after each switch\n" +
                 "%d switches; current x %.0f from start to end" %
                 (len(times), y[-1] / y[0]), fontsize=10)
    ax.grid(alpha=0.25, which="both")


def panel_distribution(ax, results):
    # (c) P(S) pooled over the networks, one colour per V*.
    good = [r for r in results if not np.isnan(r[1])]
    for k in range(len(V_STAR_LIST)):
        f = V_STAR_LIST[k]
        arrays = []
        for r in good:
            if r[2][f].shape[0] > 0:
                arrays.append(r[2][f])
        if len(arrays) == 0:
            continue
        pooled = np.concatenate(arrays)
        values, counts = np.unique(pooled, return_counts=True)
        ax.loglog(values, counts / counts.sum(), "o", markersize=4,
                  color=V_COLORS[k],
                  label="V* = %.1f (%d avalanches)" % (f, pooled.shape[0]))
    # reference slope; the factor 0.5 only moves the line next to the data
    ss = np.array([1.0, 100.0])
    ax.loglog(ss, 0.5 * ss ** (-1.95), "k--", linewidth=1,
              label="slope -1.95 (Hochstetter)")
    ax.set_xlabel("avalanche size S (number of growth episodes)")
    ax.set_ylabel("P(S)")
    ax.set_title("(c) Avalanche sizes, %d networks pooled\n" % len(good) +
                 "(V* = V / Vth of each network)", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    # Network of panels (a) and (b): threshold, then one run at V* = 1.05.
    net, s, d = make_network(SEED_EXAMPLE)
    vth = find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
    if np.isnan(vth):
        raise SystemExit("No path forms in the example network (seed " +
                         str(SEED_EXAMPLE) + "): choose another SEED_EXAMPLE.")
    v_ext = V_STAR_EXAMPLE * vth
    res = run_loop_levels(net.n_wires(), net.junctions, s, d, v_ext,
                          V_SET, LAM_CRIT, G_OFF, G_ON, K)
    print("Example network (seed %d): %d wires, %d junctions, Vth = %.4f V "
          "(%.1f x Vset)" % (SEED_EXAMPLE, net.n_wires(), net.n_junctions(),
                             vth, vth / V_SET), flush=True)

    # Panel (c) data: many networks (slow part, cached).
    results = get_results()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    panel_network(axes[0], net, s, d, res, vth)
    panel_current(axes[1], net, s, d, res, v_ext)
    panel_distribution(axes[2], results)
    fig.suptitle("Multi-level threshold model (K = " + str(K) +
                 ", Gon/Goff = 1000), 100 wires, point electrodes",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(IMAGES_FOLDER, "model_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)
