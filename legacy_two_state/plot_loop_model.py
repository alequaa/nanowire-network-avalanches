# plot_loop_model.py
# Three figures for the fixed-voltage loop (saved in the images folder
# next to this script). Run: python plot_loop_model.py
#   1. network_after_loop.png : ON junctions (orange) and OFF junctions (blue)
#   2. switches_vs_time.png   : number of ON junctions vs clock
#   3. current_vs_switches.png: total current after each switch

# The shared library files (nanowire_network.py, nw_solver.py, ...) are in the
# main folder, one level up: add it to the search path.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nanowire_network as nn
from nw_loop_model import run_loop

# Parameters of Hochstetter et al. (Methods); ratio 1000 = Roff/Rf in the paper
V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0

SEED = 7                      # same network as example_network.png
V_FOR_MAP = 0.15              # voltage used for figure 1
V_LIST = [0.08, 0.15, 0.3]    # voltages compared in figures 2 and 3

# Colours: ON = orange, OFF = blue, wires = grey. Three blue shades
# (light to dark = low to high voltage) for the voltage comparison.
COLOR_ON = "#d9622b"
COLOR_OFF = "#3b7bbf"
COLOR_WIRE = "#b5b5b5"
V_COLORS = ["#9ac0e6", "#3b7bbf", "#143d6b"]

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def run_one(net, v_ext):
    # One run of the loop with point electrodes.
    s, d = nn.point_electrodes(net)
    res = run_loop(net.n_wires(), net.junctions, s, d, v_ext,
                   V_SET, LAM_CRIT, G_OFF, G_ON)
    return res, s, d


def figure_network(net):
    res, s, d = run_one(net, V_FOR_MAP)
    n_on = int(res["is_on"].sum())

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for i in range(net.n_wires()):
        ax.plot([net.p1[i, 0], net.p2[i, 0]], [net.p1[i, 1], net.p2[i, 1]],
                color=COLOR_WIRE, linewidth=0.8, zorder=1)
    on = res["is_on"]
    off = ~on
    ax.scatter(net.junction_xy[off, 0], net.junction_xy[off, 1], s=14,
               color=COLOR_OFF, zorder=2, label="OFF junction")
    ax.scatter(net.junction_xy[on, 0], net.junction_xy[on, 1], s=34,
               color=COLOR_ON, edgecolor="white", linewidth=0.6, zorder=3,
               label="ON junction")
    # Source (S) and drain (D): thick black wire plus a letter.
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
    ax.set_title("After the loop at V = " + str(V_FOR_MAP) + " V: " +
                 str(n_on) + " of " + str(net.n_junctions()) + " junctions ON")
    ax.legend(loc="lower right", frameon=True)
    path = os.path.join(IMAGES_FOLDER, "network_after_loop.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_switches_vs_time(net):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in range(len(V_LIST)):
        res, s, d = run_one(net, V_LIST[k])
        et = res["event_time"]
        if et.shape[0] == 0:
            continue
        counts = np.arange(1, et.shape[0] + 1)
        ax.step(et, counts, where="post", color=V_COLORS[k], linewidth=1.8,
                label="V = " + str(V_LIST[k]) + " V (" + str(et.shape[0]) +
                " switches)")
    ax.set_xscale("log")
    ax.set_xlabel("clock (s, log scale)")
    ax.set_ylabel("number of ON junctions")
    ax.set_title("Switches vs time (log time axis)")
    ax.grid(alpha=0.25)
    # legend below the plot so it never covers a curve
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    path = os.path.join(IMAGES_FOLDER, "switches_vs_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_current_vs_switches(net):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in range(len(V_LIST)):
        res, s, d = run_one(net, V_LIST[k])
        ch = res["current_history"]
        if ch.shape[0] < 2:
            continue
        ax.plot(np.arange(ch.shape[0]), ch, color=V_COLORS[k], linewidth=1.8,
                label="V = " + str(V_LIST[k]) + " V")
    ax.set_yscale("log")
    ax.set_xlabel("number of ON junctions")
    ax.set_ylabel("total current (G0 * V, log scale)")
    ax.set_title("Total current vs number of switches")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    path = os.path.join(IMAGES_FOLDER, "current_vs_switches.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    network = nn.generate_network(seed=SEED)
    print("Network:", network.n_wires(), "wires,", network.n_junctions(), "junctions")
    print("Saved:", figure_network(network))
    print("Saved:", figure_switches_vs_time(network))
    print("Saved:", figure_current_vs_switches(network))
