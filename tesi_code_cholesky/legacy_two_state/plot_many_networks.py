# plot_many_networks.py
# 20 random networks at the same voltage (point electrodes), overlaid.
#   Left : total current vs number of ON junctions (one thin line per network)
#   Right: at which number of ON junctions the biggest single jump of the
#          current happens (histogram over the networks)
# Also prints, for every network, the jump and the avalanche summary.
# Run: python plot_many_networks.py   (figure goes in the images folder)

# The shared library files (nanowire_network.py, nw_solver.py, ...) are in the
# main folder, one level up: add it to the search path.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nanowire_network as nn
from nw_loop_model import run_loop
from nw_avalanches import find_avalanches

V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0

V_EXT = 0.15
N_NETWORKS = 20          # seeds 0 ... 19

COLOR_LINE = "#3b7bbf"
COLOR_JUMP = "#d9622b"

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def biggest_jump(current_history):
    # Largest ratio I[k+1] / I[k] between two consecutive steps.
    # Returns (k, ratio): the jump happens when going from k to k+1 ON.
    ratios = current_history[1:] / current_history[:-1]
    k = int(np.argmax(ratios))
    return k, float(ratios[k])


def on_path_exists(n_nodes, junctions, on_junctions, sources, drains):
    # True if the ON junctions alone connect some source wire to some drain
    # wire (breadth-first search on the graph of ON junctions only).
    neighbors = []
    for i in range(n_nodes):
        neighbors.append([])
    for k in on_junctions:
        a = int(junctions[k, 0])
        b = int(junctions[k, 1])
        neighbors[a].append(b)
        neighbors[b].append(a)
    seen = set(sources)
    queue = list(sources)
    while len(queue) > 0:
        node = queue.pop()
        for nb in neighbors[node]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    found = False
    for dr in drains:
        if dr in seen:
            found = True
    return found


def first_path_step(n_nodes, junctions, event_junction, sources, drains):
    # Smallest k such that the first k switched junctions already connect
    # source and drain. Returns None if that never happens.
    for k in range(1, len(event_junction) + 1):
        if on_path_exists(n_nodes, junctions, event_junction[:k], sources, drains):
            return k
    return None


if __name__ == "__main__":
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    jump_positions = []
    print("seed  switches  biggest-jump(k, ratio)  I before -> after   "
          "avalanches  largest S  first S-D path at k")
    same_step = 0
    used = 0
    for seed in range(N_NETWORKS):
        net = nn.generate_network(seed=seed)
        s, d = nn.point_electrodes(net)
        res = run_loop(net.n_wires(), net.junctions, s, d, V_EXT,
                       V_SET, LAM_CRIT, G_OFF, G_ON)
        ch = res["current_history"]
        n_sw = len(res["event_junction"])
        if n_sw < 2:
            print("%4d  %8d  (no jump: too few switches)" % (seed, n_sw))
            continue

        ax1.plot(np.arange(ch.shape[0]), ch, color=COLOR_LINE,
                 linewidth=1.0, alpha=0.55)
        k, ratio = biggest_jump(ch)
        jump_positions.append(k + 1)
        # mark the jump on its own curve
        ax1.plot([k + 1], [ch[k + 1]], "o", color=COLOR_JUMP, markersize=4,
                 zorder=3)

        av = find_avalanches(res["event_time"])
        k_path = first_path_step(net.n_wires(), net.junctions,
                                 res["event_junction"], s, d)
        used = used + 1
        if k_path == k + 1:
            same_step = same_step + 1
        print("%4d  %8d  (%3d, x%7.1f)       %.2e -> %.2e   %5d      %5d      %s" %
              (seed, n_sw, k + 1, ratio, ch[k], ch[k + 1],
               len(av["S"]), int(av["S"].max()), str(k_path)))

    ax1.set_yscale("log")
    ax1.set_xlabel("number of ON junctions")
    ax1.set_ylabel("total current (G0 * V, log scale)")
    ax1.set_title("Current vs switches, " + str(N_NETWORKS) + " networks at V = " +
                  str(V_EXT) + " V\n(orange dot = biggest single jump)")
    ax1.grid(alpha=0.25)

    ax2.hist(jump_positions, bins=np.arange(0, max(jump_positions) + 6, 5),
             color=COLOR_JUMP, edgecolor="white")
    ax2.set_xlabel("number of ON junctions right after the biggest jump")
    ax2.set_ylabel("number of networks")
    ax2.set_title("Where the biggest jump happens")
    ax2.grid(alpha=0.25, axis="y")

    path = os.path.join(IMAGES_FOLDER, "many_networks_current.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nJump positions (ON junctions after the jump): mean %.1f, "
          "median %.1f, min %d, max %d" % (np.mean(jump_positions),
                                           np.median(jump_positions),
                                           min(jump_positions),
                                           max(jump_positions)))
    print("The biggest jump coincides with the first ON path source-drain "
          "in", same_step, "of", used, "networks")
    print("Saved:", path)
