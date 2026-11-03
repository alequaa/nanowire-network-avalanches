# check_first_avalanche.py
# Why is the FIRST avalanche of every network so large?
# (check_last_avalanche.py showed that at V* = 1.8 it is the largest avalanche
#  of the network in 80-98% of the networks, and that it makes the bump.)
#
# Two hypotheses, three tests. This script RUNS the networks again (the cache
# files keep only the sizes S, not the times of the events), with the same
# seeds as run_average_size.py: seed 0, 1, 2, ... are the same networks.
#
# Hypothesis 1: the frame method merges the first events because at the start
#   the events are much closer in time than the average, and the frame width is
#   the average over the whole run.
#   Example: 1000 events in 100 s -> frame = 0.1 s. If the first 200 events
#   happen in 1 s, they fill 10 frames in a row (20 per frame, none empty) and
#   become ONE avalanche of S = 200.
#
#   Test A (descriptive): median time between consecutive events (dt) in the
#     first 10% of the events and in the last 10%, divided by the frame width.
#     If H1 is right, the first ratio is much smaller than 1 and the last is
#     around 1 or larger. Also: which fraction of the events and of the time is
#     the first avalanche?
#   Test B (decisive): cut the run in SEG pieces with the same number of events
#     (default 10) and compute the frame width inside each piece. The rate is
#     then roughly constant inside a piece. If the first avalanche is only a
#     frame artefact, it breaks into many smaller ones. If it stays huge, it is
#     a real cascade and H1 is wrong.
#
# Hypothesis 2: the first avalanche is the formation of the first path of ON
#   junctions from the source to the drain (the same path that defines the
#   threshold Vth in nw_threshold.py).
#
#   Test C: replay the events and compute, with Kirchhoff, the total
#     conductance G = current / V after the events. We find the exact event
#     after which a source-drain path of ON junctions first exists (bisection:
#     levels only go up, so once the path exists it stays) and ask in which
#     avalanche that event is. Also G at the start, at the end of the first
#     avalanche and at the end of the run.
#
# Run (from the main folder):
#   python scripts/check_first_avalanche.py
#   python scripts/check_first_avalanche.py 100:30:20 400:60:10
# Each argument is n_wires:side_um:n_networks. Default: 100:30:20 400:60:10
# Optional environment variables: K (default 10), RATIO (default 1000),
# PROCS (default 2), SEG (pieces for test B, default 10), SMIN (default 2).
#   PowerShell:  $env:PROCS = 4
# Time: about the same as run_average_size.py for the same number of networks
# (the replay adds a few seconds per network).
# Output: table on screen and, for the network with seed 0, one picture per
# size and V*: images/first_avalanche_K<K>_w<wires>_V<V*>.png

import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nanowire_network as nn
from nw_loop_levels import run_loop_levels
from nw_avalanches import find_avalanches
from nw_solver import solve_node_voltages, total_current
from nw_threshold import find_threshold, on_path_exists
from nw_tau import fit_tau

V_SET = 0.01
LAM_CRIT = 0.01
G_ON = 1.0
# Read at the top of the file so that worker processes (Windows starts them
# from scratch) read the same values.
RATIO = float(os.environ.get("RATIO", "1000"))
G_OFF = G_ON / RATIO
K = int(os.environ.get("K", "10"))
N_PROCS = int(os.environ.get("PROCS", "2"))
# Pieces for test B. Default 10; with K = 1 default 0 = test B skipped, because a
# network has only ~10-40 events, too few to cut in 10 pieces.
if K > 1:
    SEG_DEFAULT = "10"
else:
    SEG_DEFAULT = "0"
N_SEG = int(os.environ.get("SEG", SEG_DEFAULT))
SMIN = int(os.environ.get("SMIN", "2"))
FACTORS = [1.05, 1.8]
FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(FOLDER, "images")
N_CURVE_POINTS = 60          # points of the G(t) curve (only seed 0)

if len(sys.argv) > 1:
    SIZE_SPECS = sys.argv[1:]
else:
    SIZE_SPECS = ["100:30:20", "400:60:10"]


def levels_after(res, n_events, n_junctions):
    # Level of every junction after the first n_events events of the run.
    # Levels only go up, so for a junction with several events we keep the
    # largest level (np.maximum.at does that even for repeated indices).
    # Example: events (junction 5, level 1), (junction 5, level 2) -> level 2.
    level = np.zeros(n_junctions, dtype=int)
    if n_events > 0:
        np.maximum.at(level, res["event_junction"][:n_events],
                      res["event_level"][:n_events])
    return level


def conductance_after(net, s, d, v, res, n_events, level_g):
    # Total conductance G = current / V after n_events events.
    level = levels_after(res, n_events, net.junctions.shape[0])
    g = level_g[level]
    voltage = solve_node_voltages(net.n_wires(), net.junctions, g, s, d, v)
    current = total_current(net.n_wires(), net.junctions, g, voltage, s)
    return abs(current) / v


def path_exists_after(net, s, d, res, n_events):
    # True if the ON junctions (top level K) connect source and drain after
    # n_events events.
    level = levels_after(res, n_events, net.junctions.shape[0])
    on_list = np.nonzero(level == K)[0]
    return on_path_exists(net.n_wires(), net.junctions, on_list, s, d)


def first_path_event(net, s, d, res):
    # Smallest number of events after which an ON path exists (bisection).
    # Returns None if there is no path even at the end of the run.
    n_all = res["event_time"].shape[0]
    if not path_exists_after(net, s, d, res, n_all):
        return None
    low = 0          # no path after 0 events (all junctions at level 0)
    high = n_all     # path exists after all events
    while high - low > 1:
        mid = (low + high) // 2
        if path_exists_after(net, s, d, res, mid):
            high = mid
        else:
            low = mid
    return high


def segmented_sizes(times, n_seg):
    # Test B: avalanche sizes with the frame width computed inside each of
    # n_seg pieces with (almost) the same number of events.
    n = times.shape[0]
    bounds = np.linspace(0, n, n_seg + 1).astype(int)
    sizes = []
    for j in range(n_seg):
        chunk = times[bounds[j]:bounds[j + 1]]
        if chunk.shape[0] == 0:
            continue
        av = find_avalanches(chunk)
        for value in av["S"]:
            sizes.append(int(value))
    return np.array(sizes, dtype=int)


def analyse(net, s, d, v, res, level_g, want_curve):
    # All numbers of the three tests for one network at one voltage.
    is_new = res["event_new_episode"]
    ep_idx = np.nonzero(is_new)[0]          # positions of the episode events
    times = res["event_time"][ep_idx]
    n_ep = times.shape[0]
    if n_ep < 3:
        return None
    av = find_avalanches(times)
    sizes = np.asarray(av["S"], dtype=int)
    width = av["frame_width"]
    s0 = int(sizes[0])
    out = {}
    out["S"] = sizes
    if N_SEG > 0:
        out["S_seg"] = segmented_sizes(times, N_SEG)
    else:
        out["S_seg"] = None
    out["n_ep"] = n_ep

    # ---- test A ----
    dt = np.diff(times)
    n10 = max(1, int(0.1 * dt.shape[0]))
    out["dt_first_over_w"] = float(np.median(dt[:n10])) / width
    out["dt_last_over_w"] = float(np.median(dt[-n10:])) / width
    total_time = times[-1] - times[0]
    out["frac_events_first"] = s0 / float(n_ep)
    out["frac_time_first"] = (times[s0 - 1] - times[0]) / total_time

    # ---- test C ----
    n_all = res["event_time"].shape[0]
    # events (of the full list) up to and including the last event of the
    # first avalanche
    e0 = int(ep_idx[s0 - 1]) + 1
    g_start = conductance_after(net, s, d, v, res, 0, level_g)
    g_first = conductance_after(net, s, d, v, res, e0, level_g)
    g_final = conductance_after(net, s, d, v, res, n_all, level_g)
    out["G_start"] = g_start
    out["G_after_first"] = g_first
    out["G_final"] = g_final
    e_path = first_path_event(net, s, d, res)
    out["path_in_first"] = None
    out["path_rank_over_S0"] = None
    out["path_in_last"] = None
    out["path_in_largest"] = None
    out["n_av"] = int(sizes.shape[0])
    if e_path is not None:
        # number of episode events up to and including event e_path
        rank = int(np.sum(is_new[:e_path]))
        cum = np.cumsum(sizes)
        index_avalanche = int(np.searchsorted(cum, rank, side="left"))
        if index_avalanche == 0:
            out["path_in_first"] = True
        else:
            out["path_in_first"] = False
        out["path_rank_over_S0"] = rank / float(s0)
        # same question for the LAST and for the LARGEST avalanche
        # Example: S = [3, 1, 5, 2], path forms at episode number 7 -> cum = [3, 4, 9, 11],
        # first cum >= 7 is 9 -> index 2: the third avalanche (the largest, not the last).
        if index_avalanche == sizes.shape[0] - 1:
            out["path_in_last"] = True
        else:
            out["path_in_last"] = False
        if index_avalanche == int(np.argmax(sizes)):
            out["path_in_largest"] = True
        else:
            out["path_in_largest"] = False

    # ---- data for the picture (only the network with seed 0) ----
    if want_curve:
        out["times"] = times
        out["dt"] = dt
        out["S0"] = s0
        out["width"] = width
        out["t_first_end"] = float(times[s0 - 1])
        if e_path is not None:
            out["t_path"] = float(res["event_time"][e_path - 1])
        else:
            out["t_path"] = None
        points = np.unique(np.linspace(1, n_all, N_CURVE_POINTS).astype(int))
        curve_t = []
        curve_g = []
        for e in points:
            curve_t.append(float(res["event_time"][e - 1]))
            curve_g.append(conductance_after(net, s, d, v, res, int(e), level_g))
        out["curve_t"] = np.array(curve_t)
        out["curve_G"] = np.array(curve_g)
    return out


def one_network(args):
    # Worker: one network, all the V*. Same seeds as run_average_size.py.
    seed, n_wires, side = args
    net = nn.generate_network(n_wires=n_wires, side=side, seed=seed)
    s, d = nn.point_electrodes(net)
    vth = find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
    out = {}
    if np.isnan(vth):
        return seed, vth, out
    level_g = np.zeros(K + 1, dtype=float)
    for k in range(K + 1):
        level_g[k] = G_OFF * (G_ON / G_OFF) ** (float(k) / K)
    for f in FACTORS:
        v = f * vth
        res = run_loop_levels(net.n_wires(), net.junctions, s, d, v,
                              V_SET, LAM_CRIT, G_OFF, G_ON, K)
        out[f] = analyse(net, s, d, v, res, level_g, seed == 0)
    return seed, vth, out


def pooled_tau(list_of_sizes):
    # One tau from all the avalanches pooled (smax = largest observed, and 100).
    pooled = np.concatenate(list_of_sizes)
    t_obs = fit_tau(pooled, SMIN, int(pooled.max()))
    t_100 = fit_tau(pooled, SMIN, 100)
    return t_obs, t_100


def median_or_nan(values):
    if len(values) == 0:
        return float("nan")
    return float(np.median(values))


def draw_picture(info, n_wires, f):
    # Three panels for one network: N(t), dt, G(t). First avalanche in grey.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    times = info["times"]
    t_end = info["t_first_end"]
    blue = "#0f3d75"
    light = "#9ec5ee"

    ax = axes[0]
    ax.plot(times, np.arange(1, times.shape[0] + 1), color=blue, linewidth=1.6)
    ax.axvspan(times[0], t_end, color="0.85")
    ax.set_xscale("log")
    ax.set_xlabel("time (clock)")
    ax.set_ylabel("number of events so far")
    ax.set_title("events against time (first avalanche in grey)")

    ax = axes[1]
    dt = info["dt"]
    index = np.arange(1, dt.shape[0] + 1)
    keep = dt > 0
    ax.scatter(index[keep], dt[keep], s=6, color=blue, alpha=0.5)
    ax.axhline(info["width"], color="#c0392b", linewidth=1.4, label="frame width")
    ax.axvline(info["S0"], color="0.5", linestyle="--", label="end of first avalanche")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("event number")
    ax.set_ylabel("time to the next event, dt")
    ax.set_title("dt against event number")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    ax.plot(info["curve_t"], info["curve_G"], "-o", color=blue, markersize=3,
            linewidth=1.4, markerfacecolor=light)
    ax.axvspan(times[0], t_end, color="0.85")
    if info["t_path"] is not None:
        ax.axvline(info["t_path"], color="#c0392b", linewidth=1.4,
                   label="first ON path source-drain")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time (clock)")
    ax.set_ylabel("total conductance G = I / V")
    ax.set_title("conductance against time")

    for ax in axes:
        ax.grid(True, which="major", color="0.88", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("%d wires, V* = %.2f, network seed 0 (K = %d)" % (n_wires, f, K),
                 fontsize=11)
    fig.tight_layout()
    name = "first_avalanche_K%d_w%d_V%.2f.png" % (K, n_wires, f)
    fig.savefig(os.path.join(IMAGES, name), dpi=150)
    plt.close(fig)
    return name


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    print("K = %d, ratio %g, test B with %d pieces (0 = skipped), smin = %d"
          % (K, RATIO, N_SEG, SMIN))
    for spec in SIZE_SPECS:
        parts = spec.split(":")
        n_wires = int(parts[0])
        side = float(parts[1])
        n_networks = int(parts[2])
        jobs = [(seed, n_wires, side) for seed in range(n_networks)]
        with Pool(N_PROCS) as pool:
            results = pool.map(one_network, jobs, chunksize=1)
        for f in FACTORS:
            infos = []
            for r in results:
                if f in r[2] and r[2][f] is not None:
                    infos.append(r[2][f])
            n = len(infos)
            print("\n=== %d wires, V* = %.2f, %d networks" % (n_wires, f, n))
            if n == 0:
                continue

            # ---- A ----
            print("A) median dt / frame width: first 10%% of the events %.3f, "
                  "last 10%% %.3f" % (
                      median_or_nan([i["dt_first_over_w"] for i in infos]),
                      median_or_nan([i["dt_last_over_w"] for i in infos])))
            print("   first avalanche = median %.0f%% of the events, %.0f%% of the "
                  "time of the run" % (
                      100.0 * median_or_nan([i["frac_events_first"] for i in infos]),
                      100.0 * median_or_nan([i["frac_time_first"] for i in infos])))

            # ---- B ---- (only if SEG > 0)
            if N_SEG > 0:
                first_largest = 0
                first_largest_seg = 0
                ratio = []
                ratio_seg = []
                big = []
                big_seg = []
                s_first = []
                s_first_seg = []
                for i in infos:
                    if i["S"][0] == i["S"].max():
                        first_largest = first_largest + 1
                    if i["S_seg"][0] == i["S_seg"].max():
                        first_largest_seg = first_largest_seg + 1
                    ratio.append(i["S"][0] / float(i["S"].max()))
                    ratio_seg.append(i["S_seg"][0] / float(i["S_seg"].max()))
                    big.append(int(np.sum(i["S"] > 100)))
                    big_seg.append(int(np.sum(i["S_seg"] > 100)))
                    s_first.append(int(i["S"][0]))
                    s_first_seg.append(int(i["S_seg"][0]))
                print("B) one frame width / %d pieces:" % N_SEG)
                print("   first avalanche is the largest of its network: %3.0f%% / %3.0f%%"
                      % (100.0 * first_largest / n, 100.0 * first_largest_seg / n))
                print("   median S_first: %.0f / %.0f   median S_first/S_max: %.2f / %.2f"
                      % (np.median(s_first), np.median(s_first_seg),
                         np.median(ratio), np.median(ratio_seg)))
                print("   avalanches with S > 100 per network (mean): %.2f / %.2f"
                      % (np.mean(big), np.mean(big_seg)))
                t1 = pooled_tau([i["S"] for i in infos])
                t2 = pooled_tau([i["S_seg"] for i in infos])
                print("   tau pooled (smax observed / <=100): %.2f / %.2f   with pieces: "
                      "%.2f / %.2f" % (t1[0], t1[1], t2[0], t2[1]))

            # ---- C ----
            inside = 0
            known = 0
            ranks = []
            for i in infos:
                if i["path_in_first"] is not None:
                    known = known + 1
                    if i["path_in_first"]:
                        inside = inside + 1
                    ranks.append(i["path_rank_over_S0"])
            print("C) first ON path source-drain forms inside the first avalanche: "
                  "%d of %d networks" % (inside, known))
            print("   median (event of the path) / S_first: %.2f   "
                  "(<= 1 means inside the first avalanche)" % median_or_nan(ranks))
            n_path = 0
            in_last = 0
            in_largest = 0
            in_first = 0
            last_is_largest = 0
            for i in infos:
                if i["S"][-1] == i["S"].max():
                    last_is_largest = last_is_largest + 1
                if i["path_in_last"] is not None:
                    n_path = n_path + 1
                    if i["path_in_last"]:
                        in_last = in_last + 1
                    if i["path_in_largest"]:
                        in_largest = in_largest + 1
                    if i["path_in_first"]:
                        in_first = in_first + 1
            if n_path > 0:
                print("   the first ON path forms in the: FIRST avalanche %3.0f%%, LAST %3.0f%%, "
                      "LARGEST %3.0f%% of the %d networks with a path (%d without)" % (
                          100.0 * in_first / n_path, 100.0 * in_last / n_path,
                          100.0 * in_largest / n_path, n_path, n - n_path))
            print("   (the last avalanche is the largest in %.0f%% of the %d networks used)"
                  % (100.0 * last_is_largest / n, n))
            print("   G after first avalanche / G final: median %.2f   "
                  "G start / G final: median %.4f" % (
                      median_or_nan([i["G_after_first"] / i["G_final"] for i in infos]),
                      median_or_nan([i["G_start"] / i["G_final"] for i in infos])),
                  flush=True)

            # ---- picture (seed 0) ----
            for r in results:
                if r[0] == 0 and f in r[2] and r[2][f] is not None:
                    name = draw_picture(r[2][f], n_wires, f)
                    print("   picture: images/%s" % name)
