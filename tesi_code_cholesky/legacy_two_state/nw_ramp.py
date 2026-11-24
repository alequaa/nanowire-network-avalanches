# nw_ramp.py
# Definition C of an avalanche: slowly raise the external voltage in small
# steps. At every step run the loop until all switching times are infinite
# (steady state). The switches that happen during ONE step are ONE avalanche.
#   size S     = number of switches in the step
#   duration T = clock of the last switch minus clock of the first switch
# The state (ON junctions, buckets Lambda, clock) is carried from one step to
# the next. No decay of Lambda below Vreset (Hochstetter's model has it).

# The function also finds V_path: the first voltage step after which the ON
# junctions alone connect a source wire to a drain wire (the geometric path
# that makes the current jump).
#
# Example (hand): chain S-M-D, voltages [0.015, 0.04].
#   0.015: drops 0.0075 < Vset -> no switch, no avalanche.
#   0.04 : both junctions switch (same clock) -> one avalanche, S = 2.

import numpy as np

from nw_loop_model import run_loop


def on_path_exists(n_nodes, junctions, on_junctions, sources, drains):
    # True if the ON junctions alone connect some source wire to some drain
    # wire (breadth-first search on the graph made only of ON junctions).
    # (Same function as in plot_many_networks.py and in nw_threshold.py.)
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


def ramp_avalanches(n_nodes, junctions, sources, drains, v_values,
                    v_set, lam_crit, g_off, g_on):
    # Returns a dict: S (sizes), T (durations), V (voltage of each avalanche),
    # V_path (voltage of the step that first connects source and drain with
    # ON junctions, nan if it never happens), is_on (final state), clock.
    m = junctions.shape[0]
    is_on = np.zeros(m, dtype=bool)
    lam = np.zeros(m, dtype=float)
    clock = 0.0

    sizes = []
    durations = []
    volts = []
    v_path = float("nan")
    path_found = False
    for v in v_values:
        res = run_loop(n_nodes, junctions, sources, drains, v,
                       v_set, lam_crit, g_off, g_on,
                       lam_start=lam, is_on_start=is_on, clock_start=clock)
        et = res["event_time"]
        if len(et) > 0:
            sizes.append(len(et))
            durations.append(float(et[-1] - et[0]))
            volts.append(v)
        is_on = res["is_on"]
        lam = res["lam"]
        clock = res["clock"]
        # Look for the first ON path only on steps where something switched.
        if (not path_found) and len(et) > 0:
            on_list = np.nonzero(is_on)[0]
            if on_path_exists(n_nodes, junctions, on_list, sources, drains):
                path_found = True
                v_path = float(v)

    out = {}
    out["S"] = np.array(sizes, dtype=int)
    out["T"] = np.array(durations, dtype=float)
    out["V"] = np.array(volts, dtype=float)
    out["V_path"] = v_path
    out["is_on"] = is_on
    out["clock"] = clock
    return out
