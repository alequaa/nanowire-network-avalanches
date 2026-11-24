# nw_threshold.py
# Helpers for the activation threshold Vth of a network.
#
# Hochstetter et al. (Nat. Commun. 12, 4008, DOI 10.1038/s41467-021-24260-z):
# Vth = n * Vset, where n = number of junctions on the SHORTEST path from a
# source wire to a drain wire.
# In our model we define Vth directly: it is the smallest FIXED voltage at
# which, starting from all junctions OFF and Lambda = 0, the junctions that
# end up fully ON connect a source wire to a drain wire.
#
# Functions:
#   shortest_path_junctions : n, the number of junctions on the shortest path
#   on_path_exists          : do the ON junctions connect source and drain?
#   path_forms              : run the loop at one voltage, then on_path_exists
#   find_threshold          : Vth by bisection on the voltage

import numpy as np

from nw_loop_levels import run_loop_levels


def shortest_path_junctions(n_nodes, junctions, sources, drains):
    # Breadth-first search from all source wires; returns the smallest number
    # of junctions between any source wire and any drain wire (None if the
    # drains cannot be reached).
    # Example: chain S - M - D has 2 junctions, so the answer is 2.
    neighbors = []
    for i in range(n_nodes):
        neighbors.append([])
    for k in range(junctions.shape[0]):
        a = int(junctions[k, 0])
        b = int(junctions[k, 1])
        neighbors[a].append(b)
        neighbors[b].append(a)
    dist = {}
    queue = []
    for s in sources:
        dist[s] = 0
        queue.append(s)
    head = 0
    while head < len(queue):
        node = queue[head]
        head = head + 1
        for nb in neighbors[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    best = None
    for d in drains:
        if d in dist:
            if best is None:
                best = dist[d]
            else:
                if dist[d] < best:
                    best = dist[d]
    return best


def on_path_exists(n_nodes, junctions, on_junctions, sources, drains):
    # True if the ON junctions alone connect some source wire to some drain
    # wire (search on the graph made only of the ON junctions).
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


def path_forms(net, s, d, v, K, v_set, lam_crit, g_off, g_on):
    # Fresh run (all junctions at level 0, Lambda = 0) at the fixed voltage v.
    # Returns True if the junctions at the top level (ON) connect a source
    # wire to a drain wire.
    res = run_loop_levels(net.n_wires(), net.junctions, s, d, v,
                          v_set, lam_crit, g_off, g_on, K)
    on_list = np.nonzero(res["is_on"])[0]
    return on_path_exists(net.n_wires(), net.junctions, on_list, s, d)


def find_threshold(net, s, d, K, v_set, lam_crit, g_off, g_on,
                   tol=2.0e-4, v_low=0.011, v_first=0.05, v_max=1.6):
    # Vth by bisection. Returns nan if no path forms up to v_max.
    #   1. Double the voltage (0.05, 0.1, 0.2, ...) until a path forms.
    #   2. Halve the interval [lo, hi] until it is smaller than tol (volts).
    # Assumes that raising the voltage never destroys the path.
    # Example: no path at 0.2 V but a path at 0.4 V -> start with [0.2, 0.4],
    # test 0.3 V, and so on.
    lo = v_low              # just above Vset: below it nothing can switch
    v = v_first
    found = False
    while v <= v_max and not found:
        if path_forms(net, s, d, v, K, v_set, lam_crit, g_off, g_on):
            found = True
        else:
            lo = v
            v = v * 2.0
    if not found:
        return float("nan")
    hi = v
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if path_forms(net, s, d, mid, K, v_set, lam_crit, g_off, g_on):
            hi = mid
        else:
            lo = mid
    return hi
