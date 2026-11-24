# check_vth.py
# Hochstetter (DOI 10.1038/s41467-021-24260-z): the activation threshold is
# Vth = n * Vset, with n = number of junctions on the SHORTEST source-drain path.
# Check in our model: compare n * Vset with V_path (ramp, first ON path).
# Run: python check_vth.py

# The shared library files (nanowire_network.py, nw_solver.py, ...) are in the
# main folder, one level up: add it to the search path.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import nanowire_network as nn
from nw_ramp import ramp_avalanches

V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0
DV = 0.002

# name, number of networks, wires, side, electrodes, V_END
CASES = [
    ("small", 100, 100, 30.0, "point", 0.30),
    ("large", 20, 1000, 100.0, "strip", 0.40),
]


def shortest_path_junctions(n_nodes, junctions, sources, drains):
    # Breadth-first search from all source wires; returns the smallest number
    # of junctions between any source wire and any drain wire.
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


if __name__ == "__main__":
    for name, n_net, n_wires, side, electrodes, v_end in CASES:
        v_values = np.arange(0.011, v_end + 1.0e-12, DV)
        ns = []
        vp = []
        for seed in range(n_net):
            net = nn.generate_network(n_wires=n_wires, side=side, seed=seed)
            if electrodes == "point":
                s, d = nn.point_electrodes(net)
            else:
                s, d = nn.strip_electrodes(net, 0.025)
            n = shortest_path_junctions(net.n_wires(), net.junctions, s, d)
            out = ramp_avalanches(net.n_wires(), net.junctions, s, d, v_values,
                                  V_SET, LAM_CRIT, G_OFF, G_ON)
            ns.append(n)
            vp.append(out["V_path"] / V_SET)
        ns = np.array(ns, dtype=float)
        vp = np.array(vp, dtype=float)
        ok = ~np.isnan(vp)
        print("\n=== %s (%d networks)" % (name, n_net))
        print("shortest path n: mean %.2f, min %d, max %d" %
              (ns.mean(), ns.min(), ns.max()))
        print("V_path / Vset  : mean %.2f, min %.1f, max %.1f" %
              (vp[ok].mean(), vp[ok].min(), vp[ok].max()))
        print("V_path/Vset - n: mean %.2f, std %.2f; correlation %.2f" %
              ((vp[ok] - ns[ok]).mean(), (vp[ok] - ns[ok]).std(),
               np.corrcoef(vp[ok], ns[ok])[0, 1]))
        print("first 8 networks (n, V_path/Vset):",
              [(int(ns[i]), round(float(vp[i]), 1)) for i in range(min(8, n_net))])
