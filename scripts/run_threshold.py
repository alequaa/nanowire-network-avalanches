# run_threshold.py
# Does the multi-level loop (nw_loop_levels.py) give the threshold of
# Hochstetter, Vth = n * Vset (n = junctions on the shortest source-drain
# path)? For K = 1 we found about n/2.
# Vth here = smallest fixed voltage, from all OFF and Lambda = 0, at which
# the junctions at the top level (ON) connect source and drain (bisection,
# see nw_threshold.py).
# For every K it prints: mean n, Vth/Vset, the ratio Vth/(n*Vset) and the
# correlation between Vth and n, over N_NETWORKS random networks.
# Run: python scripts/run_threshold.py   (about 3 minutes, 2 processes)

# The library files (nanowire_network.py, nw_solver.py, ...) are in the main
# folder, one level up: add it to the search path.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from multiprocessing import Pool

import numpy as np

import nanowire_network as nn
from nw_threshold import shortest_path_junctions, find_threshold

V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0
K_LIST = [1, 3, 10]
N_NETWORKS = 60


def one(args):
    # Worker: one network (seed) and one K. Returns (n, Vth).
    seed, K = args
    net = nn.generate_network(seed=seed)
    s, d = nn.point_electrodes(net)
    n = shortest_path_junctions(net.n_wires(), net.junctions, s, d)
    vth = find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
    return n, vth


if __name__ == "__main__":
    for K in K_LIST:
        with Pool(2) as pool:
            res = pool.map(one, [(seed, K) for seed in range(N_NETWORKS)], chunksize=2)
        ns = np.array([r[0] for r in res], dtype=float)
        vth = np.array([r[1] for r in res], dtype=float) / V_SET
        ok = ~np.isnan(vth)
        ratio = vth[ok] / ns[ok]
        print("K = %2d: n mean %.2f | Vth/Vset mean %.2f (min %.1f, max %.1f) | "
              "ratio Vth/(n*Vset): mean %.2f, min %.2f, max %.2f | correlation %.2f"
              % (K, ns[ok].mean(), vth[ok].mean(), vth[ok].min(), vth[ok].max(),
                 ratio.mean(), ratio.min(), ratio.max(),
                 np.corrcoef(vth[ok], ns[ok])[0, 1]), flush=True)
