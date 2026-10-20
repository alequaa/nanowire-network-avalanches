# run_tau_ramp.py
# Tau with definition C (voltage ramp, one avalanche per voltage step),
# pooled over many random networks, for several step sizes dV.
# Point electrodes, 100-wire networks, ramp from V_START to V_END.
# Reference (Hochstetter, DOI 10.1038/s41467-021-24260-z): tau = 1.95 +- 0.05,
# obtained with a different protocol.
# Run: python run_tau_ramp.py   (figure goes in the images folder)

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
from nw_ramp import ramp_avalanches
from nw_tau import fit_tau, bootstrap_tau

V_SET = 0.01
LAM_CRIT = 0.01
G_OFF = 1.0e-3
G_ON = 1.0

N_NETWORKS = 300
V_START = 0.011          # just above Vset: below Vset nothing can switch
V_END = 0.30
DV_LIST = [0.001, 0.002, 0.005]
SMIN_LIST = [1, 2, 3]
TAU_HOCHSTETTER = 1.95

DV_COLORS = ["#9ac0e6", "#3b7bbf", "#143d6b"]
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def sizes_for_step(dv):
    # One array of avalanche sizes per network.
    v_values = np.arange(V_START, V_END + 1.0e-12, dv)
    per_network = []
    for seed in range(N_NETWORKS):
        net = nn.generate_network(seed=seed)
        s, d = nn.point_electrodes(net)
        out = ramp_avalanches(net.n_wires(), net.junctions, s, d, v_values,
                              V_SET, LAM_CRIT, G_OFF, G_ON)
        if out["S"].shape[0] > 0:
            per_network.append(out["S"])
    return per_network


if __name__ == "__main__":
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    print("Definition C, ramp %.3f -> %.2f V, %d networks per dV" %
          (V_START, V_END, N_NETWORKS))
    print("     dV  avalanches  frac(S=1)  max S |  tau (smin=1, 2, 3)"
          "          | bootstrap std (smin=1)")
    for k in range(len(DV_LIST)):
        dv = DV_LIST[k]
        per_network = sizes_for_step(dv)
        pooled = np.concatenate(per_network)
        smax = int(pooled.max())
        taus = []
        for smin in SMIN_LIST:
            taus.append(fit_tau(pooled, smin, smax))
        mean_b, std_b = bootstrap_tau(per_network, 1, smax, n_boot=200)
        print("%7.3f  %10d  %9.3f  %5d |  %.3f  %.3f  %.3f      | %.3f" %
              (dv, pooled.shape[0], np.mean(pooled == 1), smax,
               taus[0], taus[1], taus[2], std_b))
        values, counts = np.unique(pooled, return_counts=True)
        ax.loglog(values, counts / counts.sum(), "o", markersize=3.5,
                  color=DV_COLORS[k], label="dV = " + str(dv) + " V")

    ss = np.array([1.0, 100.0])
    ax.loglog(ss, 0.6 * ss ** (-TAU_HOCHSTETTER), "k--", linewidth=1,
              label="slope -" + str(TAU_HOCHSTETTER) + " (Hochstetter)")
    ax.set_xlabel("avalanche size S (number of switches in one voltage step)")
    ax.set_ylabel("P(S)")
    ax.set_title("Definition C (voltage ramp), pooled over " + str(N_NETWORKS) +
                 " networks")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    path = os.path.join(IMAGES_FOLDER, "tau_definition_C.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)
