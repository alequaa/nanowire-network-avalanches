# nw_current.py
# Total current (and so the conductance of the network) computed from the node
# voltages, without slow Python loops.
#
# Total current = sum of the currents that leave the sources. Every junction
# that has ONE end in the sources and the other end outside carries the current
#       I = g * (V_source - V_other_end)
# (junctions with both ends in the sources have the same voltage on both sides:
# zero current, they are skipped).
# This is the same quantity as nw_solver.total_current, but computed with numpy.
#
# Numerical example (chain S - M1 - M2 - D, g = 1, V_S = 1, V_D = 0):
# voltages [1, 2/3, 1/3, 0]; the only junction that touches the source is S-M1:
# I = 1 * (1 - 2/3) = 1/3. The conductance of the network is G = I / V = 1/3.

import numpy as np


def make_current_meter(n_nodes, junctions, sources):
    # Prepares the indices once; returns a function meter(voltage, g_junction)
    # that gives the total current.
    is_source = np.zeros(n_nodes, dtype=bool)
    for i in sources:
        is_source[i] = True
    a = junctions[:, 0]
    b = junctions[:, 1]
    # junctions with the first end in the sources and the second end outside
    idx_a = np.nonzero(is_source[a] & (~is_source[b]))[0]
    # junctions with the second end in the sources and the first end outside
    idx_b = np.nonzero(is_source[b] & (~is_source[a]))[0]
    a_of_idx_a = a[idx_a]
    b_of_idx_a = b[idx_a]
    a_of_idx_b = a[idx_b]
    b_of_idx_b = b[idx_b]

    def meter(voltage, g_junction):
        current = np.sum(g_junction[idx_a] *
                         (voltage[a_of_idx_a] - voltage[b_of_idx_a]))
        current = current + np.sum(g_junction[idx_b] *
                                   (voltage[b_of_idx_b] - voltage[a_of_idx_b]))
        return float(current)

    return meter
