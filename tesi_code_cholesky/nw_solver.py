# nw_solver.py
# Kirchhoff solver for the random nanowire network.
# Nodes = nanowires, edges = junctions. The network is given by two arrays:
#   junctions  (m,2): which two wires each junction connects
#   g_junction (m,) : conductance of each junction
# Sources and drains are sets of nodes (a single point = set of size 1).
#
# Example: chain S --g1-- M --g2-- D with g1 = 1, g2 = 3, V_S = 4, V_D = 0.
# Kirchhoff on M: g1*(V_M - 4) + g2*V_M = 0 -> V_M = 4/4 = 1.
# Drops: 3 on g1, 1 on g2. Current: 1*3 = 3*1 = 3 (same, as it must be).

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def solve_node_voltages(n_nodes, junctions, g_junction, sources, drains, v_ext):
    # Returns the voltage of every node. Sources are held at v_ext,
    # drains at 0, all other nodes are unknowns.

    # Mark electrode nodes (known voltage).
    is_fixed = np.zeros(n_nodes, dtype=bool)
    v_fixed = np.zeros(n_nodes)
    for i in sources:
        is_fixed[i] = True
        v_fixed[i] = v_ext
    for i in drains:
        if is_fixed[i]:
            raise ValueError("node " + str(i) + " is both source and drain")
        is_fixed[i] = True
        v_fixed[i] = 0.0

    # Weighted Laplacian L: diagonal = sum of g at the node, off-diagonal = -g.
    # Repeated entries (several junctions on one wire) are summed by scipy.
    a = junctions[:, 0]
    b = junctions[:, 1]
    g = g_junction
    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([a, b, b, a])
    vals = np.concatenate([g, g, -g, -g])
    L = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    unknown = np.nonzero(~is_fixed)[0]
    known = np.nonzero(is_fixed)[0]

    voltage = np.zeros(n_nodes)
    voltage[known] = v_fixed[known]

    if unknown.shape[0] > 0:
        # Kirchhoff on the unknown nodes: L_uu x = - L_uk v_known
        L_uu = L[unknown][:, unknown]
        L_uk = L[unknown][:, known]
        rhs = -(L_uk @ v_fixed[known])
        x = spsolve(L_uu.tocsc(), rhs)
        voltage[unknown] = x
    return voltage


def junction_drops(junctions, voltage):
    # |V_a - V_b| on every junction (the |V_i| of the loop).
    return np.abs(voltage[junctions[:, 0]] - voltage[junctions[:, 1]])


def total_current(n_nodes, junctions, g_junction, voltage, sources):
    # Total current leaving the source set. Junctions with both ends in the
    # sources carry no current (same voltage on both ends) and are skipped.
    is_source = np.zeros(n_nodes, dtype=bool)
    for i in sources:
        is_source[i] = True
    a = junctions[:, 0]
    b = junctions[:, 1]
    current = 0.0
    for k in range(junctions.shape[0]):
        if is_source[a[k]] and (not is_source[b[k]]):
            current = current + g_junction[k] * (voltage[a[k]] - voltage[b[k]])
        elif is_source[b[k]] and (not is_source[a[k]]):
            current = current + g_junction[k] * (voltage[b[k]] - voltage[a[k]])
    return current


def total_current_into_drains(n_nodes, junctions, g_junction, voltage, drains):
    # Total current entering the drain set (must equal the source current).
    # Junctions with both ends in the drains carry no current (both at 0 V).
    is_drain = np.zeros(n_nodes, dtype=bool)
    for i in drains:
        is_drain[i] = True
    a = junctions[:, 0]
    b = junctions[:, 1]
    current = 0.0
    for k in range(junctions.shape[0]):
        if is_drain[a[k]] and (not is_drain[b[k]]):
            current = current + g_junction[k] * (voltage[b[k]] - voltage[a[k]])
        elif is_drain[b[k]] and (not is_drain[a[k]]):
            current = current + g_junction[k] * (voltage[a[k]] - voltage[b[k]])
    return current
