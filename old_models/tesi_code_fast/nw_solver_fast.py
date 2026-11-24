# nw_solver_fast.py
# Faster Kirchhoff solver for the loop in nw_loop_levels.py.
#
# The original solver (nw_solver.solve_node_voltages) builds the matrix and
# factorises it from scratch at EVERY step, although only ONE junction has
# changed. Here we factorise only once in a while and correct the solution in
# between (idea of Nukala and Simunovic, arXiv cond-mat/0503719).
#
# Notation: A = the matrix of the unknown nodes (diagonal = sum of the
# conductances at the node, off-diagonal = minus the conductance), b = right
# hand side (known voltages of the electrodes times the conductances that
# touch them), x = voltages of the unknown nodes. Kirchhoff: A x = b.
#
# Numerical example (chain S - M1 - M2 - D, all g = 1, V_S = 1, V_D = 0):
#   A = [[2,-1],[-1,2]], b = [1,0], x = [2/3, 1/3].
#   The junction M2-D goes from g = 1 to g = 3 (delta = 2). Its vector is
#   v = [0,1] (a +1 on the unknown end; the drain is not an unknown).
#   u = A^-1 v = [1/3, 2/3];  1 + delta*v.u = 1 + 2*(2/3) = 7/3
#   x_new = x - delta*(v.x)/(1 + delta*v.u) * u = [2/3,1/3] - (2/7)*[1/3,2/3]
#         = [4/7, 1/7]     (checked by hand: it solves the new system).
#
# With p junctions changed since the last factorisation (Woodbury formula):
#   A = A0 + V D V^T   (V = columns v_l, D = diag(delta_l))
#   x = y0 - W (D^-1 + V^T W)^-1 V^T y0,   with  W = A0^-1 V,  y0 = A0^-1 b
# W has one column per changed junction (one backsolve each, computed once),
# the matrix D^-1 + V^T W is only p x p (p <= maxupd): tiny.
# The conductances here only GO UP (delta > 0), so this p x p matrix is
# positive definite and well behaved.
#
# Why this is safe: every maxupd changed junctions we rebuild and refactorise
# A from the current conductances, so rounding errors cannot pile up.

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class FastKirchhoff:
    def __init__(self, n_nodes, junctions, sources, drains, v_ext, g, maxupd=25):
        self.n_nodes = n_nodes
        self.m = junctions.shape[0]
        self.maxupd = int(maxupd)

        # Electrode nodes have a known voltage (sources v_ext, drains 0).
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
        self.unknown = np.nonzero(~is_fixed)[0]
        self.known = np.nonzero(is_fixed)[0]
        self.v_known = v_fixed[self.known]
        self.ndof = self.unknown.shape[0]

        # pos[node] = index of the node among the unknowns (-1 if electrode)
        pos = -np.ones(n_nodes, dtype=int)
        pos[self.unknown] = np.arange(self.ndof)
        self.junctions = junctions
        self.ia = pos[junctions[:, 0]]      # unknown index of the first end (-1 = electrode)
        self.ib = pos[junctions[:, 1]]      # same for the second end
        self.node_a = junctions[:, 0]
        self.node_b = junctions[:, 1]

        # Junctions that feed the right hand side b: exactly one end is an
        # unknown, the other end is an electrode with a nonzero voltage.
        # b[unknown end] += g * V_electrode.
        b_j = []
        b_node = []
        b_volt = []
        for k in range(self.m):
            a_unknown = self.ia[k] >= 0
            b_unknown = self.ib[k] >= 0
            if a_unknown and (not b_unknown):
                volt = v_fixed[self.node_b[k]]
                if volt != 0.0:
                    b_j.append(k)
                    b_node.append(self.ia[k])
                    b_volt.append(volt)
            if b_unknown and (not a_unknown):
                volt = v_fixed[self.node_a[k]]
                if volt != 0.0:
                    b_j.append(k)
                    b_node.append(self.ib[k])
                    b_volt.append(volt)
        self.b_j = np.array(b_j, dtype=int)
        self.b_node = np.array(b_node, dtype=int)
        self.b_volt = np.array(b_volt, dtype=float)
        self.touches_b = np.zeros(self.m, dtype=bool)
        self.touches_b[self.b_j] = True

        self.g = np.array(g, dtype=float)
        self.n_refactor = 0
        self.refactor()

    def _rhs(self):
        # b from the CURRENT conductances.
        if self.b_j.shape[0] == 0:
            return np.zeros(self.ndof)
        return np.bincount(self.b_node, weights=self.g[self.b_j] * self.b_volt,
                           minlength=self.ndof)

    def refactor(self):
        # Build A from the current conductances and factorise it (LU).
        g = self.g
        ia = self.ia
        ib = self.ib
        both = (ia >= 0) & (ib >= 0)
        only_a = (ia >= 0) & (ib < 0)
        only_b = (ib >= 0) & (ia < 0)
        rows = np.concatenate([ia[both], ib[both], ia[both], ib[both],
                               ia[only_a], ib[only_b]])
        cols = np.concatenate([ia[both], ib[both], ib[both], ia[both],
                               ia[only_a], ib[only_b]])
        vals = np.concatenate([g[both], g[both], -g[both], -g[both],
                               g[only_a], g[only_b]])
        A = sp.coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()
        self.lu = splu(A)
        self.n_refactor = self.n_refactor + 1

        # reference state: no junction changed yet
        self.g_ref = self.g.copy()
        self.p = 0
        self.W = np.zeros((self.ndof, self.maxupd))
        self.col_junction = np.zeros(self.maxupd, dtype=int)
        self.delta = np.zeros(self.maxupd)
        self.slot = {}
        self.y0 = self.lu.solve(self._rhs())
        self.x = self.y0.copy()
        self.dirty = False

    def update_junction(self, j, g_new):
        # Tell the solver that junction j now has conductance g_new.
        self.g[j] = g_new
        has_a = self.ia[j] >= 0
        has_b = self.ib[j] >= 0
        if (not has_a) and (not has_b):
            return      # both ends are electrodes: no effect on the unknowns
        if j not in self.slot:
            if self.p >= self.maxupd:
                # no free column: start again from the current conductances
                self.refactor()
                return
            v = np.zeros(self.ndof)
            if has_a:
                v[self.ia[j]] = v[self.ia[j]] + 1.0
            if has_b:
                v[self.ib[j]] = v[self.ib[j]] - 1.0
            self.W[:, self.p] = self.lu.solve(v)       # one backsolve
            self.col_junction[self.p] = j
            self.slot[j] = self.p
            self.p = self.p + 1
        self.delta[self.slot[j]] = self.g[j] - self.g_ref[j]
        if self.touches_b[j]:
            # an electrode junction changed: b changed, so y0 must be redone
            self.y0 = self.lu.solve(self._rhs())
        self.dirty = True

    def solve(self):
        # Voltage of ALL the nodes for the current conductances.
        if self.dirty:
            p = self.p
            if p == 0:
                self.x = self.y0.copy()
            else:
                cj = self.col_junction[:p]
                ia_c = self.ia[cj]
                ib_c = self.ib[cj]
                has_a = (ia_c >= 0).astype(float)
                has_b = (ib_c >= 0).astype(float)
                Wp = self.W[:, :p]
                # V^T W (p x p) and V^T y0 (p): rows of W / y0 at the ends of
                # each changed junction (zero where the end is an electrode)
                VtW = Wp[ia_c] * has_a[:, None] - Wp[ib_c] * has_b[:, None]
                Vty = self.y0[ia_c] * has_a - self.y0[ib_c] * has_b
                C = np.diag(1.0 / self.delta[:p]) + VtW
                z = np.linalg.solve(C, Vty)
                self.x = self.y0 - Wp @ z
            self.dirty = False
        voltage = np.empty(self.n_nodes)
        voltage[self.unknown] = self.x
        voltage[self.known] = self.v_known
        return voltage
