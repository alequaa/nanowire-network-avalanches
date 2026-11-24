# nw_solver_chol.py
# Kirchhoff solver with UPDATE OF THE CHOLESKY FACTOR (idea of Nukala and
# Simunovic, DOI 10.1088/0305-4470/36/45/004), using scikit-sparse (a wrapper
# of CHOLMOD). It has the SAME interface as nw_solver_fast.FastKirchhoff
# (methods update_junction and solve), so the loop does not change.
#
# Difference from the current fast solver (LU + Woodbury formula):
#   - now: A0 is factorised with LU every maxupd junctions; between two
#     factorisations the solution is corrected with the Woodbury formula
#     (small p x p matrix);
#   - here: A is factorised with LDL^T (Cholesky without square roots) and,
#     when ONE junction changes, the factor is updated DIRECTLY:
#         L' D' L'^T = A + delta * v v^T
#     (rank-1 update of CHOLMOD). Then every step needs only two triangular
#     solves (f.solve). No p x p matrix.
#
# Numerical example (chain S - M1 - M2 - D, all g = 1, V_S = 1):
#   A = [[2,-1],[-1,2]]. The junction M2-D goes from 1 to 3: delta = 2,
#   v = [0,1]. A_new = A + 2 v v^T = [[2,-1],[-1,4]], solution [4/7, 1/7].
#
# The conductances only GO UP (delta > 0): only the update is needed, never
# the downdate (which is the numerically delicate part). If for some reason
# delta < 0, the code refactorises from scratch.
#
# Checks done on the small test (test_sksparse_update.py): C must be given in
# the ORIGINAL node ordering; the factorisation modes default / simplicial /
# supernodal / auto all give the same result.

import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import ldl_factor

from nw_solver_fast import FastKirchhoff


class CholeskyKirchhoff(FastKirchhoff):
    # From FastKirchhoff it reuses the constructor (which prepares the node
    # indices, the electrode nodes and the right hand side) and _rhs().
    # It replaces only refactor, update_junction and solve.

    def __init__(self, n_nodes, junctions, sources, drains, v_ext, g,
                 maxupd=200, supernodal_mode=None):
        # maxupd = after how many updates we refactorise from scratch (to
        # keep rounding errors and fill-in of the factor under control).
        # supernodal_mode = None (library default) or
        # "simplicial", "supernodal", "auto".
        self.supernodal_mode = supernodal_mode
        self.n_updates_total = 0
        # the constructor of the parent class calls self.refactor() at the end
        FastKirchhoff.__init__(self, n_nodes, junctions, sources, drains,
                               v_ext, g, maxupd)

    def _build_matrix(self):
        # Matrix A (unknowns x unknowns) from the CURRENT conductances.
        # Same code as FastKirchhoff.refactor.
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
        A = sp.coo_matrix((vals, (rows, cols)),
                          shape=(self.ndof, self.ndof)).tocsc()
        # sksparse wants a csc_array
        return sp.csc_array(A)

    def refactor(self):
        A = self._build_matrix()
        if self.supernodal_mode is None:
            self.f = ldl_factor(A)
        else:
            self.f = ldl_factor(A, supernodal_mode=self.supernodal_mode)
        self.n_refactor = self.n_refactor + 1
        # conductances that the factor represents at this moment
        self.g_fact = self.g.copy()
        self.n_updates = 0
        self.x = self.f.solve(self._rhs())
        self.dirty = False

    def update_junction(self, j, g_new):
        # Tell the solver that junction j now has conductance g_new.
        self.g[j] = g_new
        has_a = self.ia[j] >= 0
        has_b = self.ib[j] >= 0
        if (not has_a) and (not has_b):
            return      # both ends are electrodes: no effect on the unknowns
        delta = g_new - self.g_fact[j]
        if delta < 0.0:
            # should not happen (the conductances only go up): to be safe we
            # refactorise from scratch instead of doing a downdate
            self.refactor()
            self.dirty = True
            return
        if delta > 0.0:
            # column C = sqrt(delta) * v, with v = +1 on end a and -1 on
            # end b (only for the ends that are unknowns)
            rows = []
            vals = []
            root = np.sqrt(delta)
            if has_a:
                rows.append(int(self.ia[j]))
                vals.append(root)
            if has_b:
                rows.append(int(self.ib[j]))
                vals.append(-root)
            cols = [0] * len(rows)
            C = sp.csc_array((np.array(vals), (np.array(rows), np.array(cols))),
                             shape=(self.ndof, 1))
            self.f.update(C)
            self.g_fact[j] = g_new
            self.n_updates = self.n_updates + 1
            self.n_updates_total = self.n_updates_total + 1
        self.dirty = True
        if self.n_updates >= self.maxupd:
            # start again from scratch from the current conductances
            self.refactor()
            self.dirty = True

    def solve(self):
        # Voltage of ALL the nodes for the current conductances.
        if self.dirty:
            # the right hand side must be rebuilt: it depends on the
            # conductances of the junctions attached to the electrode, which
            # may have changed
            self.x = self.f.solve(self._rhs())
            self.dirty = False
        voltage = np.empty(self.n_nodes)
        voltage[self.unknown] = self.x
        voltage[self.known] = self.v_known
        return voltage
