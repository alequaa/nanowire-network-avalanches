# test_sksparse_update.py
# Small test to understand how the update of the Cholesky factor works in
# scikit-sparse 0.5.1. It does NOT use the nanowire networks yet.
#
# What it does:
#   1) builds a small symmetric positive definite sparse matrix A
#      (30 x 30, like a lattice with one node tied to ground);
#   2) factorises A with ldl_factor (the update works on LDL^T);
#   3) changes ONE junction between the nodes i and j (conductance +delta):
#         A_new = A + delta * v v^T,   v = e_i - e_j
#      and updates the factor with f.update(C), where C = sqrt(delta) * v;
#   4) compares the solution after the update with the one obtained by
#      refactorising the new matrix from scratch.
# If the difference is ~1e-15 the update is correct.
# It also tries the different factorisation modes (supernodal_mode), to find
# out which ones work. Every try is protected by try/except: if one gives an
# error, it prints it and goes on with the next one.

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import sksparse
import sksparse.cholmod as cm

print("sksparse version:", sksparse.__version__)

# ------------------------------------------------------------------
# 1) Test matrix
# ------------------------------------------------------------------
n = 30                                    # number of unknown nodes
rng = np.random.default_rng(1)            # random generator with fixed seed

# We build a "graph": a chain 0-1-2-...-29 plus some random links.
# Every link is a junction with conductance g > 0.
edges = []
for k in range(n - 1):
    edges.append((k, k + 1))
for k in range(15):
    a = int(rng.integers(0, n))
    b = int(rng.integers(0, n))
    if a != b:
        edges.append((min(a, b), max(a, b)))
edges = sorted(set(edges))                # remove duplicates

g = rng.uniform(0.5, 2.0, size=len(edges))   # random conductances

# Matrix A = sum, over the junctions, of g * v v^T  (weighted Laplacian)
A = sp.lil_matrix((n, n))
for (a, b), gk in zip(edges, g):
    A[a, a] += gk
    A[b, b] += gk
    A[a, b] -= gk
    A[b, a] -= gk
# Add 1.0 on the diagonal of node 0 and node n-1: it is as if they were
# connected to an electrode. Without this A would be singular.
A[0, 0] += 1.0
A[n - 1, n - 1] += 1.0
A = sp.csc_array(A)                       # format required by sksparse

b = rng.uniform(0.0, 1.0, size=n)         # right hand side

# ------------------------------------------------------------------
# 2) Junction that changes: between nodes 3 and 12, conductance +delta
# ------------------------------------------------------------------
i = 3
j = 12
delta = 0.7
v = np.zeros(n)
v[i] = 1.0
v[j] = -1.0
# column C = sqrt(delta) * v, as a sparse (n x 1) matrix in csc format
C = sp.csc_array((np.sqrt(delta) * v).reshape(n, 1))

# Reference solution: refactorise the new matrix from scratch
A_new = A + sp.csc_array(delta * np.outer(v, v))
x_ref = spsolve(sp.csc_matrix(A_new), b)

# ------------------------------------------------------------------
# 3) Tries with the different modes
# ------------------------------------------------------------------
# We do not know in advance which values supernodal_mode accepts: try them all.
modes = ["default", "simplicial", "supernodal", "auto"]

for mode in modes:
    print("-" * 60)
    print("mode:", mode)
    try:
        if mode == "default":
            f = cm.ldl_factor(A)
        else:
            f = cm.ldl_factor(A, supernodal_mode=mode)
    except Exception as e:
        print("  ERROR in the factorisation:", type(e).__name__, e)
        continue

    # list of the public methods/attributes of the object (useful to see
    # whether the permutation exists, e.g. an attribute P or perm)
    if mode == "default":
        print("  public attributes:",
              [name for name in dir(f) if not name.startswith("_")])

    # solution before the update (must coincide with spsolve on A)
    x0 = f.solve(b)
    x0_ref = spsolve(sp.csc_matrix(A), b)
    print("  error before the update :", np.max(np.abs(x0 - x0_ref)))

    # update with C in the ORIGINAL node ordering
    try:
        f.update(C)
        x1 = f.solve(b)
        error = np.max(np.abs(x1 - x_ref))
        print("  error after the update  :", error)
        if error < 1e-10:
            print("  --> OK: C must be given in the original ordering")
        else:
            print("  --> does NOT match: maybe C must be permuted (see attributes)")
    except Exception as e:
        print("  ERROR in the update:", type(e).__name__, e)

print("-" * 60)
print("End of test.")
