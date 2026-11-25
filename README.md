# Nanowire network avalanches

Python simulation of a neuromorphic network of nanowires: a random network of
nanowires whose junctions switch from a low to a high conductance when the
voltage across them is high enough. The code solves the Kirchhoff equations
after every switch and studies the conductance of the network, the electrical
avalanches and the geometric percolation.

Bachelor thesis in Artificial Intelligence (University of Pavia), advisor
Prof. Stefano Zapperi (University of Milan). The model follows the network
model of Hochstetter et al., Nature Communications 12, 4008 (2021),
DOI 10.1038/s41467-021-24260-z, in a simpler and quasi-static form.

## Repository structure

| Folder | What it contains |
|---|---|
| tesi_code_cholesky | The current version of the code (start here). |
| old_models/tesi_code_fast | The previous version: fast Kirchhoff solver (LU factorisation with a correction formula) and fast loop. |
| old_models/tesi_code_clean | The first clean version: standard solver, loop with K conductance levels, scripts for the avalanche analysis with K = 1 and K = 10, and the folder legacy_two_state (old model with two conductance values). |

Each folder is a complete copy of the code of that moment: its files import
each other, so run the scripts from inside the folder you want to use. The
old versions are kept only to reproduce old results.

## The model in short

- Network: nanowires with gamma-distributed lengths (mean 10 um, standard
  deviation 1 um), uniform random angles and centres, in a square box. By
  default 100 wires in 30 x 30 um. The largest connected component is kept,
  so the final number of wires can be a little smaller than requested.
- Electrodes: point electrodes (the wire with the highest centre is the
  source, the one with the lowest centre is the drain).
- Dynamics (event-driven, fixed external voltage): at every step Kirchhoff
  gives the voltage drop on every junction; each junction accumulates a
  variable Lambda at a rate proportional to (drop - Vset) when the drop is
  above Vset; the junction that reaches its threshold first goes up by one
  conductance level; then Kirchhoff is solved again.
- Conductance levels: K + 1 levels from Goff to Gon, growing exponentially.
  K = 1 is the two-state loop (OFF and ON).
- Parameters: Vset = 0.01 V, Lambda_crit = 0.01 V s, Gon/Goff = 1000.
- Threshold voltage Vth of a network: the lowest fixed voltage at which a
  path of ON junctions connects source and drain (found by bisection).
  V* = V / Vth.
- Avalanches: frame method (as in Hochstetter et al.): the frame width is the
  mean time between events; an avalanche is a run of non-empty frames.

## Current version: tesi_code_cholesky

Library files (imported by the scripts):

- nanowire_network.py: random network, crossings, electrodes.
- nw_solver.py: Kirchhoff solver (standard, sparse).
- nw_solver_fast.py: faster solver (LU factorisation with a correction
  formula between two factorisations).
- nw_solver_chol.py: solver with direct update of the Cholesky factor, using
  scikit-sparse (optional; needed only with SOLVER=chol).
- nw_loop_levels.py, nw_loop_levels_fast.py, nw_loop_levels_chol.py: the loop
  with K levels with the three solvers. The header of each file explains the
  differences.
- nw_current.py: total current and conductance from the node voltages.
- nw_avalanches.py: avalanches with the frame method.
- nw_tau.py: maximum-likelihood fit of a power law for the avalanche sizes.
- nw_threshold.py: threshold voltage Vth by bisection.

Scripts (folder scripts):

- run_conductance_curves.py: simulates many networks at many values of V*
  and saves the conductance G(t) after every switch and several measures of
  the conductance jump (single switch that closes the first ON path, whole
  avalanche that contains it, largest avalanche jump, fraction of networks
  with a path, final conductance). Here V = V* x Vth of the same network.
- plot_conductance_curves.py: figures from the saved data (jump against V*
  and G(t)).
- run_conductance_curves_common.py and plot_conductance_curves_common.py: the
  same, but with a common threshold Vref(L) for all the networks of one size
  (mean of their Vth). It needs the data of the previous script for the same
  size.
- vth_statistics.py: table and histograms of the thresholds Vth.
- make_tables.py: tables of the results in csv (folder scripts/tables).
- run_1000_own.bat, run_1000_common.bat: Windows batch files that run the
  1000-wire case for K = 1 and K = 10.
- compare_lu_chol.py, test_sksparse_update.py: speed comparison of the solvers
  and a small test of the Cholesky update.
- Other scripts (avalanche analysis and comparisons with the previous
  versions) are described in the comment at the top of each file.

Tests (folder tests): test_nanowire_network.py, test_solver_avalanches_tau.py,
test_loop_levels.py, test_solver_fast.py, test_solver_chol.py,
test_conductance.py. Run each one with python from the folder
tesi_code_cholesky.

### How to run

From the folder tesi_code_cholesky. Every argument is
wires:side_in_um:number_of_networks; the side grows as 3 x sqrt(wires)
(30, 60, 90, 95 and 120 um for 100, 400, 900, 1000 and 1600 wires).

    python scripts/run_conductance_curves.py 100:30:100 400:60:100

Environment variables (Windows cmd: set K=1; PowerShell: $env:K = 1):

- K: number of conductance steps (default 10)
- RATIO: Gon/Goff (default 1000)
- FACTORS: list of V* separated by commas
- PROCS: number of processes (more processes need more RAM)
- SOLVER: lu (default) or chol (Cholesky, needs scikit-sparse)
- VTH_MODE: own (default) or common

The data are saved in a pickle file next to the script; if a run is stopped,
start it again with the same arguments and it goes on from the networks
already done. The pickle files and the images are not in the repository
(see .gitignore): they are generated by the scripts.

### Requirements

Python 3.11, numpy, scipy, matplotlib. scikit-sparse is needed only for the
Cholesky solver (SOLVER=chol).

## Notes

- The results depend on the choice of the threshold used to define V*: see
  the two modes of run_conductance_curves.py (own and common).
- The commands in the comments of the scripts are written for the Miniforge
  Prompt on Windows.
