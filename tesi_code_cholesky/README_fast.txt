tesi_code_fast - copy of the code with the fast Kirchhoff solver
(the old code stays untouched in your "old code" folder)

NEW files (do not exist in the old code):
  nw_solver_fast.py        fast Kirchhoff solver (factorise once in a while, correct in between)
  nw_loop_levels_fast.py   same loop as nw_loop_levels.py, but using the fast solver
  tests/test_solver_fast.py  checks against the original solver (about 1 minute)

CHANGED files (ONE line changed each, marked with "CHANGED" in the file):
  nw_threshold.py                 import of the loop
  scripts/run_tau_no_spanning.py  import of the loop

COPIED without any change from the old code:
  nanowire_network.py, nw_solver.py, nw_avalanches.py, nw_tau.py, nw_loop_levels.py

NOT included here (copy them from the old code if you need them):
  scripts/check_first_avalanche.py   (IMPORTED by run_tau_no_spanning.py: needed to run it)
  all the other scripts (plot_*, tau_without_largest.py, run_threshold.py, ...) and the other tests
  They read the caches or use the library and work the same.

Run the checks first:   python tests/test_solver_fast.py
Caches: new runs write cache_nospan_*.pkl next to the script. If you copy OLD caches into
scripts/, run_tau_no_spanning.py reuses their networks (computed with the old solver) and
only computes the new seeds: statistically equivalent, but not the same switch by switch.
