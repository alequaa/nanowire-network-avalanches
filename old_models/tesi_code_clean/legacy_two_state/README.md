# legacy_two_state

Earlier work with the two-state junction (each junction is either OFF, with
conductance Goff, or ON, with conductance Gon, no intermediate levels) and
with other ways to define an avalanche.

The current model (main folder) does not need anything in here, except that
tests/test_loop_levels.py uses nw_loop_model.py as the reference for K = 1.

These scripts are kept so that the numbers of the earlier results can be
reproduced. They are frozen: they import the shared files (nanowire_network.py,
nw_solver.py, nw_avalanches.py, nw_tau.py) from the main folder.

## Ways to define an avalanche used here

- Definition C (voltage ramp): the external voltage is raised in small steps;
  the switches of one voltage step form one avalanche.
- Definition D (fixed voltage V* = V / Vth): each network is run at a fixed
  voltage proportional to its own threshold Vth; avalanches are found with the
  frames of nw_avalanches.py. In the main folder, scripts/run_tau_levels.py
  does this with K levels.

## Files

| File | What it does |
|---|---|
| nw_loop_model.py | the loop with two states |
| nw_ramp.py | definition C: voltage ramp, one avalanche per voltage step |
| nw_tau_extra.py | range rule of Hochstetter with KS p-value; also smin chosen by KS distance and cutoff fit |
| check_vth.py | n * Vset against the ramp voltage that closes the first path |
| run_tau_ramp.py | tau, definition C (ramp) |
| run_tau_D.py | tau at V* = V / Vth, two-state loop |
| plot_loop_model.py | three figures of one run (seed 7) |
| plot_many_networks.py | 20 networks at the same voltage |
| test_loop_model_two_state.py | tests of nw_loop_model.py and nw_ramp.py |

Run them from this folder, for example:

    python run_tau_D.py

Figures go in an images folder created next to the script.