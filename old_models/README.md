# Nanowire network avalanches

Python simulation of a random network of nanowires with threshold junctions,
used to study electrical avalanches (bachelor thesis, Bachelor of Artificial
Intelligence, University of Pavia; supervisor: Prof. Stefano Zapperi,
University of Milan).

The goal is to find the key ingredients that give the avalanche size
distribution of Hochstetter et al. (Nat. Commun. 12, 4008,
DOI 10.1038/s41467-021-24260-z) with a much simpler model: an order of
switching, some disorder, and Kirchhoff's laws.

## The model in short

- **Network.** Random wires (gamma-distributed lengths, mean 10 um, std 1 um,
  uniform centres and angles) in a square of 30 x 30 um; both numbers are 
  parameters of generate_network (n_wires, side). Two wires that cross
  make a junction. Only the largest connected component is kept.
- **Electrodes.** One source wire (highest centre) and one drain wire (lowest
  centre).
- **Junction.** K conductance steps between OFF (Goff) and ON (Gon):
  G_k = Goff * (Gon/Goff)^(k/K), k = 0 ... K. K = 1 is the two-state junction
  (only OFF or ON).
- **Loop at fixed external voltage** (quasi-static, event driven):
  1. solve Kirchhoff for the voltage drop |V_i| on every junction;
  2. fill rate r_i = max(|V_i| - Vset, 0);
  3. time to the next level of each junction; the smallest time wins;
  4. all junctions advance their internal variable Lambda by r_i * dt, the
     winner goes up one level; go back to 1 (stop when all times are infinite).
- **Avalanches.** Frames as wide as the mean interval between events; an
  avalanche is a run of non-empty frames between two empty frames; size S =
  number of events. The exponent tau of P(S) ~ S^(-tau) is estimated by maximum
  likelihood, with a bootstrap over networks for the error.
- **Voltage.** V* = V / Vth, with Vth of each network found by bisection
  (smallest fixed voltage at which the ON junctions connect source and drain).

Parameters (Hochstetter): Vset = 0.01 V, Lambda_crit = 0.01 V s,
Gon/Goff = 1000, G0 = 1/(12.9 kOhm).

## Files

| File | What it does |
|---|---|
| nanowire_network.py | generates the network, junctions, electrodes, plot |
| nw_solver.py | Kirchhoff solver (voltages, drops, currents) |
| nw_loop_levels.py | the loop with K levels (K = 1: two states) |
| nw_avalanches.py | avalanches from the list of event times |
| nw_tau.py | tau by maximum likelihood, bootstrap over networks |
| nw_threshold.py | shortest path, ON path, threshold Vth by bisection |
| scripts/run_threshold.py | Vth against Hochstetter's n * Vset, for K = 1, 3, 10 |
| scripts/run_tau_levels.py | tau for several K, V* and ways of counting events |
| scripts/plot_model_overview.py | one figure with three panels: a network after the loop, its current, the avalanche size distribution |
| tests/test_nanowire_network.py | tests of the network generator |
| tests/test_solver_avalanches_tau.py | tests of solver, avalanches, tau fit |
| tests/test_loop_levels.py | tests of the multi-level loop |
| legacy_two_state/ | earlier two-state scripts (see its README) |

## How to run

Python 3.11, numpy, scipy, matplotlib (`pip install -r requirements.txt`).

Run every command from the main folder.

```
python tests/test_nanowire_network.py
python tests/test_solver_avalanches_tau.py
python tests/test_loop_levels.py

python scripts/plot_model_overview.py   # a few minutes
python scripts/run_threshold.py         # about 3 minutes
python scripts/run_tau_levels.py        # ratio 1000, K = 1, 3, 10: 20-30 minutes
python scripts/run_tau_levels.py 4 10   # ratio 4, only K = 10
```

`run_tau_levels.py` and `plot_model_overview.py` use 2 processes and save
their results in cache_*.pkl files (delete the file to repeat a run).
`run_tau_levels.py` uses 200 networks by default and `plot_model_overview.py`
40; set the environment variable NNET for a shorter run
(PowerShell: `$env:NNET = 10`). Figures go in an images folder next to the
script that makes them.

## Known limits and open points

- K = 10 is a choice without a physical justification. With K = 1 the
  threshold is about n * Vset / 2 (n = junctions on the shortest path); with
  K = 10 it is about n * Vset, as in Hochstetter.
- No decay of Lambda below Vreset (Hochstetter has it).
- Electrodes: highest and lowest wire, not the two corners of the paper.
- Junctions with exactly the same switching time (ties) are processed one
  after the other, the lowest index first, so that the result is the same on
  every computer. This does not split growth episodes (a pause counts only if
  the clock really advances).
- tau is not stable when smin grows: no clean power law yet.
- Only small networks (about 100 wires) have been run with K > 1.

## References

- J. Hochstetter et al., Nat. Commun. 12, 4008 (2021),
  DOI 10.1038/s41467-021-24260-z
- S. Shekhawat, S. Papanikolaou, S. Zapperi, J. P. Sethna, Phys. Rev. Lett.
  107, 276401 (2011), DOI 10.1103/PhysRevLett.107.276401
- F. C. Sheldon, M. Di Ventra, Phys. Rev. E 95, 012305 (2017),
  DOI 10.1103/PhysRevE.95.012305
