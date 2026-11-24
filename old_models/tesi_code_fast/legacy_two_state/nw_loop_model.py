# nw_loop_model.py
# Loop at fixed external voltage:
#   1. Solve Kirchhoff, get the drop |V_i| on every OFF junction.
#   2. Fill rate:   r_i = max(|V_i| - Vset, 0)
#   3. Switch time: t_i = (Lcrit - L_i) / r_i   (infinite if r_i = 0)
#   4. If all t_i are infinite: stop (steady state).
#   5. i* = argmin t_i, dt = t_i*. Every bucket: L_j += r_j * dt. Clock += dt.
#   6. Switch i* to g_on, record (clock, i*), go back to 1.
# After switching, a junction stays ON (no decay of Lambda below Vreset, which
# Hochstetter's model has).
# Ties (equal times up to roundoff) are broken by the lowest junction index,
# so that the result is the same on every computer.
#
# Example: chain S-M-D, junctions j0 (S-M), j1 (M-D), both OFF.
# V_ext = 0.04, Vset = 0.01, Lcrit = 0.01, start L = (0, 0.005).
# Drops 0.02 each -> r = 0.01 each -> t0 = 1.0, t1 = 0.5. j1 switches at 0.5.
# Then the drop on j0 is 0.04 * g_on/(g_on+g_off) = 0.03996, r0 = 0.02996,
# t0 = (0.01 - 0.005)/0.02996 = 0.1669 -> j0 switches at clock 0.6669.

import numpy as np

from nw_solver import solve_node_voltages, junction_drops, total_current

# Two switching times closer than this (relative) are treated as a tie.
TIE_TOLERANCE = 1.0e-9


def run_loop(n_nodes, junctions, sources, drains, v_ext,
             v_set, lam_crit, g_off, g_on, lam_start=None,
             is_on_start=None, clock_start=0.0):
    # Runs the loop until all switching times are infinite.
    # lam_start: optional initial Lambda per junction (default: all zero).
    # is_on_start: optional initial ON/OFF state (default: all OFF);
    # clock_start: initial clock. Used to continue a run at a higher voltage.
    # Returns a dict:
    #   event_time, event_junction : clock and index of every switch
    #   current_history : total current before each switch, plus the final one
    #   is_on, lam, clock : final state
    #   stop_reason : why the loop stopped
    m = junctions.shape[0]

    if is_on_start is None:
        is_on = np.zeros(m, dtype=bool)
    else:
        is_on = np.array(is_on_start, dtype=bool)
    g = np.full(m, g_off, dtype=float)
    g[is_on] = g_on

    if lam_start is None:
        lam = np.zeros(m, dtype=float)
    else:
        lam = np.array(lam_start, dtype=float)

    clock = float(clock_start)
    event_time = []
    event_junction = []
    current_history = []

    while True:
        # Step 1: Kirchhoff and drops
        voltage = solve_node_voltages(n_nodes, junctions, g, sources, drains, v_ext)
        drop = junction_drops(junctions, voltage)
        current_history.append(total_current(n_nodes, junctions, g, voltage, sources))

        # Step 2: fill rates
        rate = np.maximum(drop - v_set, 0.0)

        # Step 3: switch times (finite only for OFF junctions with r > 0)
        t = np.full(m, np.inf)
        candidates = (~is_on) & (rate > 0.0)
        t[candidates] = (lam_crit - lam[candidates]) / rate[candidates]
        # Rounding can give -1e-18: treat as 0 (bucket already full).
        t[candidates] = np.maximum(t[candidates], 0.0)

        # Step 4: stop if every time is infinite
        if not np.any(np.isfinite(t)):
            if np.all(is_on):
                reason = "all junctions are ON"
            else:
                reason = "steady state: no OFF junction above Vset"
            break

        # Step 5: first junction to switch, advance all OFF buckets.
        # Junctions whose times agree up to roundoff (relative 1e-9) count as
        # TIED, and the tie is broken by the lowest junction index. Without
        # this, the last bit of the solver result decides who goes first, and
        # the event order changes from one computer to another.
        dt = np.min(t)
        tied = np.nonzero(t <= dt + TIE_TOLERANCE * dt + 1.0e-15)[0]
        i_star = int(tied[0])
        off = ~is_on
        lam[off] = lam[off] + rate[off] * dt
        clock = clock + dt

        # Step 6: switch i* (Lambda set exactly to Lcrit, avoids rounding)
        lam[i_star] = lam_crit
        g[i_star] = g_on
        is_on[i_star] = True
        event_time.append(clock)
        event_junction.append(i_star)

    result = {}
    result["event_time"] = np.array(event_time)
    result["event_junction"] = np.array(event_junction, dtype=int)
    result["current_history"] = np.array(current_history)
    result["is_on"] = is_on
    result["lam"] = lam
    result["clock"] = clock
    result["stop_reason"] = reason
    return result
