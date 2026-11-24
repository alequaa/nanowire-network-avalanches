# nw_loop_levels.py
# Event-driven loop at fixed external voltage. Each junction has K conductance
# LEVELS (K = 1: two states, OFF and ON).
#   level k = 0, 1, ..., K ; conductance G_k = Goff * (Gon/Goff)^(k/K)
#   (exponential growth, a crude version of the tunnelling regime)
# At every iteration: Kirchhoff, fill rate r_i = max(|V_i| - Vset, 0), time to
# the NEXT level t_i = ((k_i + 1) * Lcrit / K - L_i) / r_i, the smallest time
# wins, all buckets advance, that junction goes up ONE level.
# A junction at level K is ON (it does not move any more).
# Why levels: when a junction rises one level its own drop falls, so its fill
# rate falls too (it "waits" near Vset, as in Hochstetter's model).
#
# Growth EPISODES (Hochstetter counts one event per episode of rising
# conductance of a junction): an episode starts at the first level step of a
# junction and goes on while its drop stays above Vset (rate > 0). If at any
# loop iteration its rate is 0 (plateau), the episode is over; the next level
# step of that junction starts a NEW episode. event_new_episode[k] is True
# when step k starts a new episode.
# A plateau is counted only if the clock really advances while the rate is 0:
# junctions with the same switching time (ties) are processed one after the
# other with dt = 0, and that is NOT a pause.
#
# Example (K = 2, chain S-M-D, both junctions at level 0, V = 0.04):
#   drops 0.02 -> r = 0.01 each; level thresholds are Lcrit/2 = 0.005 and 0.01.
#   t0 = t1 = 0.005/0.01 = 0.5 (tie): j0 goes to level 1 first (lowest index).
#   Now G0 = Goff*sqrt(1000) = 0.0316 -> j0 drop falls, j1 drop rises:
#   j1 then reaches its level 1 shortly after, and so on.
#   Episode flags of this example: [True, True, False, False] (each junction
#   grows in one episode; the tie does not create a false pause).

import numpy as np

from nw_solver import solve_node_voltages, junction_drops

TIE_TOLERANCE = 1.0e-9


def run_loop_levels(n_nodes, junctions, sources, drains, v_ext,
                    v_set, lam_crit, g_off, g_on, n_levels,
                    max_steps=2000000):
    # Returns a dict: event_time, event_junction, event_level (level reached),
    # level (final level of every junction), is_on (level == K), lam, clock.
    m = junctions.shape[0]
    K = int(n_levels)
    step = lam_crit / K
    level = np.zeros(m, dtype=int)
    lam = np.zeros(m, dtype=float)
    interrupted = np.zeros(m, dtype=bool)   # plateau seen since last step
    event_new = []
    # conductance of each level
    level_g = np.zeros(K + 1, dtype=float)
    for k in range(K + 1):
        level_g[k] = g_off * (g_on / g_off) ** (float(k) / K)

    clock = 0.0
    event_time = []
    event_junction = []
    event_level = []
    n_steps = 0
    while n_steps < max_steps:
        n_steps = n_steps + 1
        g = level_g[level]
        voltage = solve_node_voltages(n_nodes, junctions, g, sources, drains, v_ext)
        drop = np.abs(junction_drops(junctions, voltage))
        movable = level < K
        rate = np.where(movable, np.maximum(drop - v_set, 0.0), 0.0)
        # a junction that has started to rise but now has rate 0 is on a plateau
        stalled = (rate <= 0.0) & movable & (level > 0)
        t = np.full(m, np.inf)
        positive = rate > 0.0
        next_threshold = (level + 1) * step
        t[positive] = (next_threshold[positive] - lam[positive]) / rate[positive]
        if not np.any(np.isfinite(t)):
            break
        dt = np.min(t)
        tied = np.nonzero(t <= dt + TIE_TOLERANCE * dt + 1.0e-15)[0]
        i_star = int(tied[0])
        # A plateau counts only if time really passes while the rate is 0.
        # If dt is (almost) zero, this switch is simultaneous with the previous
        # one (tie, processed one after the other): no real pause happened.
        if dt > TIE_TOLERANCE * clock + 1.0e-15:
            interrupted[stalled] = True
        lam[movable] = lam[movable] + rate[movable] * dt
        clock = clock + dt
        # new episode if first step of this junction, or it had a plateau
        if level[i_star] == 0:
            is_new = True
        else:
            is_new = bool(interrupted[i_star])
        interrupted[i_star] = False
        event_new.append(is_new)
        # junction i* goes up one level; its Lambda is set exactly
        level[i_star] = level[i_star] + 1
        lam[i_star] = level[i_star] * step
        event_time.append(clock)
        event_junction.append(i_star)
        event_level.append(int(level[i_star]))

    out = {}
    out["event_time"] = np.array(event_time, dtype=float)
    out["event_junction"] = np.array(event_junction, dtype=int)
    out["event_level"] = np.array(event_level, dtype=int)
    out["event_new_episode"] = np.array(event_new, dtype=bool)
    out["level"] = level
    out["is_on"] = level == K
    out["lam"] = lam
    out["clock"] = clock
    return out
