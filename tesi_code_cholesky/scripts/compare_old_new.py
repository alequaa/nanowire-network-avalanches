# compare_old_new.py
# Confronto tra il metodo VECCHIO (nw_loop_levels.py, risolve Kirchhoff da zero
# a ogni scatto) e il metodo NUOVO (nw_loop_levels_fast.py, aggiornamento di
# rango basso + rifattorizzazione ogni 50 giunzioni cambiate).
# Risponde a due domande del prof:
#   1. Si ottiene ESATTAMENTE lo stesso risultato?
#   2. Di quanto migliora la velocita'?
#
# Le reti sono le stesse di tutti gli altri script (seed 0, 1, 2, ...). Per ogni
# rete e per ogni V* si fa girare il loop vecchio e il loop nuovo alla STESSA
# tensione V = V* x Vth e si misura il tempo di ciascuno (un processo solo, senza
# multiprocessing, cosi' i tempi sono confrontabili).
#
# PARTE A - confronto a traiettorie indipendenti (vecchio contro nuovo):
#   - numero di scatti uguale? sequenza di giunzioni identica? tempi uguali?
#   - se la sequenza differisce: dopo quanti scatti si separa la prima volta
#   - numero di valanghe, dimensioni S uguali?
#   - livelli finali di tutte le giunzioni uguali?
#
# PARTE B - controllo "passo per passo" (solo per le prime SHADOW_NETS reti, e
#   solo per taglie <= SHADOW_MAX_WIRES, perche' costa come un run vecchio):
#   si segue la traiettoria del metodo NUOVO e a ogni passo si risolve ANCHE
#   con il solver vecchio sullo stesso stato. Si misura:
#   - la differenza massima tra le tensioni dei due solver (relativa a V);
#   - in quanti passi il vecchio avrebbe scelto una giunzione DIVERSA;
#   - in quei passi, di quanto (in secondi) il tempo di scatto della giunzione
#     scelta dal nuovo supera il minimo calcolato col solver vecchio.
#   Se questo scarto e' dell'ordine di 1e-15 s (rumore numerico), le due scelte
#   erano "pari" (stessa giunzione a meno di arrotondamenti): la sequenza del
#   nuovo e' una sequenza valida anche per il vecchio.
#
# PARTE C - tempi: secondi totali vecchio e nuovo, e rapporto (speed-up).
#   Il tempo comprende solo il loop (non la ricerca della soglia Vth).
#
# Esempio numerico (inventato): una rete ha 1000 scatti. Il vecchio impiega 40 s,
# il nuovo 5 s -> speed-up 8. Se le due sequenze sono identiche per i primi 640
# scatti e poi il nuovo sceglie un'altra giunzione, ma al passo 641 i due tempi
# di scatto differivano di 3e-16 s, le due scelte erano un pareggio numerico.
#
# Come lanciare (dalla cartella principale):
#   python scripts/compare_old_new.py 100:30:10 400:60:3 900:90:1
# Ogni argomento e'  fili:lato_um:numero_reti.  Default: 100:30:10 400:60:3
# Variabili d'ambiente opzionali:
#   K (default 10), RATIO (1000), FACTORS (default 1.05,1.8)
#   SHADOW_NETS (reti per taglia della parte B, default 3)
#   SHADOW_MAX_WIRES (taglia massima della parte B, default 400)
#   THR_OLD_MAX_WIRES (taglia massima per cui si confronta anche la soglia Vth
#      calcolata col loop vecchio, default 400)
# Attenzione ai tempi: il loop VECCHIO con K = 10 a 900 fili e' lentissimo
# (decine di minuti per rete e per V*). Per 900 fili usa 1 sola rete.
# PowerShell:  $env:K = 10

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nanowire_network as nn
import nw_threshold as thr
from nw_solver import solve_node_voltages, junction_drops
from nw_loop_levels import run_loop_levels as run_old
from nw_loop_levels_fast import run_loop_levels_fast as run_new
from nw_solver_fast import FastKirchhoff
from nw_avalanches import find_avalanches
from nw_tau import fit_tau

V_SET = 0.01
LAM_CRIT = 0.01
G_ON = 1.0
RATIO = float(os.environ.get("RATIO", "1000"))
G_OFF = G_ON / RATIO
K = int(os.environ.get("K", "10"))
FACTORS = [float(x) for x in os.environ.get("FACTORS", "1.05,1.8").split(",")]
SHADOW_NETS = int(os.environ.get("SHADOW_NETS", "3"))
SHADOW_MAX_WIRES = int(os.environ.get("SHADOW_MAX_WIRES", "400"))
THR_OLD_MAX_WIRES = int(os.environ.get("THR_OLD_MAX_WIRES", "400"))
TIE_TOLERANCE = 1.0e-9

if len(sys.argv) > 1:
    SIZE_SPECS = sys.argv[1:]
else:
    SIZE_SPECS = ["100:30:10", "400:60:3"]


def level_conductances():
    # Conduttanza di ogni livello: G_k = Goff * (Gon/Goff)^(k/K)
    level_g = np.zeros(K + 1, dtype=float)
    for k in range(K + 1):
        level_g[k] = G_OFF * (G_ON / G_OFF) ** (float(k) / K)
    return level_g


def decide(junctions, voltage, level, lam, step):
    # Stessa regola del loop: velocita' di riempimento, tempo di scatto di ogni
    # giunzione, vince il minimo; tra i pareggi vince l'indice piu' basso.
    # Restituisce (rate, t, i_star, dt); i_star = None se nessun tempo e' finito.
    drop = np.abs(junction_drops(junctions, voltage))
    movable = level < K
    rate = np.where(movable, np.maximum(drop - V_SET, 0.0), 0.0)
    t = np.full(junctions.shape[0], np.inf)
    positive = rate > 0.0
    next_threshold = (level + 1) * step
    t[positive] = (next_threshold[positive] - lam[positive]) / rate[positive]
    if not np.any(np.isfinite(t)):
        return rate, t, None, None
    dt = np.min(t)
    tied = np.nonzero(t <= dt + TIE_TOLERANCE * dt + 1.0e-15)[0]
    return rate, t, int(tied[0]), float(dt)


def shadow_check(net, s, d, v):
    # PARTE B: traiettoria del metodo nuovo, con in parallelo il solver vecchio
    # sullo stesso stato a ogni passo.
    junctions = net.junctions
    n_nodes = net.n_wires()
    m = junctions.shape[0]
    step = LAM_CRIT / K
    level_g = level_conductances()
    level = np.zeros(m, dtype=int)
    lam = np.zeros(m, dtype=float)
    solver = FastKirchhoff(n_nodes, junctions, s, d, v, level_g[level], 50)
    n_steps = 0
    max_dv = 0.0
    n_diff = 0
    max_gap = 0.0
    n_incons = 0
    clock = 0.0
    while True:
        v_fast = solver.solve()
        v_old = solve_node_voltages(n_nodes, junctions, level_g[level], s, d, v)
        dv = float(np.max(np.abs(v_fast - v_old))) / v
        if dv > max_dv:
            max_dv = dv
        rate_f, t_f, i_f, dt_f = decide(junctions, v_fast, level, lam, step)
        rate_o, t_o, i_o, dt_o = decide(junctions, v_old, level, lam, step)
        if i_f is None:
            # il nuovo si ferma: anche il vecchio deve fermarsi
            if i_o is not None:
                n_incons = n_incons + 1
            break
        if i_o is None:
            # il nuovo scatta ma il vecchio no: incoerenza vera
            n_incons = n_incons + 1
        else:
            if i_f != i_o:
                n_diff = n_diff + 1
                if np.isfinite(t_o[i_f]):
                    gap = float(t_o[i_f] - dt_o)
                    if gap > max_gap:
                        max_gap = gap
                else:
                    # la giunzione scelta dal nuovo non puo' scattare per il vecchio
                    n_incons = n_incons + 1
        # si prosegue con la scelta del metodo nuovo
        movable = level < K
        lam[movable] = lam[movable] + rate_f[movable] * dt_f
        clock = clock + dt_f
        level[i_f] = level[i_f] + 1
        lam[i_f] = level[i_f] * step
        solver.update_junction(i_f, level_g[level[i_f]])
        n_steps = n_steps + 1
    return n_steps, max_dv, n_diff, max_gap, n_incons, clock


def diagnose_divergence(net, s, d, v, res_o, res_n, div):
    # Spiega la PRIMA divergenza tra due sequenze. Si rifa' girare il solver
    # VECCHIO seguendo la sequenza comune (i primi div scatti uguali) e, al passo
    # div, si guardano i tempi di scatto calcolati: j_o = giunzione scelta dal
    # vecchio, j_n = giunzione scelta dal nuovo.
    # Restituisce (dt, gap, rel_gap): dt = minimo tempo di scatto al passo div,
    # gap = tempo di scatto di j_n meno dt (in secondi, >= 0), rel_gap = gap/clock.
    # Se gap e' dell'ordine di 1e-15 s, le due giunzioni erano "simultanee"
    # a meno di arrotondamenti: il pareggio e' deciso dal rumore numerico.
    junctions = net.junctions
    n_nodes = net.n_wires()
    m = junctions.shape[0]
    step = LAM_CRIT / K
    level_g = level_conductances()
    level = np.zeros(m, dtype=int)
    lam = np.zeros(m, dtype=float)
    clock = 0.0
    for i in range(div + 1):
        voltage = solve_node_voltages(n_nodes, junctions, level_g[level], s, d, v)
        rate, t, i_star, dt = decide(junctions, voltage, level, lam, step)
        if i == div:
            j_n = int(res_n["event_junction"][div])
            if i_star is None or not np.isfinite(t[j_n]):
                return None
            gap = float(t[j_n] - dt)
            return dt, gap, gap / max(clock, 1.0e-300), clock
        # si segue la sequenza comune (uguale nei due metodi fino a div - 1)
        j = int(res_o["event_junction"][i])
        movable = level < K
        lam[movable] = lam[movable] + rate[movable] * dt
        clock = clock + dt
        level[j] = level[j] + 1
        lam[j] = level[j] * step
    return None


def first_divergence(res_a, res_b):
    # Indice (da 0) del primo scatto in cui le due sequenze di giunzioni (o i
    # livelli raggiunti) differiscono. -1 se sono identiche (stessa lunghezza).
    ja = res_a["event_junction"]
    jb = res_b["event_junction"]
    n = min(ja.shape[0], jb.shape[0])
    for i in range(n):
        if ja[i] != jb[i]:
            return i
    if ja.shape[0] != jb.shape[0]:
        return n
    return -1


def sizes_of(res):
    # Dimensioni delle valanghe, un evento per episodio di crescita
    times = res["event_time"][res["event_new_episode"]]
    return np.asarray(find_avalanches(times)["S"], dtype=int)


if __name__ == "__main__":
    print("K = %d, rapporto Gon/Goff = %g, V* = %s" % (K, RATIO, FACTORS))
    for spec in SIZE_SPECS:
        parts = spec.split(":")
        n_wires = int(parts[0])
        side = float(parts[1])
        n_nets = int(parts[2])
        print("\n================ %d fili (%g x %g um), %d reti ================"
              % (n_wires, side, side, n_nets), flush=True)

        # ---- accumulatori per V* ----
        stats = {}
        for f in FACTORS:
            stats[f] = {"n_runs": 0, "same_seq": 0, "same_len": 0, "same_S": 0,
                        "same_levels": 0, "time_old": 0.0, "time_new": 0.0,
                        "events_old": 0, "events_new": 0, "div": [],
                        "S_old": [], "S_new": [], "max_time_rel": 0.0,
                        "max_level_diff": 0}
        thr_time_old = 0.0
        thr_time_new = 0.0
        thr_pairs = 0
        thr_max_diff = 0.0
        shadow_rows = []
        shadow_nets_done = 0
        diverged_factors = {}    # seed -> lista dei V* con sequenze diverse
        diag_rows = []

        for seed in range(n_nets):
            net = nn.generate_network(n_wires=n_wires, side=side, seed=seed)
            s, d = nn.point_electrodes(net)

            # ---- soglia Vth: sempre col metodo nuovo; col vecchio solo se piccola
            thr.run_loop_levels = run_new
            t0 = time.perf_counter()
            vth = thr.find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
            thr_time_new = thr_time_new + (time.perf_counter() - t0)
            if np.isnan(vth):
                print("  rete %d: nessun cammino, saltata" % seed, flush=True)
                continue
            if n_wires <= THR_OLD_MAX_WIRES:
                thr.run_loop_levels = run_old
                t0 = time.perf_counter()
                vth_old = thr.find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
                thr_time_old = thr_time_old + (time.perf_counter() - t0)
                thr.run_loop_levels = run_new
                thr_pairs = thr_pairs + 1
                diff = abs(vth_old - vth)
                if diff > thr_max_diff:
                    thr_max_diff = diff

            # ---- loop vecchio e nuovo alla stessa tensione ----
            for f in FACTORS:
                v = f * vth
                t0 = time.perf_counter()
                res_o = run_old(net.n_wires(), net.junctions, s, d, v,
                                V_SET, LAM_CRIT, G_OFF, G_ON, K)
                t_old = time.perf_counter() - t0
                t0 = time.perf_counter()
                res_n = run_new(net.n_wires(), net.junctions, s, d, v,
                                V_SET, LAM_CRIT, G_OFF, G_ON, K)
                t_new = time.perf_counter() - t0
                st = stats[f]
                st["n_runs"] = st["n_runs"] + 1
                st["time_old"] = st["time_old"] + t_old
                st["time_new"] = st["time_new"] + t_new
                n_o = res_o["event_junction"].shape[0]
                n_n = res_n["event_junction"].shape[0]
                st["events_old"] = st["events_old"] + n_o
                st["events_new"] = st["events_new"] + n_n
                if n_o == n_n:
                    st["same_len"] = st["same_len"] + 1
                div = first_divergence(res_o, res_n)
                if div < 0:
                    st["same_seq"] = st["same_seq"] + 1
                    # sequenza identica: confronto dei tempi (relativo)
                    rel = np.max(np.abs(res_o["event_time"] - res_n["event_time"]) /
                                 np.maximum(np.abs(res_o["event_time"]), 1.0e-300))
                    if rel > st["max_time_rel"]:
                        st["max_time_rel"] = float(rel)
                else:
                    st["div"].append((div, n_o))
                    if seed not in diverged_factors:
                        diverged_factors[seed] = []
                    diverged_factors[seed].append(f)
                    if n_wires <= SHADOW_MAX_WIRES:
                        if div < min(n_o, n_n):
                            diag = diagnose_divergence(net, s, d, v, res_o, res_n, div)
                            if diag is not None:
                                diag_rows.append((seed, f, div, n_o) + diag)
                lev_diff = int(np.sum(res_o["level"] != res_n["level"]))
                if lev_diff == 0:
                    st["same_levels"] = st["same_levels"] + 1
                if lev_diff > st["max_level_diff"]:
                    st["max_level_diff"] = lev_diff
                so = sizes_of(res_o)
                sn = sizes_of(res_n)
                if so.shape[0] == sn.shape[0]:
                    if np.all(so == sn):
                        st["same_S"] = st["same_S"] + 1
                st["S_old"].append(so)
                st["S_new"].append(sn)
            print("  rete %d fatta (Vth/Vset = %.2f)" % (seed, vth / V_SET), flush=True)

            # ---- parte B (passo per passo), solo per taglie piccole ----
            # Si controllano le prime SHADOW_NETS reti e, soprattutto, i casi in cui
            # il vecchio e il nuovo hanno dato sequenze DIVERSE (sono quelli da spiegare).
            if n_wires <= SHADOW_MAX_WIRES:
                todo = []
                for f in FACTORS:
                    first_nets = shadow_nets_done < SHADOW_NETS
                    was_diverged = False
                    if seed in diverged_factors:
                        if f in diverged_factors[seed]:
                            was_diverged = True
                    if first_nets or was_diverged:
                        todo.append((f, was_diverged))
                if shadow_nets_done < SHADOW_NETS:
                    shadow_nets_done = shadow_nets_done + 1
                for f, was_diverged in todo:
                    out = shadow_check(net, s, d, f * vth)
                    shadow_rows.append((seed, f, was_diverged) + out)
                    print("    controllo passo per passo: rete %d, V* = %.2f fatto"
                          % (seed, f), flush=True)

        # ---------------- stampa dei risultati di questa taglia ----------------
        print("\n--- PARTE A: vecchio contro nuovo, traiettorie indipendenti ---")
        print("   V*  | run | stessa lunghezza | sequenza identica | livelli finali uguali "
              "| valanghe S identiche | scatti totali (vecchio / nuovo)")
        for f in FACTORS:
            st = stats[f]
            n = st["n_runs"]
            if n == 0:
                continue
            print("  %.2f | %3d | %9d / %-3d    | %10d / %-3d     | %12d / %-3d       "
                  "| %13d / %-3d      | %d / %d" % (
                      f, n, st["same_len"], n, st["same_seq"], n, st["same_levels"], n,
                      st["same_S"], n, st["events_old"], st["events_new"]))
            if len(st["div"]) > 0:
                fr = [dv / float(max(tot, 1)) for dv, tot in st["div"]]
                print("        sequenze diverse: %d. Prima divergenza (frazione degli scatti): "
                      "mediana %.2f, minima %.2f. Livelli finali diversi: al massimo %d giunzioni"
                      % (len(st["div"]), np.median(fr), np.min(fr), st["max_level_diff"]))
            if st["same_seq"] > 0:
                print("        nelle sequenze identiche, errore relativo massimo sui tempi: %.1e"
                      % st["max_time_rel"])
            # statistiche aggregate
            so = np.concatenate(st["S_old"])
            sn = np.concatenate(st["S_new"])
            line = "        aggregato: valanghe %d / %d, S medio %.2f / %.2f, S max %d / %d" % (
                so.shape[0], sn.shape[0], so.mean(), sn.mean(), so.max(), sn.max())
            if so.shape[0] >= 100:
                line = line + ", tau (smin=2) %.3f / %.3f" % (
                    fit_tau(so, 2, int(so.max())), fit_tau(sn, 2, int(sn.max())))
            print(line)

        if thr_pairs > 0:
            print("\n  Soglia Vth calcolata col loop vecchio e col nuovo su %d reti: "
                  "differenza massima %.2e V (precisione della bisezione 2e-4 V)"
                  % (thr_pairs, thr_max_diff))

        if len(shadow_rows) > 0:
            print("\n--- PARTE B: controllo passo per passo (traiettoria del nuovo) ---")
            print("  rete   V*  | sequenza diversa in A? |  passi | diff. max tensioni (rel. a V) "
                  "| passi con scelta diversa | scarto max dei tempi (s) | incoerenze")
            for row in shadow_rows:
                if row[2]:
                    flag = "SI"
                else:
                    flag = "no"
                print("  %4d  %.2f |          %-3s           | %6d |       %.1e              "
                      "|        %5d           |        %.1e          |   %d" % (
                          row[0], row[1], flag, row[3], row[4], row[5], row[6], row[7]))
            print("  Scarto dei tempi: quanto, secondo il solver vecchio, il tempo di scatto della"
                  " giunzione scelta dal nuovo supera il minimo.")
            print("  Se e' dell'ordine di 1e-15 s le due scelte erano un pareggio numerico."
                  " Incoerenza = il nuovo sceglie una giunzione che per il vecchio non potrebbe"
                  " scattare, o i due si fermano in momenti diversi: deve essere 0.")

        if len(diag_rows) > 0:
            print("\n--- PARTE B2: perche' le sequenze si separano (prima divergenza) ---")
            print("  rete   V*  | scatto n. (su totale) | tempo minimo dt (s) | scarto della scelta "
                  "del nuovo (s) | scarto / clock")
            for row in diag_rows:
                print("  %4d  %.2f | %6d (su %5d)      |      %.2e       |        %.2e         "
                      "|   %.1e" % (row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
            print("  Scarto = di quanto, nello stato del solver vecchio, il tempo di scatto della "
                  "giunzione scelta dal nuovo supera il minimo. Un valore dell'ordine di 1e-15 s "
                  "(o minore) significa che le due giunzioni scattano 'insieme' a meno di arrotondamenti.")

        print("\n--- PARTE C: tempi (secondi, un processo) ---")
        print("   V*  | loop vecchio | loop nuovo | speed-up")
        tot_old = 0.0
        tot_new = 0.0
        for f in FACTORS:
            st = stats[f]
            if st["n_runs"] == 0:
                continue
            tot_old = tot_old + st["time_old"]
            tot_new = tot_new + st["time_new"]
            print("  %.2f | %10.1f s | %8.1f s | %6.1f x   (%d run, %.2f s / %.2f s per run)" % (
                f, st["time_old"], st["time_new"], st["time_old"] / st["time_new"],
                st["n_runs"], st["time_old"] / st["n_runs"], st["time_new"] / st["n_runs"]))
        if tot_new > 0.0:
            print("  tutti i V* insieme: %.1f s contro %.1f s, speed-up %.1f x"
                  % (tot_old, tot_new, tot_old / tot_new))
        if thr_pairs > 0 and thr_time_new > 0.0:
            print("  ricerca della soglia Vth (solo reti confrontate): vecchio %.1f s, nuovo %.1f s"
                  % (thr_time_old, thr_time_new))
        sys.stdout.flush()
