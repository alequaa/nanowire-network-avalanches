# check_old_vs_fast.py
# Le prime N_OLD reti della cache (seed 0..N_OLD-1) sono state calcolate col
# solver ORIGINALE; le altre col solver VELOCE. Qui confrontiamo i due gruppi
# di reti, a ogni V*, sulla quota di eventi che sta nella valanga del cammino
# (spanning) e sul numero totale di eventi. Nessuna simulazione.
#
# Esempio numerico: una rete ha valanghe S = [3, 1, 20, 2], la valanga del
# cammino e' quella da 20 -> quota = 20 / 26 = 0.77.
#
# Cosa guardare: se i due gruppi hanno medie e mediane vicine (dentro il
# rumore), il solver veloce non cambia la fisica. Il rumore e' stimato con
# bootstrap sulle reti (barra = 1 deviazione standard).
#
# Come lanciare (dalla cartella principale, script dentro scripts/):
#   python scripts/check_old_vs_fast.py
# Variabili: K (default 10), WIRES (900), SIDE (90), RATIO (1000),
# NOLD (reti calcolate col solver originale, default 100).

import os
import pickle

import numpy as np

RATIO = float(os.environ.get("RATIO", "1000"))
K = int(os.environ.get("K", "10"))
WIRES = int(os.environ.get("WIRES", "900"))
SIDE = float(os.environ.get("SIDE", "90"))
N_OLD = int(os.environ.get("NOLD", "100"))
VSTARS = [1.0, 1.05, 1.2, 1.4, 1.8]
FOLDER = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 1000


def find_cache():
    # Cache con piu' reti per questa taglia.
    prefix = "cache_nospan_r%g_K%d_w%d_L%g_n" % (RATIO, K, WIRES, SIDE)
    best_n = 0
    best_file = None
    for fname in os.listdir(FOLDER):
        if fname.startswith(prefix) and fname.endswith(".pkl"):
            n = int(fname[len(prefix):-4])
            if n > best_n:
                best_n = n
                best_file = fname
    return best_file


def shares_and_totals(results, f):
    # Per ogni rete con cammino: quota di eventi nella valanga del cammino e
    # numero totale di eventi (somma delle S).
    shares = []
    totals = []
    for r in results:
        if f in r[2]:
            rec = r[2][f]
            if rec[2]:
                total = float(np.sum(rec[0]))
                shares.append(rec[3] / total)
                totals.append(total)
    return np.array(shares), np.array(totals)


def boot_std(values, func, rng):
    # Deviazione standard bootstrap di func(values).
    n = values.shape[0]
    out = []
    for b in range(N_BOOT):
        pick = rng.integers(0, n, size=n)
        out.append(func(values[pick]))
    return float(np.std(out))


if __name__ == "__main__":
    fname = find_cache()
    if fname is None:
        raise SystemExit("nessuna cache trovata per questa taglia")
    with open(os.path.join(FOLDER, fname), "rb") as fh:
        results = pickle.load(fh)
    old = results[:N_OLD]
    new = results[N_OLD:]
    print("cache %s: %d reti vecchie (seed 0..%d) contro %d reti nuove" %
          (fname, len(old), N_OLD - 1, len(new)))
    rng = np.random.default_rng(0)
    print("\n V*   | gruppo | reti con cammino | quota mediana (+-)  | quota media (+-)   "
          "| frazione quota>0.5 | eventi totali medi")
    for f in VSTARS:
        for name, group in (("vecchie", old), ("nuove", new)):
            sh, tot = shares_and_totals(group, f)
            if sh.shape[0] < 5:
                print(" %.2f | %-7s | %4d | troppo poche" % (f, name, sh.shape[0]))
                continue
            print(" %.2f | %-7s | %4d | %5.2f +- %.2f        | %5.2f +- %.2f       | %5.2f"
                  "              | %8.0f" % (
                      f, name, sh.shape[0],
                      np.median(sh), boot_std(sh, np.median, rng),
                      np.mean(sh), boot_std(sh, np.mean, rng),
                      np.mean(sh > 0.5), np.mean(tot)))
