# run_conductance_curves.py
# Runs the loop on many networks at many values of V* = V / Vth and SAVES the
# conductance of the network G(t) after every switch, plus a few numbers that
# describe the "jump" of G. The data are written to a cache file (pickle) so
# that plot_conductance_curves.py can draw the curves without running again.
#
# What is saved, for every network and every V*:
#   state_time, state_conductance : time and G = I / V_ext of every state
#                                   (state 0 = before any switch, state k =
#                                   after switch k; G in units of Gon)
#   path_event   : index of the switch that closes the FIRST path of ON
#                  junctions (level K) between a source and a drain
#                  (-1 if no path forms)
#   jump_path    : G after that switch minus G before it (0.0 if no path)
#   jump_path_ratio : G after / G before that switch (nan if no path)
#   dg_max       : the largest increase of G in ONE single switch
#   dg_max_event : index of that switch
#   jump_path_aval : increase of G over the whole AVALANCHE that contains the
#                  path-closing switch (0.0 if no path forms)
#   dg_aval_max  : the largest increase of G over one whole avalanche
#   event_new_episode : flag of every switch (True = it starts an episode)
#   G_start, G_end, n_events
# Several definitions of the jump are saved (a single switch, or a whole
# avalanche), together with the full curves, so the choice can be made later
# without running again. Avalanches are built with the frame method of
# nw_avalanches.py: frame width = mean interval between the episode starts;
# here ALL the switches are put in the frames, and an avalanche is a run of
# non-empty frames. With K = 10 the closing of the path is spread over many
# small switches, so the avalanche measure can be much bigger than dg_max.
#
# Numerical example: G goes 0.001, 0.0011, 0.0012, 0.30, 0.31 and the switch
# number 2 (from 0.0012 to 0.30) closes the first ON path. Then
# path_event = 2, jump_path = 0.30 - 0.0012 = 0.2988, dg_max = 0.2988 as well.
#
# How to run (from the main folder, in the Miniforge Prompt):
#   python scripts/run_conductance_curves.py 100:30:100 400:60:100 900:90:10
# Every argument is  wires:side_um:number_of_networks.
# Environment variables (cmd: set K=1   PowerShell: $env:K = 1):
#   K        number of conductance levels (default 10)
#   RATIO    Gon/Goff (default 1000)
#   FACTORS  list of V* separated by commas
#   PROCS    number of processes (default 1; more processes use more RAM)
#   SOLVER   "lu" (default, fast LU) or "chol" (Cholesky, needs scikit-sparse)
# The cache file is saved in the folder of this script and its name contains a
# short code of the list FACTORS. If the run is interrupted, run it again with
# the same arguments: the networks already done are skipped. If you ask for
# more networks than before, only the new ones are simulated.
# THIS COPY of run_conductance_curves.py uses by DEFAULT the mode "common".
# The original run_conductance_curves.py is unchanged in behaviour (default
# "own"). Run the original first: the "common" mode needs its caches (Vth).
# TWO WAYS TO DEFINE V* (environment variable VTH_MODE):
#   own    V = V* x Vth of THE SAME network (Vth by bisection).
#   common (default of this copy) V = V* x Vref(L), where Vref(L) is the MEAN of the Vth of the
#          networks of that size, so that all the networks of one size are run
#          at exactly the same voltages. Vref is read from the caches of the
#          "own" mode (run that mode first), and is saved in the new cache, so
#          that a restarted run uses the same Vref. The files of this mode are
#          called conductance_common_*.pkl and never mix with the others.
#          Its default V* list is wider (0.5 ... 2.5) because the thresholds
#          of the single networks are spread around the mean.
# The threshold Vth of a network is taken from an old cache of
# run_tau_no_spanning.py (cache_nospan_*.pkl) when it exists; otherwise it is
# computed by bisection (slow for 900 wires).

import os
import sys
import glob
import time
import pickle
import hashlib
from multiprocessing import Pool

import numpy as np

# The library files are in the main folder, one level up.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import nanowire_network as nn
import nw_threshold as thr
from nw_threshold import on_path_exists

# Parameters are read at the top of the file, so that the child processes on
# Windows see them too.
V_SET = 0.01
LAM_CRIT = 0.01
G_ON = 1.0
RATIO = float(os.environ.get("RATIO", "1000"))
G_OFF = G_ON / RATIO
K = int(os.environ.get("K", "10"))
DEFAULT_FACTORS = "0.9,0.95,1.0,1.02,1.04,1.06,1.08,1.1,1.15,1.2,1.3,1.5,1.8"
DEFAULT_FACTORS_COMMON = "0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.6,1.8,2.0,2.5"
VTH_MODE = os.environ.get("VTH_MODE", "common")
if VTH_MODE == "common":
    FACTORS = [float(x) for x in os.environ.get("FACTORS", DEFAULT_FACTORS_COMMON).split(",")]
else:
    FACTORS = [float(x) for x in os.environ.get("FACTORS", DEFAULT_FACTORS).split(",")]
PROCS = int(os.environ.get("PROCS", "1"))
SOLVER = os.environ.get("SOLVER", "lu")

if SOLVER == "chol":
    from nw_loop_levels_chol import run_loop_levels_chol as run_loop
else:
    from nw_loop_levels_fast import run_loop_levels_fast as run_loop

if len(sys.argv) > 1:
    SIZE_SPECS = sys.argv[1:]
else:
    SIZE_SPECS = ["100:30:10"]


def factors_code():
    # Short code of the list of V*: it goes in the file name, so that caches
    # made with different lists are never mixed.
    text = ",".join(["%.4f" % f for f in FACTORS])
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:6]


def cache_name(n_wires, side):
    prefix = "conductance"
    if VTH_MODE == "common":
        prefix = "conductance_common"
    return os.path.join(HERE, "%s_r%d_K%d_w%d_L%d_f%s.pkl"
                        % (prefix, int(RATIO), K, n_wires, int(round(side)), factors_code()))


def load_cache(path):
    # Returns (dictionary {seed: data of that network}, Vref saved in the
    # cache or None). Vref exists only in the "common" mode.
    if not os.path.exists(path):
        return {}, None
    with open(path, "rb") as fh:
        content = pickle.load(fh)
    if content["factors"] != FACTORS:
        raise ValueError("The cache " + path + " was made with other values of V*.")
    return content["nets"], content.get("vth_ref", None)


def save_cache(path, nets, vth_ref=None):
    # Writes to a temporary file first and then renames it, so that a crash
    # in the middle of the writing cannot destroy the old cache.
    content = {"factors": FACTORS, "K": K, "ratio": RATIO, "nets": nets,
               "vth_ref": vth_ref, "vth_mode": VTH_MODE}
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(content, fh)
    os.replace(tmp, path)


def old_thresholds(n_wires, side):
    # Reads {seed: Vth} from the biggest cache of run_tau_no_spanning.py with
    # the same size, if any (such a cache is a list of (seed, vth, out)).
    pattern = "cache_nospan_r%d_K%d_w%d_L%d_n*.pkl" % (int(RATIO), K, n_wires, int(round(side)))
    files = glob.glob(os.path.join(HERE, pattern))
    files = files + glob.glob(os.path.join(HERE, "..", pattern))
    best = {}
    for path in files:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if len(data) > len(best):
            best = {}
            for item in data:
                best[int(item[0])] = float(item[1])
    return best


def first_path_event(net, s, d, res):
    # Index of the first switch after which the junctions at level K (ON)
    # connect a source wire to a drain wire (-1 if it never happens).
    # The levels are replayed (no Kirchhoff). A path can only appear when a
    # junction reaches level K, so the check is done only then.
    m = net.junctions.shape[0]
    level = np.zeros(m, dtype=int)
    answer = -1
    k = 0
    n_ev = res["event_junction"].shape[0]
    while k < n_ev and answer < 0:
        j = res["event_junction"][k]
        level[j] = res["event_level"][k]
        if level[j] == K:
            on_list = np.nonzero(level == K)[0]
            if on_path_exists(net.n_wires(), net.junctions, on_list, s, d):
                answer = k
        k = k + 1
    return answer


def avalanche_jumps(res, k_path):
    # Increase of G over every avalanche (frame method, see the header).
    # Returns (jump of the avalanche that contains switch k_path (0.0 if
    # k_path < 0), largest jump of any avalanche).
    g = res["state_conductance"]
    times = res["event_time"]
    n_ev = times.shape[0]
    if n_ev == 0:
        return 0.0, 0.0
    ep_times = times[res["event_new_episode"]]
    width = 0.0
    if ep_times.shape[0] >= 2:
        width = (ep_times[-1] - ep_times[0]) / (ep_times.shape[0] - 1)
    # label[k] = number of the avalanche that contains switch k
    label = np.zeros(n_ev, dtype=int)
    if width > 0.0:
        frame = np.floor((times - times[0]) / width + 1.0e-9).astype(int)
        # a new avalanche starts when at least one EMPTY frame lies between
        # two consecutive switches, i.e. when the frame number jumps by 2 or more
        new_aval = np.diff(frame) >= 2
        label[1:] = np.cumsum(new_aval)
    n_aval = int(label[-1]) + 1
    best = 0.0
    path_jump = 0.0
    for a in range(n_aval):
        members = np.nonzero(label == a)[0]
        first = members[0]
        last = members[-1]
        # G before the first switch of the avalanche = state number first,
        # G after its last switch = state number last + 1
        jump = float(g[last + 1] - g[first])
        if jump > best:
            best = jump
        if k_path >= 0:
            if (k_path >= first) and (k_path <= last):
                path_jump = jump
    return path_jump, best


def summarise_run(net, s, d, res):
    # Builds the dictionary of one run (one network at one V*).
    g = res["state_conductance"]
    n_ev = res["event_junction"].shape[0]
    run = {}
    run["state_time"] = res["state_time"]
    run["state_conductance"] = g
    run["n_events"] = n_ev
    run["G_start"] = float(g[0])
    run["G_end"] = float(g[-1])
    if n_ev > 0:
        dg = np.diff(g)
        i_max = int(np.argmax(dg))
        run["dg_max"] = float(dg[i_max])
        run["dg_max_event"] = i_max
    else:
        run["dg_max"] = 0.0
        run["dg_max_event"] = -1
    k_path = first_path_event(net, s, d, res)
    run["path_event"] = k_path
    if k_path >= 0:
        run["jump_path"] = float(g[k_path + 1] - g[k_path])
        run["jump_path_ratio"] = float(g[k_path + 1] / g[k_path])
    else:
        run["jump_path"] = 0.0
        run["jump_path_ratio"] = float("nan")
    path_aval, best_aval = avalanche_jumps(res, k_path)
    run["jump_path_aval"] = path_aval
    run["dg_aval_max"] = best_aval
    run["event_new_episode"] = res["event_new_episode"]
    return run


def reference_threshold(n_wires, side, n_nets):
    # Mean of the Vth of the networks of this size, read from the caches of
    # the "own" mode (the one with the most networks). If there is none, the
    # old caches of run_tau_no_spanning.py are used. Returns (Vref, number of
    # networks used).
    pattern = os.path.join(HERE, "conductance_r%d_K%d_w%d_L%d_f*.pkl"
                           % (int(RATIO), K, n_wires, int(round(side))))
    best = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as fh:
            content = pickle.load(fh)
        values = []
        for seed in content["nets"]:
            v = content["nets"][seed]["vth"]
            if not np.isnan(v):
                values.append(float(v))
        if len(values) > len(best):
            best = values
    if len(best) == 0:
        old = old_thresholds(n_wires, side)
        for seed in old:
            if not np.isnan(old[seed]):
                best.append(float(old[seed]))
    if len(best) == 0:
        raise ValueError("No Vth found for %d wires, K = %d. Run the 'own' mode "
                         "first (without VTH_MODE)." % (n_wires, K))
    return float(np.mean(best)), len(best)


def one_network(task):
    # Worker: one network, all the V*. task = (n_wires, side, seed, vth or
    # None, vref or None). With vref (mode "common") the voltage is V* x vref
    # for every network; otherwise it is V* x the Vth of the network itself.
    n_wires, side, seed, vth, vref = task
    net = nn.generate_network(n_wires=n_wires, side=side, seed=seed)
    s, d = nn.point_electrodes(net)
    if vref is not None:
        data = {"vth": vref, "vth_own": vth, "runs": {}}
        for f in FACTORS:
            res = run_loop(net.n_wires(), net.junctions, s, d, f * vref,
                           V_SET, LAM_CRIT, G_OFF, G_ON, K)
            data["runs"][f] = summarise_run(net, s, d, res)
        return seed, data
    if vth is None:
        thr.run_loop_levels = run_loop
        vth = thr.find_threshold(net, s, d, K, V_SET, LAM_CRIT, G_OFF, G_ON)
    data = {"vth": vth, "runs": {}}
    if np.isnan(vth):
        return seed, data
    for f in FACTORS:
        res = run_loop(net.n_wires(), net.junctions, s, d, f * vth,
                       V_SET, LAM_CRIT, G_OFF, G_ON, K)
        data["runs"][f] = summarise_run(net, s, d, res)
    return seed, data


if __name__ == "__main__":
    for spec in SIZE_SPECS:
        parts = spec.split(":")
        n_wires = int(parts[0])
        side = float(parts[1])
        n_nets = int(parts[2])
        path = cache_name(n_wires, side)
        nets, vref = load_cache(path)
        known_vth = old_thresholds(n_wires, side)
        if VTH_MODE == "common":
            if vref is None:
                vref, n_used = reference_threshold(n_wires, side, n_nets)
                print("Vref = %.4f V = mean of the Vth of %d networks" % (vref, n_used),
                      flush=True)
            else:
                print("Vref = %.4f V (read from the cache)" % vref, flush=True)
        todo = []
        for seed in range(n_nets):
            if seed not in nets:
                own = None
                if seed in known_vth:
                    own = known_vth[seed]
                if VTH_MODE == "common":
                    todo.append((n_wires, side, seed, own, vref))
                else:
                    todo.append((n_wires, side, seed, own, None))
        print("=" * 70)
        print("%d wires, side %.0f, K = %d, %d V* values: %d networks done, %d to do"
              % (n_wires, side, K, len(FACTORS), len(nets), len(todo)), flush=True)
        print("cache file:", path, flush=True)
        t0 = time.perf_counter()
        n_done = 0
        if len(todo) > 0:
            with Pool(PROCS) as pool:
                for seed, data in pool.imap_unordered(one_network, todo):
                    nets[seed] = data
                    save_cache(path, nets, vref)
                    n_done = n_done + 1
                    print("  network %d done (%d of %d), %.0f s elapsed"
                          % (seed, n_done, len(todo), time.perf_counter() - t0),
                          flush=True)
        print("finished: %d networks in the cache" % len(nets), flush=True)
