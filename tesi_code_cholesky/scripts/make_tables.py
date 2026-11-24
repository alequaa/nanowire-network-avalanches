# make_tables.py
# Writes, as tables, ALL the numbers that are plotted by
# plot_conductance_curves.py and plot_conductance_curves_common.py, for the
# two ways of defining V*:
#   mode "own"    : V = V* x Vth of each network      (files conductance_r*.pkl)
#   mode "common" : V = V* x Vref(L), same Vref for all the networks of one size
#                   (files conductance_common_r*.pkl)
# No simulation is run here: the caches made by run_conductance_curves.py and
# run_conductance_curves_common.py are only read.
#
# For every mode, every K and every size (number of wires) there is one table
# with one row per V*, and these columns:
#   n_networks      number of networks used
#   frac_path       fraction of networks where a path of ON junctions forms
#   jump_single     mean jump of G in the single switch that closes the path
#   jump_single_gp  same, but only over the networks that have a path (gp)
#   jump_aval       mean jump of G over the avalanche that closes the path
#   jump_aval_gp    same, only over the networks that have a path
#   jump_maxaval    mean of the largest jump of G over one avalanche
#   G_end           mean final conductance
#   every quantity has a column with the same name + "_sem" (standard error
#   of the mean = standard deviation with n - 1 / sqrt(n)).
# Networks without a path count as 0 in jump_single and jump_aval (as in the
# pictures). G is in units of Gon.
#
# Numerical example of the standard error: 4 networks with jumps 0.1, 0.2,
# 0.3, 0.4: mean 0.25, standard deviation 0.129, error 0.129 / 2 = 0.065.
#
# Output: folder scripts/tables
#   tables_<mode>_K<K>.csv    (one file per mode and K, all the sizes inside)
#   vref_table.csv            the reference threshold used by the mode "common"
#   tables_all.xlsx           the same tables in one Excel file, one sheet for
#                             each mode and K (only if the package openpyxl
#                             is installed; the csv files are always written)
#
# How to run (from the main folder):  python scripts/make_tables.py
# Environment variables: KS (list of K, default "1,10"), RATIO (default 1000).

import os
import glob
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RATIO = int(os.environ.get("RATIO", "1000"))
KS = []
for piece in os.environ.get("KS", "1,10").split(","):
    KS.append(int(piece))
OUT = os.path.join(HERE, "tables")

# (column name, key in the cache or None for "has a path")
COLUMNS = ["frac_path", "jump_single", "jump_single_gp", "jump_aval",
           "jump_aval_gp", "jump_maxaval", "G_end"]


def find_caches(prefix, k_value):
    # {n_wires: content of the cache with the most networks}.
    pattern = os.path.join(HERE, "%s_r%d_K%d_w*_L*_f*.pkl" % (prefix, RATIO, k_value))
    best = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        n_wires = int(name.split("_w")[1].split("_")[0])
        with open(path, "rb") as fh:
            content = pickle.load(fh)
        keep = False
        if n_wires not in best:
            keep = True
        else:
            if len(content["nets"]) > len(best[n_wires]["nets"]):
                keep = True
        if keep:
            best[n_wires] = content
    return best


def mean_sem(values):
    # Mean and standard error of a list; nan if the list is empty (mean) or
    # has fewer than 2 numbers (error).
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    sem = float("nan")
    if n >= 2:
        sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mean, sem


def table_for_size(content):
    # Returns a list of rows (dictionaries), one per V*.
    rows = []
    good = []
    for seed in sorted(content["nets"].keys()):
        if not np.isnan(content["nets"][seed]["vth"]):
            good.append(seed)
    for f in content["factors"]:
        has_path = []
        single = []
        aval = []
        maxaval = []
        gend = []
        for seed in good:
            run = content["nets"][seed]["runs"][f]
            if run["path_event"] >= 0:
                has_path.append(1.0)
            else:
                has_path.append(0.0)
            single.append(run["jump_path"])
            aval.append(run["jump_path_aval"])
            maxaval.append(run["dg_aval_max"])
            gend.append(run["G_end"])
        has_arr = np.asarray(has_path, dtype=float)
        single_arr = np.asarray(single, dtype=float)
        aval_arr = np.asarray(aval, dtype=float)
        mask = has_arr > 0.5                     # networks that have a path
        row = {"Vstar": f, "n_networks": len(good)}
        row["frac_path"], row["frac_path_sem"] = mean_sem(has_arr)
        row["jump_single"], row["jump_single_sem"] = mean_sem(single_arr)
        row["jump_single_gp"], row["jump_single_gp_sem"] = mean_sem(single_arr[mask])
        row["jump_aval"], row["jump_aval_sem"] = mean_sem(aval_arr)
        row["jump_aval_gp"], row["jump_aval_gp_sem"] = mean_sem(aval_arr[mask])
        row["jump_maxaval"], row["jump_maxaval_sem"] = mean_sem(maxaval)
        row["G_end"], row["G_end_sem"] = mean_sem(gend)
        rows.append(row)
    return rows


def header_names():
    names = ["wires", "Vstar", "n_networks"]
    for c in COLUMNS:
        names.append(c)
        names.append(c + "_sem")
    return names


def row_values(n_wires, row):
    vals = [n_wires, row["Vstar"], row["n_networks"]]
    for c in COLUMNS:
        vals.append(row[c])
        vals.append(row[c + "_sem"])
    return vals


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    all_tables = {}                      # sheet name -> list of value rows
    vref_rows = []
    for mode in ["own", "common"]:
        prefix = "conductance"
        if mode == "common":
            prefix = "conductance_common"
        for k_value in KS:
            caches = find_caches(prefix, k_value)
            if len(caches) == 0:
                print("no cache for mode %s, K = %d: skipped" % (mode, k_value))
                continue
            names = header_names()
            lines = []
            for n_wires in sorted(caches.keys()):
                content = caches[n_wires]
                for row in table_for_size(content):
                    lines.append(row_values(n_wires, row))
                if mode == "common":
                    vref_rows.append([k_value, n_wires, content.get("vth_ref", float("nan"))])
            path = os.path.join(OUT, "tables_%s_K%d.csv" % (mode, k_value))
            with open(path, "w") as fh:
                fh.write(",".join(names) + "\n")
                for vals in lines:
                    fh.write(",".join([str(x) for x in vals]) + "\n")
            all_tables["%s_K%d" % (mode, k_value)] = (names, lines)
            print("saved", path, "(sizes: %s)" % ", ".join([str(w) for w in sorted(caches.keys())]))
    if len(vref_rows) > 0:
        path = os.path.join(OUT, "vref_table.csv")
        with open(path, "w") as fh:
            fh.write("K,wires,Vref_V\n")
            for r in vref_rows:
                fh.write("%d,%d,%s\n" % (r[0], r[1], str(r[2])))
        print("saved", path)
    # Excel file, if possible
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        first = True
        for sheet_name in all_tables:
            names, lines = all_tables[sheet_name]
            if first:
                ws = wb.active
                ws.title = sheet_name
                first = False
            else:
                ws = wb.create_sheet(sheet_name)
            ws.append(names)
            for vals in lines:
                ws.append(vals)
        if len(vref_rows) > 0:
            ws = wb.create_sheet("vref")
            ws.append(["K", "wires", "Vref_V"])
            for r in vref_rows:
                ws.append(r)
        xlsx_path = os.path.join(OUT, "tables_all.xlsx")
        wb.save(xlsx_path)
        print("saved", xlsx_path)
    except ImportError:
        print("openpyxl is not installed: only the csv files were written")
