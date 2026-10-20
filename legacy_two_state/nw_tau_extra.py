# nw_tau_extra.py
# Extra tools for the exponent tau, used only by run_tau_D.py in this folder:
# range rule of Hochstetter (DOI 10.1038/s41467-021-24260-z) with its KS
# p-value (select_range, ks_pvalue). Also kept: choice of smin by the minimum
# KS distance (best_smin, bootstrap_best) and power law with exponential
# cutoff (fit_cutoff), not used by any script left in this folder.
# The basic fit (fit_tau, bootstrap_tau) is in nw_tau.py in the main folder.

import numpy as np
from scipy.optimize import minimize

from nw_tau import fit_tau


def ks_distance(sizes, smin, smax, tau):
    # Kolmogorov-Smirnov distance D between the data CDF and the model CDF
    # (truncated discrete power law on smin..smax), using sizes >= smin.
    # Small D = the power law describes the data well from smin upward.
    s = np.asarray(sizes, dtype=int)
    s = s[(s >= smin) & (s <= smax)]
    n = s.shape[0]
    k = np.arange(smin, smax + 1)
    # data CDF at each k
    counts = np.bincount(s - smin, minlength=k.shape[0])
    cdf_data = np.cumsum(counts) / n
    # model CDF at each k
    weights = k.astype(float) ** (-tau)
    cdf_model = np.cumsum(weights) / np.sum(weights)
    return float(np.max(np.abs(cdf_data - cdf_model)))


def best_smin(sizes, smax, smin_list, min_count=100):
    # For each candidate smin: fit tau, compute D. Pick the smin with the
    # smallest D (minimum KS distance). Candidates with fewer than min_count
    # sizes are skipped. Returns (smin, tau, D), or (None, nan, nan) if none left.
    s = np.asarray(sizes, dtype=int)
    best = None
    for smin in smin_list:
        n = int(np.sum((s >= smin) & (s <= smax)))
        if n < min_count:
            continue
        tau = fit_tau(s, smin, smax)
        if np.isnan(tau):
            continue
        d = ks_distance(s, smin, smax, tau)
        if best is None:
            best = (smin, tau, d)
        else:
            if d < best[2]:
                best = (smin, tau, d)
    if best is None:
        return None, float("nan"), float("nan")
    return best


def bootstrap_best(sizes_per_network, smax, smin_list, n_boot=200, seed=0,
                   min_count=100):
    # Bootstrap over networks, choosing smin again in every resample
    # (so the error includes the uncertainty on smin). Returns
    # (mean tau, std tau, array of chosen smin).
    rng = np.random.default_rng(seed)
    n_net = len(sizes_per_network)
    taus = []
    smins = []
    for b in range(n_boot):
        pick = rng.integers(0, n_net, size=n_net)
        pooled = np.concatenate([sizes_per_network[i] for i in pick])
        sm, tau, d = best_smin(pooled, smax, smin_list, min_count)
        if sm is not None:
            taus.append(tau)
            smins.append(sm)
    return float(np.mean(taus)), float(np.std(taus)), np.array(smins)


def fit_cutoff(sizes, smin, smax):
    # Fit P(S) ~ S^(-tau) * exp(-lam * S) on smin..smax (lam = 1/S0), by
    # maximum likelihood. lam >= 0, so lam = 0 is the pure power law.
    # Returns (tau, lam, loglik), or (nan, nan, nan) if fewer than 5 sizes.
    s = np.asarray(sizes, dtype=float)
    s = s[(s >= smin) & (s <= smax)]
    n = s.shape[0]
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    sum_log = np.sum(np.log(s))
    sum_s = np.sum(s)
    k = np.arange(smin, smax + 1, dtype=float)
    log_k = np.log(k)

    def minus_loglik(p):
        tau = p[0]
        lam = p[1]
        # log of the normalisation, computed in a stable way
        expo = -tau * log_k - lam * k
        top = np.max(expo)
        log_z = top + np.log(np.sum(np.exp(expo - top)))
        return tau * sum_log + lam * sum_s + n * log_z

    best = None
    # several starting points, keep the best (the surface can be flat)
    for start in [(1.5, 0.01), (1.0, 0.05), (2.0, 0.001), (0.5, 0.1)]:
        out = minimize(minus_loglik, start, method="L-BFGS-B",
                       bounds=[(0.0, 6.0), (0.0, 2.0)])
        if best is None:
            best = out
        else:
            if out.fun < best.fun:
                best = out
    return float(best.x[0]), float(best.x[1]), float(-best.fun)


def ks_pvalue(sizes, xmin, xmax, tau, n_synth=500, seed=0, refit=False):
    # p-value of the power-law fit on xmin..xmax (Hochstetter's Methods):
    # draw n_synth synthetic data sets from the fitted truncated power law,
    # p = fraction of them whose KS distance is at least the empirical one.
    # refit=False: distance to the same fitted law (as written in the paper).
    # refit=True : tau refitted on every synthetic set (stricter).
    s = np.asarray(sizes, dtype=int)
    s = s[(s >= xmin) & (s <= xmax)]
    n = s.shape[0]
    d_emp = ks_distance(s, xmin, xmax, tau)
    k = np.arange(xmin, xmax + 1)
    w = k.astype(float) ** (-tau)
    cdf = np.cumsum(w) / np.sum(w)
    rng = np.random.default_rng(seed)
    count_bigger = 0
    for b in range(n_synth):
        u = rng.random(n)
        idx = np.searchsorted(cdf, u)
        idx = np.minimum(idx, k.shape[0] - 1)
        synth = idx + xmin
        if refit:
            t2 = fit_tau(synth, xmin, xmax)
            d_syn = ks_distance(synth, xmin, xmax, t2)
        else:
            d_syn = ks_distance(synth, xmin, xmax, tau)
        if d_syn >= d_emp:
            count_bigger = count_bigger + 1
    return count_bigger / n_synth


def select_range(sizes, xmin_list, p_min=0.5, min_count=30, n_synth=500):
    # Hochstetter's rule: among the ranges [xmin, xmax] with p > p_min, take
    # the one with the largest log(xmax / xmin).
    # Returns a dict with xmin, xmax, tau, p, n, passed (True if p > p_min).
    # If no range passes, returns the range with the highest p (passed False).
    s = np.asarray(sizes, dtype=int)
    top = int(s.max())
    xmax_base = [4, 6, 8, 10, 15, 20, 30, 50, 80, 120, 200, 300, 500]
    xmax_list = [x for x in xmax_base if x < top]
    xmax_list.append(top)
    best_pass = None
    best_any = None
    for xmin in xmin_list:
        for xmax in xmax_list:
            if xmax < 2 * xmin:
                continue
            n = int(np.sum((s >= xmin) & (s <= xmax)))
            if n < min_count:
                continue
            tau = fit_tau(s, xmin, xmax)
            if np.isnan(tau):
                continue
            p = ks_pvalue(s, xmin, xmax, tau, n_synth=n_synth)
            cand = {"xmin": xmin, "xmax": xmax, "tau": tau, "p": p, "n": n,
                    "passed": p > p_min}
            if best_any is None:
                best_any = cand
            else:
                if p > best_any["p"]:
                    best_any = cand
            if p > p_min:
                if best_pass is None:
                    best_pass = cand
                else:
                    if np.log(xmax / xmin) > np.log(best_pass["xmax"] / best_pass["xmin"]):
                        best_pass = cand
    if best_pass is not None:
        return best_pass
    return best_any
