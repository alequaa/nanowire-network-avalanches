# nw_tau.py
# Estimate the exponent tau of a discrete power law P(S) ~ S^(-tau) from a
# list of avalanche sizes, by maximum likelihood, with an error bar obtained
# by resampling whole networks (bootstrap).
#
# Example (by hand): sizes [1, 1, 2, 4], smin = 1.
#   log-likelihood(tau) = -tau * (ln1 + ln1 + ln2 + ln4) - 4 * ln(Z(tau)),
#   where Z(tau) = sum over S >= smin of S^(-tau) is the normalisation.
#   The best tau is the one that makes this number as large as possible.
#
# smax: if given, sizes are assumed to live in [smin, smax] (finite network:
# an avalanche cannot be larger than the number of junctions), and Z is the
# sum from smin to smax. If None, Z is the infinite sum (Hurwitz zeta).

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta


def normalisation(tau, smin, smax):
    if smax is None:
        return zeta(tau, smin)
    k = np.arange(smin, smax + 1, dtype=float)
    return np.sum(k ** (-tau))


def fit_tau(sizes, smin=1, smax=None):
    # Returns tau (float), or nan if there are fewer than 3 sizes >= smin.
    s = np.asarray(sizes, dtype=float)
    s = s[s >= smin]
    if smax is not None:
        s = s[s <= smax]
    n = s.shape[0]
    if n < 3:
        return float("nan")
    sum_log = np.sum(np.log(s))

    def minus_loglik(tau):
        return tau * sum_log + n * np.log(normalisation(tau, smin, smax))

    # tau in (1.01, 8): the infinite sum needs tau > 1.
    out = minimize_scalar(minus_loglik, bounds=(1.01, 8.0), method="bounded")
    return float(out.x)


def bootstrap_tau(sizes_per_network, smin=1, smax=None, n_boot=200, seed=0):
    # sizes_per_network: list of arrays, one per network.
    # Resample NETWORKS with replacement (avalanches inside one network are
    # not independent), pool their sizes, refit. Returns (mean, std).
    rng = np.random.default_rng(seed)
    n_net = len(sizes_per_network)
    taus = []
    for b in range(n_boot):
        pick = rng.integers(0, n_net, size=n_net)
        pooled = np.concatenate([sizes_per_network[i] for i in pick])
        t = fit_tau(pooled, smin, smax)
        if not np.isnan(t):
            taus.append(t)
    taus = np.array(taus)
    return float(np.mean(taus)), float(np.std(taus))
