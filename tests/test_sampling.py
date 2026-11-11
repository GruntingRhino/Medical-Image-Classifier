"""Tests for sampling methods: correctness of weighting logic and controls."""

import numpy as np
import pytest

from proxystudy.proxies.quality import (
    bin_proxy_quantiles,
    cohens_kappa,
    normalized_mutual_info,
    proxy_quality_summary,
)
from proxystudy.sampling.methods import (
    MethodConfig,
    dro_update_weights,
    fit_method,
    resample_from_weights,
)


def _toy_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    G = (rng.random(n) < 0.25).astype(int)
    y = (rng.random(n) < np.where(G == 1, 0.4, 0.2)).astype(int)
    P = 0.7 * G + 0.3 * rng.random(n) + 0.1 * rng.standard_normal(n)
    return y, P, G


def test_erm_uniform():
    y, P, _ = _toy_data()
    out = fit_method(MethodConfig("erm"), y, P)
    assert np.allclose(out.weights, out.weights[0])


def test_class_strat_equalizes_class_mass():
    y, P, _ = _toy_data()
    out = fit_method(MethodConfig("class_strat"), y, P)
    m0 = out.weights[y == 0].sum()
    m1 = out.weights[y == 1].sum()
    assert abs(m0 - 0.5) < 1e-10 and abs(m1 - 0.5) < 1e-10


def test_proxy_strat_equalizes_bin_mass():
    y, P, _ = _toy_data()
    cfg = MethodConfig("proxy_strat", n_bins=5)
    out = fit_method(cfg, y, P)
    assert out.bins is not None
    for b in np.unique(out.bins):
        assert abs(out.weights[out.bins == b].sum() - 1 / len(np.unique(out.bins))) < 1e-9


def test_proxy_strat_joint_cells():
    y, P, _ = _toy_data()
    cfg = MethodConfig("proxy_strat", n_bins=3, joint_with_label=True)
    out = fit_method(cfg, y, P)
    cells = set(zip(out.bins.tolist(), y.tolist()))
    for (b, c) in cells:
        m = (out.bins == b) & (y == c)
        assert abs(out.weights[m].sum() - 1 / len(cells)) < 1e-9


def test_proxy_iw_inverse_frequency():
    y, P, _ = _toy_data()
    out = fit_method(MethodConfig("proxy_iw", n_bins=4, cap_factor=1e9), y, P)
    bins, _ = bin_proxy_quantiles(P, 4)
    counts = {b: (bins == b).sum() for b in np.unique(bins)}
    raw = np.array([1.0 / counts[bins[i]] for i in range(len(y))])
    np.testing.assert_allclose(out.weights, raw / raw.sum())


def test_proxy_iw_capping_bounds_weight_ratio():
    """Stabilized IW: max weight ratio must not exceed cap_factor."""
    y, P, _ = _toy_data(6000, seed=3)
    # make one bin extremely sparse: extreme proxy value
    P = np.concatenate([P[:-20], P[-20:] + 100.0])
    out = fit_method(MethodConfig("proxy_iw", n_bins=5, cap_factor=5.0), y, P)
    w = out.weights / out.weights.min()
    assert w.max() <= 5.0 + 1e-9


def test_random_proxy_destroys_association():
    y, P, G = _toy_data(4000, seed=1)
    real = fit_method(MethodConfig("proxy_strat", n_bins=5, seed=99), y, P)
    shuf = fit_method(MethodConfig("random_proxy_strat", n_bins=5, seed=99), y, P)
    # compare dichotomized kappa against true G: real proxy informative,
    # shuffled proxy uninformative
    from proxystudy.proxies.quality import dichotomize_proxy

    P_arr = np.asarray(P, dtype=float)
    k_real = cohens_kappa(dichotomize_proxy(P_arr), G)
    P_shuffled = np.array(shuf.meta.get("shuffled_proxy", P_arr))
    k_shuf = cohens_kappa(dichotomize_proxy(P_shuffled), G)
    assert k_real > 0.25
    assert abs(k_shuf) < 0.05


def test_oracle_requires_G():
    y, P, G = _toy_data()
    with pytest.raises(ValueError):
        fit_method(MethodConfig("oracle_strat"), y, P)  # no G_train
    out = fit_method(MethodConfig("oracle_strat"), y, P, G_train=G)
    g1 = out.weights[G == 1].sum()
    assert abs(g1 - 0.5) < 1e-9


def test_proxy_methods_require_P():
    y, P, G = _toy_data()
    for name in ["proxy_strat", "proxy_iw", "proxy_dro"]:
        with pytest.raises(ValueError):
            fit_method(MethodConfig(name), y, None)


def test_resample_deterministic_and_matches_weights():
    y, P, _ = _toy_data()
    out = fit_method(MethodConfig("class_strat"), y, P)
    idx1 = resample_from_weights(out.weights, 200_000, seed=123)
    idx2 = resample_from_weights(out.weights, 200_000, seed=123)
    np.testing.assert_array_equal(idx1, idx2)
    # empirical frequencies track weights
    freq = np.bincount(idx1, minlength=len(y)) / len(idx1)
    corr = np.corrcoef(freq, out.weights)[0, 1]
    assert corr > 0.98


def test_dro_update_shifts_mass_to_high_loss_bin():
    y, P, _ = _toy_data()
    out = fit_method(MethodConfig("proxy_dro", n_bins=2), y, P)
    bins = out.bins
    w = out.weights.copy()
    losses = np.where(bins == np.max(bins), 10.0, 0.0)
    w2 = dro_update_weights(losses, bins, w, eta=0.5)
    mass_before = w[bins == np.max(bins)].sum()
    mass_after = w2[bins == np.max(bins)].sum()
    assert mass_after > mass_before
    assert abs(w2.sum() - 1.0) < 1e-9


def test_quantile_binning_edges():
    P = np.arange(1000, dtype=float)
    bins, edges = bin_proxy_quantiles(P, 5)
    assert len(np.unique(bins)) == 5
    counts = [np.sum(bins == b) for b in range(5)]
    assert all(abs(c - 200) <= 2 for c in counts)


def test_proxy_quality_summary():
    rng = np.random.default_rng(2)
    n = 3000
    G = (rng.random(n) < 0.3).astype(int)
    P = 0.8 * G + 0.2 * rng.random(n)
    q = proxy_quality_summary(P, G, n_bins=5)
    assert q["auc_for_group"] > 0.85
    assert q["kappa"] > 0.5
    assert 0 <= q["nmi"] <= 1
    # independent proxy
    P_ind = rng.random(n)
    q_ind = proxy_quality_summary(P_ind, G, n_bins=5)
    assert abs(q_ind["auc_for_group"] - 0.5) < 0.05
    assert abs(q_ind["kappa"]) < 0.05
