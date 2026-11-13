"""Tests for the hierarchical (test-unit x seed) paired bootstrap."""

import numpy as np
from scipy import stats as st

from proxystudy.stats.bootstrap import paired_bootstrap
from proxystudy.stats.hierarchical import (
    cell_weights,
    cluster_weights,
    hierarchical_paired,
    weighted_bacc,
)
from proxystudy.stats.metrics import balanced_accuracy


def test_weighted_bacc_matches_index_resample():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    pred = rng.integers(0, 2, (3, 300))
    idx = rng.integers(0, 300, 300)
    w = np.bincount(idx, minlength=300).astype(float)
    got = weighted_bacc(pred.astype(float), y.astype(float), w)
    want = [balanced_accuracy(y[idx], p[idx]) for p in pred]
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_cluster_weights_keep_clusters_whole():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(50), 3)
    w = cluster_weights(clusters, rng)
    assert w.sum() == len(clusters)
    for c in range(50):
        assert len(set(w[clusters == c])) == 1


def test_cell_weights_preserve_cell_sizes():
    rng = np.random.default_rng(2)
    cells = np.array([0] * 40 + [1] * 25 + [2] * 10)
    w = cell_weights(cells, rng)
    for c, n in ((0, 40), (1, 25), (2, 10)):
        assert w[cells == c].sum() == n


def test_identical_procedures_give_zero_delta():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200).astype(float)
    pred = rng.integers(0, 2, (4, 200)).astype(float)
    res = hierarchical_paired(y, pred, pred, lambda r: cell_weights(y, r), n_boot=200, seed=0)
    assert res["overall"]["delta"] == 0.0
    assert res["overall"]["ci"] == [0.0, 0.0]
    assert res["overall"]["p_t"] == 1.0


def _null_world(rng, n=400, n_seeds=5, tau=0.6):
    """Two exchangeable training procedures scored on one shared test set.

    Seeds differ through a model-specific random slope on an image feature,
    so fitted models disagree on individual images even though the
    procedures are identical in distribution (the null is true).
    """
    y = (rng.random(n) < 0.3).astype(float)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)

    def fit():
        u = rng.normal(0, tau, size=(n_seeds, 1))
        e = rng.normal(0, 0.3, size=(n_seeds, n))
        return ((1.2 * y + 0.5 * z + u * x + e) > 0.6).astype(float)

    return y, fit(), fit()


def _fisher(p):
    p = np.clip(np.asarray(p), 1e-12, 1)
    return st.chi2.sf(-2 * np.log(p).sum(), 2 * len(p))


def test_null_calibration_beats_fisher_over_seeds():
    """Fisher over per-seed bootstrap p-values over-rejects a true null;
    the t-corrected hierarchical test stays at or below the nominal level."""
    rng = np.random.default_rng(20261113)
    reps, alpha = 60, 0.05
    rej_h = rej_f = 0
    for r in range(reps):
        y, a, b = _null_world(rng)
        h = hierarchical_paired(y, a, b, lambda g: cell_weights(y, g), n_boot=300, seed=r)
        rej_h += h["overall"]["p_t"] < alpha
        p_seed = [paired_bootstrap(y, a[s], b[s], groups=np.zeros(len(y), int),
                                   n_boot=300, seed=r * 10 + s).p_value for s in range(a.shape[0])]
        rej_f += _fisher(p_seed) < alpha
    assert rej_h / reps <= 0.10
    assert rej_f / reps >= 2 * max(rej_h / reps, alpha)


def test_deterministic_seeds_use_percentile():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300).astype(float)
    a1, b1 = rng.integers(0, 2, (2, 300)).astype(float)
    a, b = np.vstack([a1] * 5), np.vstack([b1] * 5)
    res = hierarchical_paired(y, a, b, lambda r: cell_weights(y, r), n_boot=200, seed=1)["overall"]
    assert res["primary"] == "percentile"
    assert res["p_primary"] == res["p"] and res["ci_primary"] == res["ci"]
