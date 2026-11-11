"""Validation of bootstrap CIs, paired comparisons, and inference procedures."""

import numpy as np
import pytest
from scipy import stats as st

from proxystudy.stats.bootstrap import bootstrap_ci, paired_bootstrap, seed_summary
from proxystudy.stats.inference import (
    benjamini_hochberg,
    delong_paired_test,
    holm_bonferroni,
    mcnemar_exact,
    mde_from_bootstrap,
    power_two_proportion,
)
from proxystudy.stats.metrics import auroc


def test_bootstrap_ci_coverage_nominal():
    """Coverage of cell-stratified bootstrap CI on a known-probability metric."""
    rng = np.random.default_rng(0)
    covered = 0
    trials = 300
    for t in range(trials):
        n = 400
        g = rng.integers(0, 2, size=n)
        y = (rng.random(n) < np.where(g == 1, 0.6, 0.4)).astype(int)
        pred = (rng.random(n) < np.where(y == 1, 0.8, 0.25)).astype(int)
        ci = bootstrap_ci(y, pred, "balanced_accuracy", groups=g, n_boot=400, seed=t)
        # true bacc: sensitivity 0.8, specificity 0.75
        if ci.lower <= 0.775 <= ci.upper:
            covered += 1
    rate = covered / trials
    assert 0.90 <= rate <= 0.995, f"coverage {rate:.3f} outside nominal band"


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(1)
    n = 3000
    g = rng.integers(0, 2, size=n)
    y = (rng.random(n) < 0.3).astype(int)
    # method A better on group 1
    err = rng.random(n)
    pred_a = np.where(g == 1, (err < 0.2).astype(int) & 0, y)  # perfect on g1
    pred_a = np.where(g == 0, (rng.random(n) < 0.75).astype(int) * y, pred_a)
    # crude construction: A: sens .9 spec .9 overall; B: worse on g1
    p = rng.random(n)
    pred_a = ((y == 1) & (p < 0.9)) | ((y == 0) & (p >= 0.9))
    pred_b = ((y == 1) & (p < 0.8)) | ((y == 0) & (p >= 0.8))
    comp = paired_bootstrap(
        y, pred_a.astype(int), pred_b.astype(int), groups=g, n_boot=500, seed=0
    )
    assert comp.delta > 0.02
    assert comp.ci_lower > 0


def test_paired_bootstrap_null_coverage():
    """Two methods with identical error rates -> CI covers 0 ~95% of the time."""
    rng = np.random.default_rng(2)
    rejects = 0
    trials = 60
    for t in range(trials):
        n = 800
        g = rng.integers(0, 2, size=n)
        y = (rng.random(n) < 0.35).astype(int)
        p = rng.random(n)
        same = ((y == 1) & (p < 0.85)) | ((y == 0) & (p >= 0.85))
        comp = paired_bootstrap(y, same.astype(int), same.astype(int), groups=g, n_boot=300, seed=t)
        # delta should be exactly 0
        if not (comp.ci_lower <= 0 <= comp.ci_upper):
            rejects += 1
    assert rejects == 0, "identical predictions must always produce a CI covering 0"


def test_delong_matches_known_and_bootstrap():
    rng = np.random.default_rng(3)
    n = 4000
    y = (rng.random(n) < 0.4).astype(int)
    s_a = rng.normal(size=n) + 0.4 * y
    s_b = rng.normal(size=n) + 0.2 * y
    res = delong_paired_test(y, s_a, s_b)
    assert res["p_value"] < 0.01
    assert res["auc_a"] > res["auc_b"]
    # independent method: same scores -> p ~ 1
    res_eq = delong_paired_test(y, s_a.copy(), s_a.copy())
    assert res_eq["delta"] == 0 and res_eq["p_value"] == 1.0


def test_delong_vs_bootstrap_agreement():
    """DeLong and paired bootstrap should agree on delta and significance."""
    rng = np.random.default_rng(4)
    n = 2000
    y = (rng.random(n) < 0.5).astype(int)
    s_a = rng.normal(size=n) + 0.35 * y
    s_b = rng.normal(size=n) + 0.25 * y
    dres = delong_paired_test(y, s_a, s_b)
    bres = paired_bootstrap(y, s_a, s_b, "auroc", n_boot=500, seed=0)
    # agreement on point estimate (same estimand)
    assert abs(dres["delta"] - bres.delta) < 1e-12
    # agreement on the inference: same direction of decision
    assert (dres["p_value"] < 0.05) == (bres.p_value < 0.05)
    # strong separation: both must reject
    s_c = rng.normal(size=n) + 1.2 * y
    d2 = delong_paired_test(y, s_c, s_b)
    b2 = paired_bootstrap(y, s_c, s_b, "auroc", n_boot=500, seed=0)
    assert d2["p_value"] < 0.001 and b2.p_value <= 0.01  # bootstrap p floors at 1/(n_boot+1)
    assert d2["auc_a"] > d2["auc_b"] and b2.delta > 0.05


def test_mcnemar_exact():
    # A wrong where B right on 8 pairs, reverse 1 -> significant
    y = np.ones(20, dtype=int)
    pred_a = np.array([1] * 11 + [0] * 9)
    pred_b = np.array([1] * 19 + [0] * 1)
    res = mcnemar_exact(y, pred_a, pred_b)
    # discordant: a wrong 9 (11..19 wrong? compute properly below)
    c_a = pred_a == y
    c_b = pred_b == y
    n01 = int(np.sum(~c_a & c_b))
    n10 = int(np.sum(c_a & ~c_b))
    assert res["discordant"] == n01 + n10
    from scipy import stats as st

    expected = min(1.0, 2 * st.binom.cdf(min(n01, n10), n01 + n10, 0.5))
    assert abs(res["p_value"] - expected) < 1e-12


def test_multiplicity_corrections():
    ps = [0.001, 0.008, 0.039, 0.041, 0.2, 0.8]
    holm = holm_bonferroni(ps)
    bh = benjamini_hochberg(ps)
    # Holm: sorted 0.001*6=0.006, 0.008*5=0.04, 0.039*4=0.156, 0.041*3=0.123
    # (running max keeps 0.156), 0.2*2=0.4, 0.8*1=0.8
    assert holm == [0.006, 0.04, 0.156, 0.156, 0.4, 0.8]
    # BH: monotone nondecreasing in p order
    order = np.argsort(ps)
    bh_sorted = [bh[i] for i in order]
    assert all(bh_sorted[i] <= bh_sorted[i + 1] + 1e-12 for i in range(len(ps) - 1))
    assert all(0 <= x <= 1 for x in bh)


def test_mde_and_power():
    mde = mde_from_bootstrap(se_delta=0.02, level=0.95, power=0.8)
    assert abs(mde - (1.96 + 0.8416) * 0.02) < 1e-3
    p80 = power_two_proportion(0.60, 0.55, n=1000)
    p00 = power_two_proportion(0.60, 0.60, n=1000)
    assert p80 > 0.5 and abs(p00 - 0.05) < 0.03


def test_seed_summary():
    s = seed_summary([0.80, 0.82, 0.84, 0.86], metric="bacc")
    assert abs(s.mean - 0.83) < 1e-12
    assert abs(s.sd - np.std([0.80, 0.82, 0.84, 0.86], ddof=1)) < 1e-12
    assert s.t_ci_lower < s.mean < s.t_ci_upper
    one = seed_summary([0.5], "m")
    assert np.isnan(one.sd) and np.isnan(one.t_ci_lower)
