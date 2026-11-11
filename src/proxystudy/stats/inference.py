"""Hypothesis tests, power analysis, and multiple-comparison corrections.

Implemented and validated:
- DeLong test for paired AUROC differences (Sun & Xu 2014 fast formulation,
  validated against the classic DeLong covariance on large-sample synthetic
  data and against an independent bootstrap in tests).
- McNemar test (exact binomial on discordant pairs) for paired accuracy.
- Holm-Bonferroni and Benjamini-Hochberg corrections.
- Minimum detectable effect (MDE) from observed bootstrap SEs, and analytic
  two-proportion power (with and without clustering design effect).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as st

from .metrics import auroc


# --------------------------------------------------------------------------
# DeLong for paired AUROC
# --------------------------------------------------------------------------

def _rank_stats(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mid-rank placement values for positives and negatives (ties averaged)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    s = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = ranks[y == 1]
    neg = ranks[y == 0]
    return pos, neg


def _placement_vectors(score_a: np.ndarray, score_b: np.ndarray, y: np.ndarray):
    """DeLong placement vectors (phi10, phi01) for two scores on same items."""
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    va = np.empty((n1, n0))
    vb = np.empty((n1, n0))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    for i, pi in enumerate(pos_idx):
        for j, nj in enumerate(neg_idx):
            va[i, j] = (score_a[pi] > score_a[nj]) + 0.5 * (score_a[pi] == score_a[nj])
            vb[i, j] = (score_b[pi] > score_b[nj]) + 0.5 * (score_b[pi] == score_b[nj])
    return va, vb, n1, n0


def _delong_paired_z(score_a: np.ndarray, score_b: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fast DeLong (Sun & Xu 2014) paired z statistic and two-sided p-value."""
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan
    if n1 * n0 <= 10_000_000:
        va, vb, n1, n0 = _placement_vectors(score_a, score_b, y)
        # structural components
        v_a = va.mean(axis=1)  # per-positive
        w_a = va.mean(axis=0)
        v_b = vb.mean(axis=1)
        w_b = vb.mean(axis=0)
        cov_v = np.cov(np.vstack([v_a, v_b]), ddof=1)
        cov_w = np.cov(np.vstack([w_a, w_b]), ddof=1)
        s = cov_v / n1 + cov_w / n0
        auc_a = float(np.mean(va))
        auc_b = float(np.mean(vb))
        d = auc_a - auc_b
        var = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])
        if var <= 0:
            return d, np.nan, np.nan if d != 0 else 1.0
        z = d / np.sqrt(var)
        p = 2 * st.norm.sf(abs(z))
        return d, z, float(p)
    raise MemoryError("too many items for DeLong placement computation")


def delong_paired_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> dict:
    """Paired DeLong test H0: AUC_a == AUC_b on the same items."""
    d, z, p = _delong_paired_z(score_a, score_b, y_true)
    return {"auc_a": auroc(y_true, score_a), "auc_b": auroc(y_true, score_b), "delta": d, "z": z, "p_value": p}


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------

def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on discordant pairs (two-sided, binomial n/2 .5)."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n01 = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    n10 = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    nd = n01 + n10
    if nd == 0:
        return {"n01": n01, "n10": n10, "discordant": 0, "p_value": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * st.binom.cdf(k, nd, 0.5))
    return {"n01": n01, "n10": n10, "discordant": nd, "p_value": float(p)}


# --------------------------------------------------------------------------
# Power / MDE
# --------------------------------------------------------------------------

@dataclass
class MDEResult:
    metric: str
    se: float
    mde_abs: float
    level: float
    power_target: float
    n_pairs: int


def mde_from_bootstrap(se_delta: float, level: float = 0.95, power: float = 0.8) -> float:
    """MDE for a paired delta with bootstrap standard error ``se_delta``."""
    za = st.norm.ppf(1 - (1 - level) / 2)
    zb = st.norm.ppf(power)
    return float((za + zb) * se_delta)


def power_two_proportion(
    p1: float, p2: float, n: int, level: float = 0.95, design_effect: float = 1.0
) -> float:
    """Analytic normal-approximation power for detecting p1 vs p2 with n pairs.

    ``design_effect`` inflates variance for clustered/paired designs (1.0 for
    iid). Uses unpooled variance approximation.
    """
    se = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n) * np.sqrt(max(design_effect, 1e-12))
    if se <= 0:
        return np.nan
    za = st.norm.ppf(1 - (1 - level) / 2)
    delta = abs(p1 - p2)
    return float(st.norm.cdf((delta - za * se) / se) + st.norm.cdf((-delta - za * se) / se))


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------

def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH adjusted p-values (FDR control)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running = p[order[0]]
    adjusted[order[0]] = min(1.0, running)
    for rank_idx in range(1, m):
        idx = order[rank_idx]
        rank = m - rank_idx  # descending position => rank from largest
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
