"""Hierarchical paired bootstrap over test units and training seeds.

Why this exists
---------------
Every training seed scores the SAME test images. Per-seed p-values are
therefore dependent, and combining them with Fisher's method (which assumes
independent experiments) overstates the evidence. A per-seed image bootstrap
also ignores training randomness: it tests whether two *fixed* fitted models
differ, not whether two *training procedures* differ.

This module resamples test units (lesion clusters, or (group, label) cells
for single-image data) and training seeds jointly within each bootstrap
draw. The estimand is the expected metric difference between two training
procedures, averaged over training randomness, on the test population.
Seeds are paired: seed s of method A is compared with seed s of the
reference, and the seed index is resampled as a unit.

Balanced accuracy is computed from resampling multiplicities (weights), so
one draw is a pair of matrix-vector products over all seeds at once.

Primary inference (``ci_t``, ``p_t``) is a t reference distribution with
n_seeds - 1 degrees of freedom on the bootstrap standard error, inflated by
sqrt(S / (S - 1)). With few seeds the plain percentile bootstrap
underestimates between-seed variance and over-rejects a true null (about 8-10%
at alpha = 0.05 with 5-10 seeds when seed noise dominates; see
tests/test_hierarchical.py and scripts/sim_seed_inference.py). The t version
was at or below nominal level in every simulated setting. The percentile
interval and centred-bootstrap p-value (same convention as
``bootstrap.paired_bootstrap``) are kept as secondary output, and become
primary (``p_primary``) when every seed gives the same delta: deterministic
training has known zero seed variance, so no t correction is needed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as st


def cluster_weights(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-item multiplicities from an iid bootstrap over whole clusters."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    counts = np.bincount(rng.integers(0, len(uniq), size=len(uniq)), minlength=len(uniq))
    return counts[inv].astype(float)


def cell_weights(cells: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-item multiplicities from a bootstrap stratified by cell label."""
    w = np.zeros(len(cells))
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        w += np.bincount(idx[rng.integers(0, len(idx), size=len(idx))], minlength=len(cells))
    return w


def weighted_bacc(pred: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Balanced accuracy of each row of ``pred`` (seeds x items) under weights ``w``."""
    pos = w * y
    neg = w * (1 - y)
    sp, sn = pos.sum(), neg.sum()
    if sp == 0 or sn == 0:
        return np.full(pred.shape[0], np.nan)
    return 0.5 * (pred @ pos / sp + (1 - pred) @ neg / sn)


def group_baccs(pred: np.ndarray, y: np.ndarray, w: np.ndarray,
                masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-seed balanced accuracy overall, per masked group, worst group and gap (SBAG)."""
    out = {"overall": weighted_bacc(pred, y, w)}
    groups = []
    for name, m in masks.items():
        out[name] = weighted_bacc(pred[:, m], y[m], w[m])
        groups.append(out[name])
    if groups:
        stacked = np.vstack(groups)
        out["worst_group"] = np.min(stacked, axis=0)
        out["gap"] = np.max(stacked, axis=0) - out["worst_group"]
    return out


def hierarchical_paired(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    unit_weights: Callable[[np.random.Generator], np.ndarray],
    masks: dict[str, np.ndarray] | None = None,
    n_boot: int = 9999,
    seed: int = 0,
    resample_seeds: bool = True,
    level: float = 0.95,
) -> dict[str, dict]:
    """Paired comparison of training procedures A and B.

    ``pred_a`` and ``pred_b`` are (seeds x items) binary predictions; row s of
    each comes from the same training seed. Returns, per metric, the
    seed-averaged delta (A - B), its percentile interval, a two-sided p-value,
    the bootstrap SE and the fraction of degenerate draws.
    """
    y = np.asarray(y, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    if pred_a.shape != pred_b.shape or pred_a.shape[1] != len(y):
        raise ValueError("prediction matrices must both be (seeds x items) matching y")
    masks = masks or {}
    n_seeds = pred_a.shape[0]
    ones = np.ones(len(y))
    est_a = group_baccs(pred_a, y, ones, masks)
    est_b = group_baccs(pred_b, y, ones, masks)
    estimate = {k: float(np.mean(est_a[k] - est_b[k])) for k in est_a}
    per_seed = {k: (est_a[k] - est_b[k]).tolist() for k in est_a}

    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_boot) for k in estimate}
    for b in range(n_boot):
        w = unit_weights(rng)
        s = rng.integers(0, n_seeds, size=n_seeds) if resample_seeds else np.arange(n_seeds)
        da = group_baccs(pred_a[s], y, w, masks)
        db = group_baccs(pred_b[s], y, w, masks)
        for k in draws:
            draws[k][b] = np.mean(da[k] - db[k])

    alpha = 1 - level
    out = {}
    for k, d in draws.items():
        ok = d[np.isfinite(d)]
        est = estimate[k]
        if len(ok) == 0 or not np.isfinite(est):
            out[k] = {"delta": est, "primary": "none", "p_primary": np.nan,
                      "ci_primary": [np.nan, np.nan], "ci_t": [np.nan, np.nan], "p_t": np.nan,
                      "ci": [np.nan, np.nan], "p": np.nan, "se": np.nan,
                      "nan_fraction": 1.0, "per_seed_delta": per_seed[k]}
            continue
        lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        p = float(np.mean(np.abs(ok - est) >= abs(est)))
        se = float(ok.std(ddof=1))
        df = max(n_seeds - 1, 1)
        se_t = se * np.sqrt(n_seeds / df)
        half = float(st.t.ppf(1 - alpha / 2, df) * se_t)
        p_t = float(2 * st.t.sf(abs(est) / se_t, df)) if se_t > 0 else (0.0 if est else 1.0)
        # Deterministic training (identical per-seed deltas) has known zero
        # seed variance, so the t correction does not apply: use percentile.
        deterministic = bool(np.ptp(per_seed[k]) == 0)
        out[k] = {
            "delta": est,
            "primary": "percentile" if deterministic else "t",
            "p_primary": max(p, 1.0 / (len(ok) + 1)) if deterministic else p_t,
            "ci_primary": [float(lo), float(hi)] if deterministic else [est - half, est + half],
            "ci_t": [est - half, est + half],
            "p_t": p_t,
            "ci": [float(lo), float(hi)],
            "p": max(p, 1.0 / (len(ok) + 1)),
            "se": se,
            "nan_fraction": float(1 - len(ok) / n_boot),
            "per_seed_delta": per_seed[k],
        }
    out["_design"] = {"n_seeds": n_seeds, "n_items": int(len(y)), "n_boot": n_boot,
                      "resample_seeds": resample_seeds, "level": level, "boot_seed": seed}
    return out
