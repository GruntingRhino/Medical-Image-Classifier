"""Bootstrap and permutation procedures for paired/unpaired metric comparisons.

Design notes
------------
- All resampling is explicit and seeded; no global RNG state is used.
- Metric resampling uses *cell-stratified* bootstrap over (group, label) cells
  by default, which preserves class and group marginals within each draw and
  avoids degenerate (no-positive/no-negative) draws more often than iid
  resampling. Cell-stratified bootstrap is standard for balanced-accuracy-type
  metrics; iid bootstrap over clusters is provided for patient-clustered data.
- Paired comparisons resample the SAME indices for both methods, giving paired
  differences (correct for any dependence between methods on shared data).
- Zero-positive/zero-negative degenerate draws produce NaN; percentile
  intervals are computed over non-NaN draws and the NaN fraction is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import auroc, balanced_accuracy, subgroup_metrics


def _safe_metric(
    y_true: np.ndarray, y_pred: np.ndarray, metric: str
) -> float:
    try:
        if metric == "balanced_accuracy":
            return balanced_accuracy(y_true, y_pred)
        if metric == "auroc":
            return auroc(y_true, y_pred)
        raise KeyError(metric)
    except Exception:
        return np.nan


def cell_stratified_indices(
    groups: np.ndarray, y_true: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap indices resampled within each (group, label) cell."""
    idx_out = []
    for g in np.unique(groups):
        for c in (0, 1):
            cell = np.where((groups == g) & (y_true == c))[0]
            if len(cell) == 0:
                continue
            draw = rng.integers(0, len(cell), size=len(cell))
            idx_out.append(cell[draw])
    if not idx_out:
        raise ValueError("no non-empty cells")
    return np.concatenate(idx_out)


def iid_cluster_indices(
    clusters: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Cluster bootstrap: resample whole clusters (e.g., patients)."""
    uniq = np.unique(clusters)
    draw = rng.integers(0, len(uniq), size=len(uniq))
    parts = []
    # bucket cluster members once
    buckets: dict[object, np.ndarray] = {}
    for c in uniq:
        buckets[c] = np.where(clusters == c)[0]
    for c in draw:
        parts.append(buckets[uniq[c]])
    return np.concatenate(parts)


@dataclass
class BootstrapCI:
    metric: str
    estimate: float
    lower: float
    upper: float
    level: float
    n_draws: int
    nan_fraction: float
    method: str  # "cell_stratified" or "cluster_iid"


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "balanced_accuracy",
    groups: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for a metric on one (fixed) model."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    estimate = _safe_metric(y_true, y_pred, metric)

    draws = np.empty(n_boot)
    for b in range(n_boot):
        if clusters is not None:
            idx = iid_cluster_indices(np.asarray(clusters), rng)
            yt, yp = y_true[idx], y_pred[idx]
            if groups is not None:
                # metric on resampled clusters restricted to full-cluster draws
                draws[b] = _safe_metric(yt, yp, metric)
                continue
        elif groups is not None:
            idx = cell_stratified_indices(np.asarray(groups), y_true, rng)
        else:
            idx = rng.integers(0, n, size=n)
        draws[b] = _safe_metric(y_true[idx], y_pred[idx], metric)

    ok = ~np.isnan(draws)
    nan_fraction = float(np.mean(~ok))
    if not ok.any():
        return BootstrapCI(metric, estimate, np.nan, np.nan, level, n_boot, nan_fraction, "cluster_iid" if clusters is not None else ("cell_stratified" if groups is not None else "iid"))
    alpha = 1 - level
    lo, hi = np.percentile(draws[ok], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    method = "cluster_iid" if clusters is not None else ("cell_stratified" if groups is not None else "iid")
    return BootstrapCI(metric, float(estimate), float(lo), float(hi), level, n_boot, nan_fraction, method)


@dataclass
class PairedComparison:
    metric: str
    est_a: float
    est_b: float
    delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    level: float
    n_draws: int
    nan_fraction: float
    method: str
    alt: str


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str = "balanced_accuracy",
    groups: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    alt: str = "two-sided",
) -> PairedComparison:
    """Paired bootstrap comparison of two methods on the same test items.

    The same resampled index set is applied to both methods' predictions, so
    the delta distribution reflects paired differences. The p-value is a
    sign-flipping test on centered bootstrap deltas (approximates the
    permutation null of symmetric error around 0) combined with the percentile
    CI. For a strictly exchangeability-based p-value with two methods, the
    sign-flip test on deltas is appropriate when deltas are approximately
    iid-symmetric; validity is checked by the unit tests on synthetic data.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    est_a = _safe_metric(y_true, pred_a, metric)
    est_b = _safe_metric(y_true, pred_b, metric)
    delta = est_a - est_b

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        if clusters is not None:
            idx = iid_cluster_indices(np.asarray(clusters), rng)
        elif groups is not None:
            idx = cell_stratified_indices(np.asarray(groups), y_true, rng)
        else:
            idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        da = _safe_metric(yt, pred_a[idx], metric)
        db = _safe_metric(yt, pred_b[idx], metric)
        deltas[b] = da - db

    ok = ~np.isnan(deltas)
    nan_fraction = float(np.mean(~ok))
    dd = deltas[ok]
    method = "cluster_iid" if clusters is not None else ("cell_stratified" if groups is not None else "iid")
    if len(dd) == 0 or np.isnan(delta):
        return PairedComparison(metric, est_a, est_b, delta, np.nan, np.nan, np.nan, level, n_boot, nan_fraction, method, alt)

    alpha = 1 - level
    lo, hi = np.percentile(dd, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # sign-flip null distribution around observed delta
    centered = dd - delta
    signs = rng.choice([-1.0, 1.0], size=(len(centered),))
    null = np.abs(signs * centered)  # |delta_null| under symmetric null
    if alt == "two-sided":
        p = float(np.mean(null >= np.abs(delta)))
    elif alt == "greater":
        p = float(np.mean(dd >= delta))  # fraction of bootstrap draws supporting H1 direction
        p = float(np.mean(centered <= 0)) if False else p  # keep simple
    else:  # less
        p = float(np.mean(dd <= delta))
    p = min(1.0, max(p, 1.0 / (len(dd) + 1)))

    return PairedComparison(metric, float(est_a), float(est_b), float(delta), float(lo), float(hi), float(p), level, n_boot, nan_fraction, method, alt)


@dataclass
class SeedSummary:
    metric: str
    values: list[float]
    mean: float
    sd: float
    t_ci_lower: float
    t_ci_upper: float
    n_seeds: int


def seed_summary(values: list[float], metric: str = "", level: float = 0.95) -> SeedSummary:
    """Summary across training seeds (uncertainty due to model training)."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    n = len(v)
    if n == 0:
        return SeedSummary(metric, values, np.nan, np.nan, np.nan, np.nan, 0)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else np.nan
    if n > 1 and sd == sd:
        from scipy import stats as st

        tcrit = st.t.ppf(1 - (1 - level) / 2, df=n - 1)
        half = tcrit * sd / np.sqrt(n)
    else:
        half = np.nan
    return SeedSummary(metric, values, mean, sd, mean - half if half == half else np.nan, mean + half if half == half else np.nan, n)
