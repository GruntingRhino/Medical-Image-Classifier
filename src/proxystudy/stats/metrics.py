"""Core metrics for subgroup classification analysis.

All metric functions are pure, dependency-light (numpy only), and explicitly
handle undefined cells (empty groups, zero positives/negatives) by returning
NaN and flagging the reason. Validation against scikit-learn lives in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _check_binary(y: np.ndarray, name: str) -> np.ndarray:
    """Validate that values are in {0,1} (both classes need not be present,
    since subgroup slices may contain a single class)."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    yf = y.astype(float, copy=False)
    if np.isnan(yf).any() or not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError(f"{name} must contain only 0/1 values")
    return y


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) for binary predictions."""
    y_true = _check_binary(y_true, "y_true")
    y_pred = _check_binary(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, _, _, fn = confusion(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom > 0 else np.nan


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, tn, _ = confusion(y_true, y_pred)
    denom = tn + fp
    return tn / denom if denom > 0 else np.nan


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of sensitivity and specificity. NaN if either is undefined."""
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    if np.isnan(sens) or np.isnan(spec):
        return np.nan
    return 0.5 * (sens + spec)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic with tie handling."""
    y_true = _check_binary(y_true, "y_true")
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("shape mismatch")
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    # Average ranks within ties
    sorted_scores = y_score[order]
    i = 0
    n = len(y_score)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r1 = ranks[y_true == 1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


@dataclass
class SubgroupMetrics:
    """Metrics computed within demographic groups.

    NaN values indicate undefined metrics (empty group or a class with zero
    members); ``undefined_groups`` records why, per group.
    """

    groups: list[int] = field(default_factory=list)
    group_sizes: dict[int, int] = field(default_factory=dict)
    group_bacc: dict[int, float] = field(default_factory=dict)
    group_sens: dict[int, float] = field(default_factory=dict)
    group_spec: dict[int, float] = field(default_factory=dict)
    group_auroc: dict[int, float] = field(default_factory=dict)
    undefined_groups: dict[int, str] = field(default_factory=dict)
    worst_group_bacc: float = np.nan
    best_group_bacc: float = np.nan
    sbag: float = np.nan  # subgroup balanced-accuracy gap: best - worst


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    y_score: np.ndarray | None = None,
) -> SubgroupMetrics:
    """Per-group metrics, worst-group balanced accuracy, and SBAG.

    ``groups`` holds the true demographic label for every test item. Groups are
    the sorted unique values. A group whose sensitivity or specificity is
    undefined (empty class within the group) is excluded from worst/best/SBAG
    and recorded in ``undefined_groups``.
    """
    y_true = _check_binary(y_true, "y_true")
    y_pred = _check_binary(y_pred, "y_pred")
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("length mismatch")

    out = SubgroupMetrics()
    baccs: list[float] = []
    for g in np.unique(groups):
        g = int(g)
        m = groups == g
        out.groups.append(g)
        out.group_sizes[g] = int(m.sum())
        b = balanced_accuracy(y_true[m], y_pred[m])
        s = sensitivity(y_true[m], y_pred[m])
        p = specificity(y_true[m], y_pred[m])
        if np.isnan(b):
            if s == s:  # sensitivity defined, specificity not
                out.undefined_groups[g] = "specificity_undefined_no_negatives"
            elif p == p:
                out.undefined_groups[g] = "sensitivity_undefined_no_positives"
            else:
                out.undefined_groups[g] = "both_undefined_empty_or_degenerate"
        out.group_bacc[g] = b
        out.group_sens[g] = s
        out.group_spec[g] = p
        if y_score is not None:
            out.group_auroc[g] = auroc(y_true[m], y_score[m])
        if not np.isnan(b):
            baccs.append(b)
    if baccs:
        out.worst_group_bacc = float(np.min(baccs))
        out.best_group_bacc = float(np.max(baccs))
        out.sbag = out.best_group_bacc - out.worst_group_bacc
    return out


def to_dict(m: SubgroupMetrics) -> dict:
    d = {
        "groups": m.groups,
        "group_sizes": m.group_sizes,
        "group_bacc": m.group_bacc,
        "group_sens": m.group_sens,
        "group_spec": m.group_spec,
        "group_auroc": m.group_auroc,
        "undefined_groups": m.undefined_groups,
        "worst_group_bacc": m.worst_group_bacc,
        "best_group_bacc": m.best_group_bacc,
        "sbag": m.sbag,
    }
    # JSON-safe rounding of floats
    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    return _clean(d)  # type: ignore[arg-type]


def get_metric_fn(name: str) -> Callable:
    registry: dict[str, Callable] = {
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
    }
    if name not in registry:
        raise KeyError(f"unknown metric {name!r}; available: {sorted(registry)}")
    return registry[name]
