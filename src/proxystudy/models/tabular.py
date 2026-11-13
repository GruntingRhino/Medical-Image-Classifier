"""Model training for simulations and tabular experiments.

All methods are trained with sample weights so every sampling method gets an
identical architecture, optimization budget, and hyperparameter sweep.
Only validation data is used for hyperparameter selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..sampling.methods import dro_update_weights


@dataclass
class TabularModelConfig:
    model: str = "logistic"  # "logistic" | "hgb"
    C: float = 1.0
    max_leaf_nodes: int = 15
    max_iter: int = 300
    dro_eta: float = 0.1
    dro_steps: int = 0       # 0 = single-stage; >0 = iterative DRO over epochs
    seed: int = 0


def _base(cfg: TabularModelConfig):
    if cfg.model == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, random_state=cfg.seed),
        )
    if cfg.model == "hgb":
        return HistGradientBoostingClassifier(
            max_leaf_nodes=cfg.max_leaf_nodes,
            max_iter=cfg.max_iter,
            random_state=cfg.seed,
            early_stopping=False,
        )
    raise KeyError(cfg.model)


def _fit_estimator(est, X, y, sample_weight):
    """sample_weight passthrough that handles sklearn Pipeline naming."""
    if sample_weight is None:
        est.fit(X, y)
        return
    try:
        est.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        # Pipeline: route to final step
        steps = getattr(est, "named_steps", None) or {}
        if steps:
            final_name = list(steps)[-1]
            est.fit(X, y, **{f"{final_name}__sample_weight": sample_weight})
        else:
            raise


def fit_tabular(
    cfg: TabularModelConfig,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    bins: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
):
    """Fit with optional iterative proxy-DRO reweighting.

    Returns the fitted estimator. If ``cfg.dro_steps > 0`` and ``bins`` is
    provided, the estimator is refit ``dro_steps`` times, each time reweighting
    proxy-bin groups by their latest validation- or train-loss ranking.
    """
    model = _base(cfg)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float).copy()
    _fit_estimator(model, X, y, w)

    if cfg.dro_steps > 0:
        if bins is None:
            raise ValueError("iterative DRO requires proxy bins")
        for _ in range(cfg.dro_steps):
            # loss surface: use training log-loss per row (constant model class)
            p = model.predict_proba(X)[:, 1]
            eps = 1e-12
            losses = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            w = dro_update_weights(losses, bins, w if w is not None else np.ones(len(y)), cfg.dro_eta)
            model = _base(cfg)
            _fit_estimator(model, X, y, w)
    return model


def select_threshold_bacc(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Decision threshold maximizing validation balanced accuracy.

    Vectorized: sort once and sweep cumulative class counts. Thresholds are
    the midpoints between sorted unique validation scores. Identical protocol
    is applied to every sampling method, so threshold selection never favors
    a method.
    """
    y_val = np.asarray(y_val, dtype=int)
    s = np.asarray(scores_val, dtype=float)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y_val[order]
    n = len(s)
    if n == 0:
        return 0.5
    # candidate thresholds: midpoints between consecutive unique scores
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = s_sorted[1:] != s_sorted[:-1]
    # cumulative sums evaluated AFTER each unique-score block:
    # pred=1 for scores > t  <=>  items after the block boundary
    block_end = np.where(np.concatenate([is_new[1:], [True]]))[0]  # last idx of each block
    cum_pos = np.cumsum(y_sorted)
    tot_pos = cum_pos[-1]
    tot_neg = n - tot_pos
    # for a cut after block k (0-indexed), predicted positive = items beyond cut
    # tp = positives beyond cut, fp = negatives beyond cut
    pos_before = cum_pos[block_end]          # positives at or before cut
    neg_before = (block_end + 1) - pos_before
    tp = tot_pos - pos_before
    fp = tot_neg - neg_before
    fn = tot_pos - tp
    tn = tot_neg - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.where(tot_pos > 0, tp / tot_pos, np.nan)
        spec = np.where(tot_neg > 0, tn / tot_neg, np.nan)
    bacc = (sens + spec) / 2
    valid = ~np.isnan(bacc)
    if not valid.any():
        return 0.5
    k = int(np.nanargmax(bacc))
    cut_idx = block_end[k]
    if cut_idx >= n - 1:
        return float(s_sorted[-1] + 1.0)
    t = (s_sorted[cut_idx] + s_sorted[cut_idx + 1]) / 2.0
    return float(t)


def sweep_C(
    X_tr, y_tr, w_tr, X_val, y_val, seed: int, Cs=(0.03, 0.1, 0.3, 1.0, 3.0)
) -> tuple[float, float]:
    """Validation sweep for logistic C; returns (best_C, best_val_loss)."""
    best = (None, np.inf)
    for C in Cs:
        cfg = TabularModelConfig(model="logistic", C=C, seed=seed)
        m = fit_tabular(cfg, X_tr, y_tr, w_tr)
        p = m.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, p, labels=[0, 1])
        if ll < best[1]:
            best = (C, ll)
    return best


def sweep_hgb(
    X_tr, y_tr, w_tr, X_val, y_val, seed: int, leafs=(7, 15, 31), iters=(100, 200)
) -> tuple[tuple[int, int], float]:
    best = (None, np.inf)
    for L in leafs:
        for it in iters:
            cfg = TabularModelConfig(model="hgb", max_leaf_nodes=L, max_iter=it, seed=seed)
            m = fit_tabular(cfg, X_tr, y_tr, w_tr)
            p = m.predict_proba(X_val)[:, 1]
            ll = log_loss(y_val, p, labels=[0, 1])
            if ll < best[1]:
                best = ((L, it), ll)
    return best
