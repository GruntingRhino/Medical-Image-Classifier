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

    Thresholds are the midpoints between sorted unique validation scores (plus
    the extremes). Identical protocol is applied to every sampling method, so
    threshold selection never favors a method.
    """
    y_val = np.asarray(y_val, dtype=int)
    s = np.asarray(scores_val, dtype=float)
    uniq = np.unique(s)
    if len(uniq) == 1:
        return 0.5
    cands = np.concatenate([[-np.inf], (uniq[:-1] + uniq[1:]) / 2.0, [np.inf]])
    best_t, best_b = 0.5, -1.0
    for t in cands:
        pred = (s > t).astype(int)
        from ..stats.metrics import balanced_accuracy

        b = balanced_accuracy(y_val, pred)
        if b == b and b > best_b:
            best_b, best_t = b, float(t)
    return best_t


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
