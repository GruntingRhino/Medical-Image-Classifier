"""Sampling and weighting methods under demographic uncertainty.

Setting: training data has features X, binary target Y, and a proxy P
correlated with an unobserved demographic group G. The true group is available
only at evaluation.

Every method returns a nonnegative weight vector over the training rows
( or a metadata record for methods needing multi-stage handling). Models are
trained with sample weights, so all methods see identical data, architecture,
and optimization budget. Resampling is expressed as weights via a seeded
multinomial draw in ``resample_from_weights`` when a hard resample is needed.

Methods
-------
- ``erm``              : uniform weights (empirical risk minimization)
- ``class_strat``      : class-stratified representation (uses Y only)
- ``proxy_strat``      : quantile-bin P, equalize bin representation;
                         optional joint (bin, Y) equalization
- ``proxy_iw``         : importance weights 1 / p_hat(P bin) (inverse frequency)
- ``proxy_dro``        : group-DRO over proxy bins (exponential-weights
                         reweighting; implemented as multi-stage reweighting
                         handled by the training loop)
- ``random_proxy``     : proxy-shuffled control (destroys P-G association)
- ``confounded_proxy`` : user-supplied corrupted proxy (control)
- ``oracle_strat``     : stratify on TRUE group (reference upper bound only;
                         requires G at training time, never a fair method)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..proxies.quality import bin_proxy_equal_width, bin_proxy_quantiles


@dataclass
class MethodConfig:
    name: str
    n_bins: int = 5
    bin_mode: Literal["uniform_width", "quantile"] = "uniform_width"
    # "quantile" bins make bin-equalization a no-op (all bins equal count);
    # kept as an explicit degenerate control. "uniform_width" is the operative
    # variant: sparse minority-enriched extreme bins get upweighted.
    joint_with_label: bool = False   # equalize (bin, y) cells instead of bins
    target: Literal["uniform", "balanced_class"] = "uniform"
    smoothing: float = 0.0           # pseudo-counts for weight estimation
    cap_factor: float | None = None  # cap max/min weight ratio (None = no cap)
    dro_eta: float = 0.1             # DRO learning rate (per stage)
    dro_steps: int = 0               # >0 => iterative DRO handled by trainer
    seed: int = 0


@dataclass
class MethodOutput:
    name: str
    weights: np.ndarray
    bins: np.ndarray | None = None       # bin ids actually used
    bin_edges: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def _normalize(w: np.ndarray, cap_factor: float | None) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 0.0)
    if cap_factor is not None and len(w) > 0:
        hi = w.max()
        lo = w[w > 0].min() if (w > 0).any() else 1.0
        if lo > 0 and hi / lo > cap_factor:
            w = np.minimum(w, lo * cap_factor)
    s = w.sum()
    if s <= 0:
        raise ValueError("all-zero weights")
    return w / s


def resample_from_weights(
    weights: np.ndarray, size: int, seed: int
) -> np.ndarray:
    """Deterministic index resample according to weights (with replacement)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    return rng.choice(len(p), size=size, replace=True, p=p)


def _bin_ids(
    P: np.ndarray, cfg: MethodConfig, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    P = np.asarray(P)
    if P.dtype.kind in "iu" and len(np.unique(P)) <= cfg.n_bins:
        return P.astype(int), None
    if cfg.bin_mode == "quantile":
        ids, e = bin_proxy_quantiles(P, cfg.n_bins, edges)
    else:
        ids, e = bin_proxy_equal_width(P, cfg.n_bins, edges)
    return ids, e


def fit_method(
    cfg: MethodConfig,
    y: np.ndarray,
    P: np.ndarray | None,
    *,
    G_train: np.ndarray | None = None,
) -> MethodOutput:
    """Compute sample weights for a method.

    Parameters
    ----------
    cfg   : method configuration.
    y     : training labels (0/1).
    P     : training proxy (continuous or binned). Required by all proxy methods.
    G_train : true group for TRAINING rows; only used by ``oracle_strat``
              (reference method). Must be None for every fair method.
    """
    y = np.asarray(y)
    n = len(y)
    name = cfg.name

    if name == "erm":
        w = np.ones(n)
        return MethodOutput(name, _normalize(w, None))

    if name == "class_strat":
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        w = np.where(y == 1, 0.5 / max(n1, 1), 0.5 / max(n0, 1))
        return MethodOutput(name, _normalize(w, cfg.cap_factor))

    if name in {"proxy_strat", "proxy_strat_joint", "proxy_iw", "proxy_dro", "random_proxy_strat", "oracle_strat"}:
        if name == "random_proxy_strat":
            rng = np.random.default_rng(cfg.seed)
            P = np.asarray(P, dtype=float)
            shuffled = rng.permutation(P)
            _shuffled_proxy = shuffled
            P = shuffled
        else:
            _shuffled_proxy = None
        if name == "oracle_strat":
            if G_train is None:
                raise ValueError("oracle_strat requires G_train")
            bins = np.asarray(G_train).astype(int)
            edges = None
        else:
            if P is None:
                raise ValueError(f"{name} requires a proxy P")
            bins, edges = _bin_ids(P, cfg)

        uniq = np.unique(bins)
        if cfg.joint_with_label:
            cells = np.unique(np.stack([bins, y], axis=1), axis=0)
            # target: equal mass per (bin, y) cell
            target_mass = 1.0 / len(cells)
            counts = {
                (int(cb), int(cy)): float(np.sum((bins == cb) & (y == cy)))
                for cb, cy in cells
            }
            w = np.empty(n)
            for i in range(n):
                key = (int(bins[i]), int(y[i]))
                c = counts[key]
                w[i] = (target_mass / c) if c > 0 else 0.0
        else:
            target_mass = 1.0 / len(uniq)
            counts = {int(b): float(np.sum(bins == b)) for b in uniq}
            w = np.empty(n)
            for i in range(n):
                c = counts[int(bins[i])]
                w[i] = (target_mass / c) if c > 0 else 0.0

        if name == "proxy_iw":
            # stabilized importance weighting: inverse empirical bin density,
            # CAPPED at cap_factor (default 10) to bound the weight variance
            # (standard practice; uncapped IW is high-variance in sparse bins)
            cap = cfg.cap_factor if cfg.cap_factor is not None else 10.0
            w = np.empty(n)
            for b in uniq:
                c = counts[int(b)]
                w[bins == b] = min(1.0 / c, cap / n)
        if name == "proxy_dro":
            # single-stage fallback; run_sims uses dro_steps>0 for the
            # iterative exponential-weights variant
            w = np.empty(n)
            for b in uniq:
                c = counts[int(b)]
                w[bins == b] = 1.0 / c

        return MethodOutput(
            name,
            _normalize(w, cfg.cap_factor),
            bins=bins,
            bin_edges=edges,
            meta={
                "bins": int(len(uniq)),
                "counts": {str(k): v for k, v in counts.items()},
                **({"shuffled_proxy": _shuffled_proxy} if _shuffled_proxy is not None else {}),
            },
        )

    raise KeyError(f"unknown method {name!r}")


def dro_update_weights(
    losses_per_row: np.ndarray,
    bins: np.ndarray,
    weights: np.ndarray,
    eta: float,
) -> np.ndarray:
    """One exponential-weights group-DRO update over proxy-bin groups.

    Losses are per-row training losses from the current model. Bin weights
    ``group_w`` are maintained implicitly through row weights: after update,
    rows in bins with higher average loss get more weight.
    """
    losses = np.asarray(losses_per_row, dtype=float)
    bins = np.asarray(bins)
    w = np.asarray(weights, dtype=float).copy()
    uniq = np.unique(bins)
    # current group distribution implied by row weights
    row_dist = w / w.sum()
    group_mass = np.array([row_dist[bins == b].sum() for b in uniq])
    group_loss = np.array(
        [
            np.average(losses[bins == b], weights=row_dist[bins == b])
            if (bins == b).any() and group_mass[i] > 0
            else np.mean(losses[bins == b]) if (bins == b).any() else 0.0
            for i, b in enumerate(uniq)
        ]
    )
    # soft-min over groups: increase mass on high-loss groups
    tilts = np.exp(eta * group_loss / max(np.mean(group_loss), 1e-12))
    new_group_mass = group_mass * tilts
    new_group_mass = new_group_mass / new_group_mass.sum()
    # spread group mass uniformly over rows in each bin
    new_w = np.empty_like(w)
    for i, b in enumerate(uniq):
        m = bins == b
        k = int(m.sum())
        new_w[m] = new_group_mass[i] / max(k, 1)
    return _normalize(new_w, None)
