"""Controlled simulation DGP (v3) with known ground truth.

Feature blocks
--------------
- v-block (first half of features): carries the class signal.
- u-block (second half): nuisance/acquisition dimensions (e.g., luminance,
  lighting, contrast) that are label-independent for the majority group but
  NOT for the minority group.

Generative process
------------------
- G ~ Bern(p_g): true demographic group (1 = minority).
- Z ~ Bern(p_z): acquisition confounder (e.g., imaging protocol). Z shifts the
  u-block for everyone (``zfeat``) and shifts label prevalence (case-mix
  confounding: ``P(Y=1|Z=1) = min(1, prev[g] * z_label_mult)``).
- Y ~ Bern(prev[g] * z_mult^Z).
- X:
    v-block mean = beta0 * (1 - signal_gap * G) * y
      (minority class signal attenuated: signal_gap > 0)
    u-block mean = shift * G + nuisance_gamma * G * y + zfeat * Z
      (minority mean offset; minority positives differ along the nuisance
      direction; Z shifts everyone)

Why this creates a genuine directional disparity
------------------------------------------------
- For G=0 (majority), u carries no label information: the optimal classifier
  ignores u (coefficient 0 on the u-block) and thresholds the v-projection.
- For G=1 (minority), u DOES carry label information (nuisance_gamma != 0):
  the optimal coefficient vector includes a u-component.
- A shared linear classifier must compromise. ERM, dominated by the majority,
  fits ~0 u-coefficient and a majority threshold => minority balanced
  accuracy suffers.
- Upweighting minority-enriched samples moves the shared coefficient vector
  toward the minority optimum => genuine worst-group improvement, at a cost
  to majority performance (quantified trade-off).
- A proxy confounded through Z (reliability * G + confound * Z) upweights
  Z-shifted rows when equalizing bins; if Z also distorts the u-block
  (``zfeat``) or case mix (``z_label_mult``), this introduces bias => harm.

Ground truth: full G (and Z) known for every simulated item at evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimParams:
    n: int = 2000
    n_test: int = 100_000
    p_group: float = 0.25               # P(G=1), minority fraction
    prevalence: tuple[float, float] = (0.15, 0.25)  # P(Y=1|G=0), P(Y=1|G=1)
    beta0: float = 0.6                  # majority class separation (per feature)
    signal_gap: float = 0.5             # minority separation = beta0*(1-gap)
    shift: float = 0.4                  # minority u-block mean offset
    nuisance_gamma: float = 0.25        # minority label-dependence along u
    zfeat: float = 0.0                  # Z shift along u (lighting artifact)
    z_label_mult: float = 1.0           # P(y=1|Z=1) = prev * z_label_mult (cap 1)
    feature_noise: float = 1.0
    n_features: int = 8                 # first half v-block, second half u-block
    reliability: float = 0.5            # weight of G in the proxy
    proxy_noise_sd: float = 0.25
    confound: float = 0.0               # weight of Z in the proxy
    p_z: float = 0.3
    n_bins: int = 5
    model: str = "logistic"             # "logistic" | "hgb"
    seed: int = 0

    def validate(self) -> None:
        assert 0 < self.p_group < 1
        assert 0 <= self.reliability <= 1
        assert 0 <= self.confound <= 1
        assert 0 <= self.signal_gap <= 1
        assert self.n_features % 2 == 0
        assert self.model in {"logistic", "hgb"}


@dataclass
class SimDataset:
    X: np.ndarray
    y: np.ndarray
    G: np.ndarray
    P: np.ndarray
    Z: np.ndarray | None
    meta: dict = field(default_factory=dict)


def _sample(params: SimParams, rng: np.random.Generator, n: int):
    d = params.n_features
    dv = d // 2
    G = (rng.random(n) < params.p_group).astype(int)
    Z = (rng.random(n) < params.p_z).astype(int)
    prev = np.where(G == 1, params.prevalence[1], params.prevalence[0])
    prev = np.minimum(1.0, prev * np.where(Z == 1, params.z_label_mult, 1.0))
    y = (rng.random(n) < prev).astype(int)

    mean_v = (params.beta0 * (1 - params.signal_gap * G) * y)[:, None]
    mean_u = (
        params.shift * G
        + params.nuisance_gamma * G * y
        + params.zfeat * Z
    )[:, None]
    mean = np.concatenate(
        [np.repeat(mean_v, dv, axis=1), np.repeat(mean_u, d - dv, axis=1)], axis=1
    )
    X = mean + params.feature_noise * rng.standard_normal((n, d))
    U01 = rng.random(n)
    P = (
        params.reliability * G
        + (1 - params.reliability) * U01
        + params.confound * Z
        + params.proxy_noise_sd * rng.standard_normal(n)
    )
    return X, y, G, P, Z


def generate(params: SimParams) -> SimDataset:
    params.validate()
    rng = np.random.default_rng(params.seed)
    X_tr, y_tr, G_tr, P_tr, Z_tr = _sample(params, rng, params.n)
    X_te, y_te, G_te, P_te, Z_te = _sample(params, rng, params.n_test)
    return SimDataset(
        X_tr, y_tr, G_tr, P_tr, Z_tr,
        meta={"X_test": X_te, "y_test": y_te, "G_test": G_te, "P_test": P_te, "Z_test": Z_te},
    )
