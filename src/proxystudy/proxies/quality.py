"""Proxy quality metrics: association with true demographics and calibration.

Functions here quantify how well a training-time proxy P represents a
demographic group G that is only available at evaluation time:

- Cohen's kappa for binned/discrete proxy vs binary group
- Normalized mutual information (NMI)
- AUROC of P for G (continuous or ordinal proxy)
- Per-bin group purity and counts (calibration table)
- Attenuation ratio used in the simulation analysis
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two discrete labelings over a SHARED category set.

    Both inputs must take values in the same category space (e.g., both
    binary). To score a multi-level proxy against a binary group, dichotomize
    the proxy first (see ``dichotomize_proxy``).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("length mismatch")
    cats_a, cats_b = set(np.unique(a).tolist()), set(np.unique(b).tolist())
    if cats_a != cats_b:
        raise ValueError(
            f"category spaces differ: {sorted(cats_a)} vs {sorted(cats_b)}; dichotomize first"
        )
    n = len(a)
    po = float(np.mean(a == b))
    pe = 0.0
    for c in sorted(cats_a):
        pe += (np.mean(a == c)) * (np.mean(b == c))
    if pe == 1.0:
        return np.nan
    return (po - pe) / (1 - pe)


def dichotomize_proxy(proxy: np.ndarray, cut: float | None = None) -> np.ndarray:
    """Dichotomize a continuous/ordinal proxy at ``cut`` (default: median).

    Already-binary proxies pass through unchanged. If the median cut produces
    a single category (heavily skewed discrete proxy), fall back to the mean;
    if that also collapses, return the identity encoding.
    """
    proxy = np.asarray(proxy, dtype=float)
    if len(np.unique(proxy)) <= 2:
        return proxy.astype(int)
    if cut is None:
        cut = float(np.median(proxy))
    d = (proxy >= cut).astype(int)
    if len(np.unique(d)) == 1:
        d = (proxy > float(np.mean(proxy))).astype(int)
    if len(np.unique(d)) == 1:
        d = proxy.astype(int)
    return d


def normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """NMI in [0,1], arithmetic-mean normalization."""
    a = np.asarray(a)
    b = np.asarray(b)
    cats_a = np.unique(a)
    cats_b = np.unique(b)
    n = len(a)
    mi = 0.0
    for ca in cats_a:
        pa = np.mean(a == ca)
        for cb in cats_b:
            pab = np.mean((a == ca) & (b == cb))
            pb = np.mean(b == cb)
            if pab > 0:
                mi += pab * np.log(pab / (pa * pb))
    h_a = st.entropy(np.bincount(np.searchsorted(cats_a, a)))
    h_b = st.entropy(np.bincount(np.searchsorted(cats_b, b)))
    denom = (h_a + h_b) / 2
    return float(mi / denom) if denom > 0 else np.nan


def proxy_auc_for_group(proxy: np.ndarray, group: np.ndarray, positive: int = 1) -> float:
    """AUROC of the proxy as a score for group membership."""
    from ..stats.metrics import auroc

    return auroc((group == positive).astype(int), np.asarray(proxy, dtype=float))


def bin_proxy_quantiles(
    proxy: np.ndarray, n_bins: int, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-FREQUENCY (quantile) bins into [0, n_bins).

    NOTE: with equal-frequency bins, every bin holds ~n/B rows, so methods
    that "equalize bin mass" become a no-op (weights ~ uniform). Equal-width
    bins (``bin_proxy_equal_width``) are the variant under which proxy
    stratification has any effect; keep the quantile variant as an explicit
    degenerate control.
    """
    proxy = np.asarray(proxy, dtype=float)
    if edges is None:
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(proxy, qs))
    return np.searchsorted(edges, proxy, side="right"), edges


def bin_proxy_equal_width(
    proxy: np.ndarray, n_bins: int, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-WIDTH bins over the observed proxy range (edges from train data).

    Bins away from the mode are sparse; if the proxy correlates with the
    minority group, sparse extreme bins are minority-enriched, and equalizing
    bin mass increases effective minority representation. Ties beyond the
    last edge collapse into the top bin; degenerate ranges (all-equal proxy)
    yield a single bin.
    """
    proxy = np.asarray(proxy, dtype=float)
    if edges is None:
        lo, hi = float(np.min(proxy)), float(np.max(proxy))
        if hi <= lo:
            edges = np.array([])
        else:
            edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
            edges = np.unique(edges)
    return np.searchsorted(edges, proxy, side="right"), edges


def proxy_calibration_table(
    proxy_bin: np.ndarray, group: np.ndarray, positive: int = 1
) -> dict:
    """Per-bin group composition: P(G=positive | P bin = b), counts."""
    proxy_bin = np.asarray(proxy_bin)
    group = np.asarray(group)
    rows = []
    for b in np.unique(proxy_bin):
        m = proxy_bin == b
        gpos = group[m] == positive
        rows.append(
            {
                "bin": int(b),
                "n": int(m.sum()),
                "n_group_positive": int(gpos.sum()),
                "purity": float(np.mean(gpos)) if m.sum() > 0 else np.nan,
            }
        )
    overall = {
        "n": int(len(group)),
        "overall_positive_rate": float(np.mean(group == positive)),
        "bins": rows,
    }
    return overall


def proxy_quality_summary(
    proxy: np.ndarray, group: np.ndarray, n_bins: int = 5, positive: int = 1
) -> dict:
    """One-stop summary of proxy informativeness about the true group.

    Kappa is computed on the median-dichotomized proxy (both inputs binary);
    NMI is computed on the n_bins quantile binning vs the group (handles
    differing cardinalities); AUC uses the raw proxy as a score.
    """
    proxy = np.asarray(proxy, dtype=float)
    group = np.asarray(group)
    if len(np.unique(proxy)) <= 20:
        binned = proxy.astype(int)
    else:
        binned, _ = bin_proxy_quantiles(proxy, n_bins)
    dichot = dichotomize_proxy(proxy)
    return {
        "kappa": cohens_kappa(dichot, group),
        "nmi": normalized_mutual_info(binned, group),
        "auc_for_group": proxy_auc_for_group(proxy, group, positive),
        "calibration": proxy_calibration_table(binned, group, positive),
        "n_bins_effective": int(len(np.unique(binned))),
    }
