"""Why does proxy sampling hurt external (Fitzpatrick17k) performance?

Hypotheses tested against the stored caches and predictions:
H-a  Proxy distribution shift: the HAM10000 ITA proxy lives on dermoscopic
     (darkfield, lesion-filling) images; Fitzpatrick17k is clinical
     photography. Quantify the distribution overlap and where FST 5-6 images
     fall in HAM's ITA bins.
H-b  Disease prevalence shift: HAM melanoma prevalence (~11%) vs Fitz
     malignant prevalence (~53%) — a HAM-val threshold applied to a very
     different prior; per-group positive-prediction rates vs true prevalence.
H-c  Imaging-condition confound: border-patch statistics (background/skin
     framing) differ systematically; ITA may encode illumination, not skin.
H-d  Method mechanics: effective sample size of sampler weights; whether
     proxy bins in HAM upweight image categories (specular/bright) that
     anti-correlate with Fitz's dark-skin presentation.
H-e  FST 5-6 decomposition: sensitivity/specificity/B_acc/AUROC per FST level
     for ERM vs proxy methods (from aggregate_vision per_fst_level output).

All quantities computed from files on disk; no model retraining.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from ..data import ham10000
from ..sampling.methods import MethodConfig, fit_method

OUT = Path("results/audit/external_degradation.json")


def _dist(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()),
            "q10": float(np.quantile(x, 0.1)), "q50": float(np.quantile(x, 0.5)),
            "q90": float(np.quantile(x, 0.9))}


def _overlap(a, b, grid=200):
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, grid + 1)
    ha, _ = np.histogram(a, edges, density=True)
    hb, _ = np.histogram(b, edges, density=True)
    w = edges[1] - edges[0]
    return float(np.minimum(ha, hb).sum() * w)


def main():
    out = {}

    # ---------- proxy caches ----------
    ham_prox = pd.read_csv(ham10000.PROC / "proxies.csv")
    meta = ham10000.lesion_split(ham10000.load_metadata(), seed=20261111)
    meta = meta.merge(ham_prox, on="image_id", validate="1:1")
    ham_tr_ita = meta.loc[meta.split == "train", "ita_center"].to_numpy()
    fitz_meta = pd.read_csv(Path("data/processed/fitzpatrick17k/eval_malignant_benign_meta.csv"))
    fitz_prox = pd.read_csv(Path("data/processed/fitzpatrick17k/eval_malignant_benign_proxies.csv"))
    fitz = fitz_meta.merge(fitz_prox, on="md5hash", validate="1:1")
    fst = fitz.fst.to_numpy(dtype=float)
    fitz_ita = fitz.ita_center.to_numpy()

    # H-a: distribution shift + per-FST placement in HAM bins
    qs = np.quantile(ham_tr_ita, [0, .2, .4, .6, .8, 1.0])
    fitz_bin_frac = {f"bin{i}": float(((fitz_ita >= qs[i]) & (fitz_ita <= qs[i + 1])).mean())
                     for i in range(5)}
    fitz_bin_frac_fst56 = {f"bin{i}": float(((fitz_ita[fst >= 5] >= qs[i]) & (fitz_ita[fst >= 5] <= qs[i + 1])).mean())
                           for i in range(5)}
    out["proxy_distribution"] = {
        "ham_train_ita": _dist(ham_tr_ita),
        "fitz_eval_ita": _dist(fitz_ita),
        "overlap_coefficient_ham_vs_fitz": _overlap(ham_tr_ita, fitz_ita),
        "ham_quantile_bin_edges": [float(x) for x in qs],
        "fitz_fraction_per_ham_bin": fitz_bin_frac,
        "fitz_fst56_fraction_per_ham_bin": fitz_bin_frac_fst56,
        "ita_by_fst_level": {str(int(l)): _dist(fitz_ita[fst == l]) for l in range(1, 7)},
        "ita_unknown_fst0": _dist(fitz_ita[fst == 0]) if (fst == 0).any() else None,
    }
    # border-region contrast (illumination/background confound)
    for col in ("ita_border", "lum_med_border", "lum_med_center", "sat_center", "lum_p10_center"):
        out["proxy_distribution"].setdefault("border_center_contrast", {})[col] = {
            "ham_train": _dist(meta.loc[meta.split == "train", col]),
            "fitz_eval": _dist(fitz[col].to_numpy()),
            "overlap": _overlap(meta.loc[meta.split == "train", col], fitz[col].to_numpy()),
        }

    # H-b: prevalence
    y_fitz = fitz.y.to_numpy().astype(int)
    prev_by_fst = {str(int(l)): float(y_fitz[fst == l].mean()) for l in range(1, 7)}
    out["prevalence"] = {
        "ham_train_melanoma": float(meta.loc[meta.split == "train", "y"].mean()),
        "ham_val_melanoma": float(meta.loc[meta.split == "val", "y"].mean()),
        "ham_test_melanoma": float(meta.loc[meta.split == "test", "y"].mean()),
        "fitz_eval_malignant": float(y_fitz.mean()),
        "fitz_prevalence_by_fst": prev_by_fst,
    }

    # ---------- predictions ----------
    preds = json.loads((Path("results/preds") / "vision_ham10000-main.json").read_text())
    g = np.where((fst >= 5) & (fst <= 6), 1, np.where((fst >= 1) & (fst <= 2), 0, np.nan))

    # H-e + threshold-transfer per run (mean across seeds)
    from ..stats.metrics import balanced_accuracy
    rows = []
    for key, payload in preds.items():
        variant, method, seed = key.rsplit("::", 2)
        if variant != "corrupt_eps0.0":
            continue
        thr = payload["threshold"]
        s = np.asarray(payload["fitz_score"])
        pred = (s > thr).astype(int)
        rec = {"method": method, "seed": int(seed), "threshold": float(thr)}
        for name, m in (("dark", g == 1), ("light", g == 0), ("mid_unknown", np.isnan(g))):
            if not m.any():
                continue
            rec[f"pospred_{name}"] = float(pred[m].mean())
            rec[f"trueprev_{name}"] = float(y_fitz[m].mean())
            rec[f"bacc_{name}"] = float(balanced_accuracy(y_fitz[m], pred[m]))
            pm = pred[m]
            tp = int(((pm == 1) & (y_fitz[m] == 1)).sum()); fn = int(((pm == 0) & (y_fitz[m] == 1)).sum())
            tn = int(((pm == 0) & (y_fitz[m] == 0)).sum()); fp = int(((pm == 1) & (y_fitz[m] == 0)).sum())
            rec[f"sens_{name}"] = tp / (tp + fn) if tp + fn else None
            rec[f"spec_{name}"] = tn / (tn + fp) if tn + fp else None
        rows.append(rec)
    df = pd.DataFrame(rows)
    agg = df.groupby("method").mean(numeric_only=True).round(4).to_dict("index")
    out["threshold_transfer_and_group_decomposition"] = agg

    # per-FST-level BAcc delta vs ERM (mean across seeds), eps0.0
    from ..stats.metrics import auroc
    fst_rows = []
    for key, payload in preds.items():
        variant, method, seed = key.rsplit("::", 2)
        if variant != "corrupt_eps0.0":
            continue
        thr = payload["threshold"]
        s = np.asarray(payload["fitz_score"])
        pred = (s > thr).astype(int)
        for l in range(1, 7):
            m = fst == l
            if m.sum() == 0 or len(np.unique(y_fitz[m])) < 2:
                continue
            fst_rows.append({"method": method, "seed": int(seed), "fst": l,
                             "bacc": float(balanced_accuracy(y_fitz[m], pred[m])),
                             "auroc": float(auroc(y_fitz[m], s[m]))})
    fdf = pd.DataFrame(fst_rows)
    erm = fdf[fdf.method == "erm"].groupby("fst").mean(numeric_only=True)
    comp = {}
    for method in sorted(fdf.method.unique()):
        if method == "erm":
            continue
        md = fdf[fdf.method == method].groupby("fst").mean(numeric_only=True)
        comp[method] = {str(int(l)): {"dbacc": float(md.loc[l, "bacc"] - erm.loc[l, "bacc"]),
                                      "dauroc": float(md.loc[l, "auroc"] - erm.loc[l, "auroc"])}
                        for l in erm.index if l in md.index}
    out["per_fst_delta_vs_erm"] = comp

    # H-d: sampler-weight mechanics on the train split (eps0.0, seed 0)
    y_tr = meta.loc[meta.split == "train"].y
    rng = np.random.default_rng(20261111 + 1)
    tr = meta[meta.split == "train"].copy()
    tr["r"] = rng.random(len(tr))
    lesions = tr.groupby("lesion_id").agg(y=("y", "max"), r=("r", "first")).reset_index().sort_values("r")
    val_lesions = set(lesions.lesion_id.iloc[: int(round(0.1 * len(lesions)))])
    is_val = tr.lesion_id.isin(val_lesions).to_numpy()
    tr_rows = tr[~is_val]
    P_tr = tr_rows.ita_center.to_numpy()
    y_tr = tr_rows.y.to_numpy()
    ess = {}
    for method in ("erm", "class_strat", "proxy_strat", "proxy_strat_joint", "proxy_iw", "proxy_dro", "random_proxy_strat"):
        cfg = MethodConfig(name=method, n_bins=5, seed=0, joint_with_label=(method == "proxy_strat_joint"))
        w = fit_method(cfg, y_tr, P_tr, G_train=None).weights
        w = np.asarray(w, dtype=float)
        ess[method] = {"effective_sample_size": float(w.sum() ** 2 / (w ** 2).sum()),
                       "n_train": int(len(w)), "max_min_ratio": float(w.max() / w.min())}
    out["weight_mechanics"] = ess

    OUT.write_text(json.dumps(out, indent=2, default=str))
    # console digest
    pd_ = out["proxy_distribution"]
    print("ITA overlap HAM-vs-Fitz:", round(pd_["overlap_coefficient_ham_vs_fitz"], 3))
    print("Fitz share per HAM ITA bin:", {k: round(v, 3) for k, v in pd_["fitz_fraction_per_ham_bin"].items()})
    print("FST5-6 share per HAM ITA bin:", {k: round(v, 3) for k, v in pd_["fitz_fst56_fraction_per_ham_bin"].items()})
    print("prevalence: HAM train", round(out["prevalence"]["ham_train_melanoma"], 3),
          "| Fitz", round(out["prevalence"]["fitz_eval_malignant"], 3),
          "| by FST:", {k: round(v, 2) for k, v in out["prevalence"]["fitz_prevalence_by_fst"].items()})
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
