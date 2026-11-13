"""Vision experiment aggregation v2 — corrected groups + cluster-aware inference.

Changes vs v1 (rationale recorded in results/audit/):
- Group membership recomputed from metadata (fst: dark = 5-6, light = 1-2;
  fst==0 is UNKNOWN (original -1) and is excluded, together with FST 3-4,
  from worst-group statistics). v1 trusted stored fitz_g, which misassigned
  104 unknown-FST images to the light group.
- HAM10000 BAcc inference uses a LESION-CLUSTER bootstrap (resampling
  lesions, keeping all their images together) — test images of the same
  lesion are not independent.
- Training-seed uncertainty is reported as a t-interval over per-seed paired
  deltas (seeds are training replicates, never counted as patients).
- Cross-seed combination: Fisher's method over per-seed p-values; Holm-
  Bonferroni across the method family per (variant, metric).
- Per-FST-level (1-6) decomposition, per-group sensitivity/specificity, and
  lesion-aggregated exploratory comparisons are included for the
  external-degradation investigation.
- Fitzpatrick17k items are single images (no patient clusters): cell-
  stratified bootstrap by (group, label) is appropriate there.

DeLong tests are image-level (no closed-form cluster version); the lesion-
cluster bootstrap over AUROC deltas is provided as the cluster-aware check.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from ..data import ham10000
from ..stats.bootstrap import iid_cluster_indices, paired_bootstrap
from ..stats.inference import delong_paired_test, holm_bonferroni, mcnemar_exact
from ..stats.metrics import auroc, balanced_accuracy, confusion

PREDS_DIR = Path("results/preds")
RECS_DIR = Path("results/vision_ham10000")
OUT_DIR = Path("results/vision_summary")


def _fisher(pvals: list[float]) -> float:
    p = np.asarray([p for p in pvals if np.isfinite(p)], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    if len(p) == 0:
        return np.nan
    return float(st.chi2.sf(-2.0 * np.log(p).sum(), df=2 * len(p)))


def _t_ci(vals: list[float], level: float = 0.95):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return None
    if n == 1:
        return {"mean": float(v[0]), "sd": None, "n": 1, "lo": None, "hi": None}
    sd = float(v.std(ddof=1))
    tcrit = st.t.ppf(1 - (1 - level) / 2, df=n - 1)
    half = tcrit * sd / np.sqrt(n)
    return {"mean": float(v.mean()), "sd": sd, "n": n, "lo": float(v.mean() - half), "hi": float(v.mean() + half)}


def _sens_spec(y, pred):
    tp, fp, tn, fn = confusion(y, pred)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return float(sens), float(spec)


def load_context():
    meta = ham10000.lesion_split(ham10000.load_metadata(), seed=20261111)
    y_ham = meta.loc[meta.split == "test", "y"].to_numpy().astype(int)
    lesion_ham = meta.loc[meta.split == "test", "lesion_id"].to_numpy()
    fitz = pd.read_csv(Path("data/processed/fitzpatrick17k/eval_malignant_benign_meta.csv"))
    fst = fitz.fst.to_numpy(dtype=float)
    y_fitz = fitz.y.to_numpy().astype(int)
    g_fitz = np.where((fst >= 5) & (fst <= 6), 1, np.where((fst >= 1) & (fst <= 2), 0, np.nan))
    return y_ham, lesion_ham, y_fitz, g_fitz, fst


def _group_metrics(y, pred, score, g):
    out = {}
    mask = ~np.isnan(g)
    sm_dark, sm_light = g == 1, g == 0
    baccs = {}
    for name, m in (("dark", sm_dark), ("light", sm_light)):
        baccs[name] = float(balanced_accuracy(y[m], pred[m]))
        s_, p_ = _sens_spec(y[m], pred[m])
        out[f"{name}_sens"] = s_
        out[f"{name}_spec"] = p_
    out["bacc_dark"] = baccs["dark"]
    out["bacc_light"] = baccs["light"]
    out["bacc_worst"] = float(np.nanmin(list(baccs.values())))
    out["bacc_overall"] = float(balanced_accuracy(y, pred))
    out["auroc_overall"] = float(auroc(y, score))
    out["auroc_dark"] = float(auroc(y[sm_dark], score[sm_dark]))
    out["auroc_light"] = float(auroc(y[sm_light], score[sm_light]))
    return out


def _per_fst_level(y, pred, score, fst):
    rows = {}
    for lvl in (1, 2, 3, 4, 5, 6):
        m = fst == lvl
        if not m.any():
            continue
        s_, p_ = _sens_spec(y[m], pred[m])
        rows[str(lvl)] = {
            "n": int(m.sum()),
            "prevalence": float(y[m].mean()),
            "bacc": float(balanced_accuracy(y[m], pred[m])) if len(np.unique(y[m])) == 2 else None,
            "sens": s_, "spec": p_,
            "auroc": float(auroc(y[m], score[m])) if len(np.unique(y[m])) == 2 else None,
        }
    m0 = fst == 0
    if m0.any():
        rows["0_unknown"] = {"n": int(m0.sum()), "prevalence": float(y[m0].mean()),
                             "bacc": float(balanced_accuracy(y[m0], pred[m0])) if len(np.unique(y[m0])) == 2 else None}
    return rows


def aggregate(tag: str, n_boot: int = 1000, boot_seed: int = 7) -> dict:
    y_ham, lesion_ham, y_fitz, g_fitz, fst_fitz = load_context()
    preds = json.loads((PREDS_DIR / f"vision_ham10000-{tag}.json").read_text())
    grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for key, payload in preds.items():
        variant, method, seed = key.rsplit("::", 2)
        grouped[(variant, method)][int(seed)] = payload
    # records: val_bacc + stored threshold cross-check
    recs = {}
    for p in sorted(RECS_DIR.glob("*.json")):
        r = json.loads(p.read_text())
        if r["config"].get("tag") != tag:
            continue
        c = r["config"]
        recs[(c["variant"], c["method"], r["seed"])] = r

    variants = sorted({v for (v, _) in grouped})
    methods = sorted({m for (_, m) in grouped} - {"erm"})

    # absolute per-seed metrics + seed summaries
    abs_metrics: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for (variant, method), by_seed in grouped.items():
        for s, payload in by_seed.items():
            thr = payload["threshold"]
            pred_f = (np.asarray(payload["fitz_score"]) > thr).astype(int)
            abs_metrics[(variant, method)][s] = _group_metrics(y_fitz, pred_f, np.asarray(payload["fitz_score"]), g_fitz)

    seed_tables = {}
    for key, by_seed in abs_metrics.items():
        rows = {}
        for metric in ("bacc_overall", "auroc_overall", "bacc_dark", "bacc_light",
                       "bacc_worst", "dark_sens", "dark_spec", "light_sens", "light_spec"):
            rows[metric] = _t_ci([m[metric] for m in by_seed.values()])
        rec_key = (key[0], key[1])
        vals_val, vals_fst34 = [], []
        for s in by_seed:
            r = recs.get((key[0], key[1], s))
            if r and r["metrics"].get("val_bacc") is not None:
                vals_val.append(r["metrics"]["val_bacc"])
            if r and r["metrics"].get("fitz_bacc_fst34_unknown") is not None:
                vals_fst34.append(r["metrics"]["fitz_bacc_fst34_unknown"])
        if vals_val:
            rows["ham_val_bacc"] = _t_ci(vals_val)
        if vals_fst34:
            rows["fitz_bacc_fst34_unknown"] = _t_ci(vals_fst34)
        seed_tables["::".join(key)] = rows

    # per-FST-level tables (exploratory, mean across seeds)
    fst_tables = {}
    for key, by_seed in abs_metrics.items():
        per_seed = [_per_fst_level(y_fitz, (np.asarray(grouped[key][s]["fitz_score"]) > grouped[key][s]["threshold"]).astype(int),
                                   np.asarray(grouped[key][s]["fitz_score"]), fst_fitz) for s in sorted(by_seed)]
        agg = {}
        for lvl in per_seed[0]:
            agg[lvl] = {k: _t_ci([ps[lvl][k] for ps in per_seed if k in ps[lvl] and ps[lvl][k] is not None])
                        for k in per_seed[0][lvl]}
        fst_tables["::".join(key)] = agg

    # paired vs ERM
    paired_tables = {}
    for variant in variants:
        erm = grouped.get((variant, "erm"), {})
        if not erm:
            continue
        fam = {}
        for method in methods:
            ours = grouped.get((variant, method), {})
            seeds = sorted(set(erm) & set(ours))
            if not seeds:
                continue
            pvals = defaultdict(list)
            deltas = defaultdict(list)
            for s in seeds:
                a, b = ours[s], erm[s]
                thr_a, thr_b = a["threshold"], b["threshold"]
                sa, sb = np.asarray(a["fitz_score"]), np.asarray(b["fitz_score"])
                sha, shb = np.asarray(a["ham_test_score"]), np.asarray(b["ham_test_score"])
                pa_f, pb_f = (sa > thr_a).astype(int), (sb > thr_b).astype(int)
                pa_h, pb_h = (sha > thr_a).astype(int), (shb > thr_b).astype(int)

                # HAM: lesion-cluster paired bootstrap on BAcc delta
                rng = np.random.default_rng(boot_seed + s)
                cl = np.empty(n_boot)
                for bi in range(n_boot):
                    idx = iid_cluster_indices(lesion_ham, rng)
                    cl[bi] = (balanced_accuracy(y_ham[idx], pa_h[idx])
                              - balanced_accuracy(y_ham[idx], pb_h[idx]))
                d_ham = float(balanced_accuracy(y_ham, pa_h) - balanced_accuracy(y_ham, pb_h))
                ok = cl[np.isfinite(cl)]
                ci_lo, ci_hi = np.percentile(ok, [2.5, 97.5])
                signs = np.random.default_rng(boot_seed + s + 100).choice([-1.0, 1.0], size=len(ok))
                p_ham = float(np.mean(np.abs(signs * (ok - d_ham)) >= abs(d_ham)))
                p_ham = max(p_ham, 1 / (len(ok) + 1))

                dl_ham = delong_paired_test(y_ham, sha, shb)
                mc_ham = mcnemar_exact(y_ham, pa_h, pb_h)

                # Fitz: cell-stratified (group x label) paired bootstrap
                strata = np.where(np.isnan(g_fitz), -1, g_fitz).astype(int) * 2 + y_fitz
                pb_fitz = paired_bootstrap(y_fitz, pa_f, pb_f, "balanced_accuracy",
                                           groups=strata, n_boot=n_boot, seed=boot_seed + s + 200)
                dl_fitz = delong_paired_test(y_fitz, sa, sb)

                m_dark, m_light = g_fitz == 1, g_fitz == 0
                pb_dark = paired_bootstrap(y_fitz[m_dark], pa_f[m_dark], pb_f[m_dark], "balanced_accuracy",
                                           groups=y_fitz[m_dark], n_boot=n_boot, seed=boot_seed + s + 300)
                pb_light = paired_bootstrap(y_fitz[m_light], pa_f[m_light], pb_f[m_light], "balanced_accuracy",
                                            groups=y_fitz[m_light], n_boot=n_boot, seed=boot_seed + s + 400)
                dl_dark = delong_paired_test(y_fitz[m_dark], sa[m_dark], sb[m_dark])
                dl_light = delong_paired_test(y_fitz[m_light], sa[m_light], sb[m_light])

                ga = _group_metrics(y_fitz, pa_f, sa, g_fitz)
                gb = _group_metrics(y_fitz, pb_f, sb, g_fitz)

                # exploratory lesion-level aggregation (one unit per lesion,
                # score = mean over its test images) — removes the cluster
                # dependence entirely at the cost of discarding within-lesion
                # information; descriptive, outside the Holm family.
                uq, inv = np.unique(lesion_ham, return_inverse=True)
                ya = np.zeros(len(uq))
                s_a_l = np.zeros(len(uq))
                s_b_l = np.zeros(len(uq))
                np.add.at(ya, inv, y_ham)
                np.add.at(s_a_l, inv, sha)
                np.add.at(s_b_l, inv, shb)
                cnt = np.bincount(inv).astype(float)
                ya = (ya / cnt > 0.5).astype(int)
                s_a_l /= cnt
                s_b_l /= cnt
                multi = cnt > 1  # keep lesions with >1 image (the clustered ones)
                if multi.sum() >= 50:
                    pb_les = paired_bootstrap(ya[multi], (s_a_l[multi] > thr_a).astype(int),
                                              (s_b_l[multi] > thr_b).astype(int), "balanced_accuracy",
                                              n_boot=n_boot, seed=boot_seed + s + 500)
                    deltas["lesion_bacc_multi"].append(pb_les.delta)
                    pvals["lesion_bacc_multi"].append(pb_les.p_value)
                    dl_les = delong_paired_test(ya[multi], s_a_l[multi], s_b_l[multi])
                    deltas["lesion_auroc_multi"].append(dl_les["delta"])
                    pvals["lesion_auroc_multi"].append(dl_les["p_value"])

                for name, p, d in (
                    ("ham_bacc", p_ham, d_ham),
                    ("ham_auroc", dl_ham["p_value"], dl_ham["delta"]),
                    ("ham_mcnemar", mc_ham["p_value"], np.nan),
                    ("fitz_bacc", pb_fitz.p_value, pb_fitz.delta),
                    ("fitz_auroc", dl_fitz["p_value"], dl_fitz["delta"]),
                    ("fitz_bacc_dark", pb_dark.p_value, pb_dark.delta),
                    ("fitz_bacc_light", pb_light.p_value, pb_light.delta),
                    ("fitz_auroc_dark", dl_dark["p_value"], dl_dark["delta"]),
                    ("fitz_auroc_light", dl_light["p_value"], dl_light["delta"]),
                    ("fitz_dark_sens", None, ga["dark_sens"] - gb["dark_sens"]),
                    ("fitz_dark_spec", None, ga["dark_spec"] - gb["dark_spec"]),
                    ("fitz_light_sens", None, ga["light_sens"] - gb["light_sens"]),
                    ("fitz_light_spec", None, ga["light_spec"] - gb["light_spec"]),
                    ("fitz_worst_group", None, ga["bacc_worst"] - gb["bacc_worst"]),
                ):
                    if p is not None:
                        pvals[name].append(p)
                    deltas[name].append(d)

            fam[method] = {
                "n_seeds": len(seeds),
                "p_fisher": {k: _fisher(v) for k, v in pvals.items()},
                "delta_seed_summary": {k: _t_ci(v) for k, v in deltas.items()},
            }
        metric_names = sorted({k for m in fam.values() for k in m["p_fisher"]})
        holm = {}
        for metric in metric_names:
            pvals_raw = [fam[m]["p_fisher"].get(metric, np.nan) for m in sorted(fam)]
            adj = holm_bonferroni([p if np.isfinite(p) else 1.0 for p in pvals_raw])
            holm[metric] = dict(zip(sorted(fam), adj))
        paired_tables[variant] = {"methods": fam, "holm": holm}

    out = {"tag": tag, "n_boot": n_boot, "seed_summaries": seed_tables,
           "per_fst_level": fst_tables, "paired_vs_erm": paired_tables,
           "context": {"ham_test_n": int(len(y_ham)), "ham_test_lesions": int(len(np.unique(lesion_ham))),
                       "fitz_n": int(len(y_fitz)), "fitz_dark": int(np.nansum(g_fitz == 1)),
                       "fitz_light": int(np.nansum(g_fitz == 0)),
                       "fitz_mid_or_unknown": int(np.isnan(g_fitz).sum())}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"vision-{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    out = aggregate(args.tag, n_boot=args.n_boot)
    for variant, tab in out["paired_vs_erm"].items():
        print(f"\n=== {variant} (vs ERM; Holm-adj Fisher p in (); cluster/CB bootstrap for BAcc) ===")
        print(f"{'method':<22}{'ham_bacc':>18}{'fitz_bacc':>18}{'fitz_dark':>18}{'fitz_light':>18}{'worst':>10}")
        for method, row in tab["methods"].items():
            ds = row["delta_seed_summary"]
            def cell(name, holm_name=None, no_holm=False):
                v = ds.get(name)
                if not v:
                    return "n/a"
                if no_holm:
                    return f"{v['mean']:+.4f}"
                h = tab["holm"].get(holm_name or name, {}).get(method)
                return f"{v['mean']:+.4f}({h:.2g})" if h is not None else f"{v['mean']:+.4f}"
            print(f"{method:<22}{cell('ham_bacc'):>18}{cell('fitz_bacc'):>18}"
                  f"{cell('fitz_bacc_dark'):>18}{cell('fitz_bacc_light'):>18}"
                  f"{cell('fitz_worst_group', no_holm=True):>10}")
    print(f"\nsaved -> {OUT_DIR / ('vision-' + args.tag + '.json')}")


if __name__ == "__main__":
    main()
