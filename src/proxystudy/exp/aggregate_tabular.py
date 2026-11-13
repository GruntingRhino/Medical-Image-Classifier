"""Aggregate tabular experiment results: seed summaries + paired inference vs ERM.

Mirrors ``aggregate_vision`` for the Adult / German experiments. Reads stored
test predictions (``results/preds/tabular-<dataset>-<tag>.json``, written by
``run_tabular --store-preds``) and per-run records.

Protocol
--------
- Seed summaries (mean, sd, t-CI) for every metric, per (variant, method).
- Paired comparisons vs ERM on identical test items, per seed:
  * BAcc: cell-stratified paired bootstrap (strata = true group).
  * AUROC: paired DeLong.
  * Hard predictions: exact McNemar.
- Cross-seed combination: Fisher's method over per-seed p-values.
- Multiplicity: Holm-Bonferroni across the method family (excluding ERM),
  per variant and metric family. oracle_strat is reported but never
  counted as a competing method (reference only).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats as st

from .ledger import RESULTS_DIR
from ..stats.bootstrap import paired_bootstrap
from ..stats.inference import delong_paired_test, holm_bonferroni, mcnemar_exact
from ..stats.metrics import auroc, balanced_accuracy

PREDS_DIR = RESULTS_DIR / "preds"
OUT_DIR = RESULTS_DIR / "tabular_summary"


def _fisher(pvals: list[float]) -> float:
    p = np.asarray([p for p in pvals if np.isfinite(p)], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    if len(p) == 0:
        return np.nan
    chi2 = -2.0 * np.log(p).sum()
    return float(st.chi2.sf(chi2, df=2 * len(p)))


def load(dataset: str, tag: str) -> tuple[dict, dict, dict]:
    payload = json.loads((PREDS_DIR / f"{dataset}-{tag}.json").read_text())
    y = np.asarray(payload["y_test"], dtype=int)
    g = np.asarray(payload["g_test"], dtype=int)
    grouped: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for key, slot in payload["variants"].items():
        variant, method = key.split("::", 1)
        for i, seed in enumerate(slot["seeds"]):
            grouped[(variant, method)][int(seed)] = {
                "pred": np.asarray(slot["preds"][i]),
                "score": np.asarray(slot["scores"][i]),
            }
    return grouped, y, g


def _metrics_from_pred(y, g, pred, score) -> dict:
    """Overall + per-group metrics from one run's stored predictions."""
    out = {
        "overall_bacc": float(balanced_accuracy(y, pred)),
        "auroc": float(auroc(y, score)),
    }
    baccs = {}
    for gg in sorted(np.unique(g)):
        m = g == gg
        if m.sum() and len(np.unique(y[m])) == 2:
            baccs[int(gg)] = float(balanced_accuracy(y[m], pred[m]))
    out["group_bacc"] = baccs
    if baccs:
        out["worst_group_bacc"] = float(min(baccs.values()))
        out["best_group_bacc"] = float(max(baccs.values()))
        out["sbag"] = float(max(baccs.values()) - min(baccs.values()))
    return out


def aggregate(dataset: str, tag: str, n_boot: int = 1000, boot_seed: int = 7) -> dict:
    grouped, y, g = load(dataset, tag)
    if not grouped:
        raise SystemExit(f"no stored predictions for {dataset}/{tag}; run run_tabular --store-preds")

    # ---------- seed summaries (computed from stored predictions) ----------
    metric_names = ("overall_bacc", "auroc", "worst_group_bacc", "sbag", "group_bacc")
    seed_tables = {}
    per_seed_metrics: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for key, by_seed in sorted(grouped.items()):
        for s, payload in by_seed.items():
            per_seed_metrics[key][s] = _metrics_from_pred(y, g, payload["pred"], payload["score"])
        rows = {}
        for metric in metric_names:
            if metric == "group_bacc":
                vals = {gg: [m["group_bacc"][gg] for m in per_seed_metrics[key].values()
                             if gg in m["group_bacc"]] for gg in (0, 1)}
                for gg, v in vals.items():
                    if v:
                        rows[f"group_{gg}_bacc"] = dict(
                            mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v))
                continue
            vals = [m[metric] for m in per_seed_metrics[key].values() if metric in m]
            if vals:
                rows[metric] = dict(
                    mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    n=len(vals),
                )
        seed_tables["::".join(key)] = rows

    # ---------- paired tests vs ERM from stored predictions ----------
    paired_tables = {}
    variants = sorted({v for (v, _) in grouped})
    methods = sorted({m for (_, m) in grouped} - {"erm"})
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
                pa, pb = np.asarray(a["pred"]), np.asarray(b["pred"])
                sa, sb = np.asarray(a["score"]), np.asarray(b["score"])
                pb_all = paired_bootstrap(y, pa, pb, "balanced_accuracy",
                                          groups=g, n_boot=n_boot, seed=boot_seed + s)
                dl = delong_paired_test(y, sa, sb)
                mc = mcnemar_exact(y, pa, pb)
                pvals["bacc"].append(pb_all.p_value)
                deltas["bacc"].append(float(pb_all.delta))
                pvals["auroc"].append(dl["p_value"])
                deltas["auroc"].append(float(dl["delta"]))
                pvals["mcnemar"].append(mc["p_value"])
                # subgroup deltas per seed (point estimates)
                for m_, mask in (("group_0_bacc", g == 0), ("group_1_bacc", g == 1)):
                    if mask.sum() and len(np.unique(y[mask])) == 2:
                        deltas[m_].append(
                            float(balanced_accuracy(y[mask], pa[mask])
                                  - balanced_accuracy(y[mask], pb[mask])))
                wg_a = float(np.nanmin([balanced_accuracy(y[g == gg], pa[g == gg]) for gg in (0, 1) if (g == gg).any()]))
                wg_b = float(np.nanmin([balanced_accuracy(y[g == gg], pb[g == gg]) for gg in (0, 1) if (g == gg).any()]))
                deltas["worst_group_bacc"].append(wg_a - wg_b)
            fam[method] = {
                "n_seeds": len(seeds),
                "p_fisher": {k: _fisher(v) for k, v in pvals.items()},
                "delta_mean": {k: (float(np.mean(deltas[k])) if deltas[k] else None) for k in deltas},
                "delta_sd": {k: (float(np.std(deltas[k], ddof=1)) if len(deltas[k]) > 1 else None) for k in deltas},
            }
            # seed-level t-CIs (training-seed variability of the paired delta)
            t_crit = {1: np.nan, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                      7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 20: 2.086}
            fam[method]["delta_seed_summary"] = {}
            for k, vals in deltas.items():
                n_ = len(vals)
                if n_ == 0:
                    continue
                mean = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if n_ > 1 else np.nan
                half = (t_crit.get(n_, 2.0) * sd / np.sqrt(n_)) if n_ > 1 and np.isfinite(sd) else np.nan
                fam[method]["delta_seed_summary"][k] = {
                    "mean": mean, "sd": sd, "n_seeds": n_,
                    "t_ci_lo": mean - half, "t_ci_hi": mean + half,
                    "values": [float(v) for v in vals]}
        metric_names_p = sorted({k for m in fam.values() for k in m["p_fisher"]})
        holm = {}
        for metric in metric_names_p:
            pvals = [fam[m]["p_fisher"].get(metric, np.nan) for m in sorted(fam)]
            adj = holm_bonferroni([0.0 if not np.isfinite(p) else p for p in pvals])
            holm[metric] = {m: adjv for m, adjv in zip(sorted(fam), adj)}
        paired_tables[variant] = {"methods": fam, "holm": holm}

    out = {"dataset": dataset, "tag": tag, "n_boot": n_boot,
           "seed_summaries": seed_tables, "paired_vs_erm": paired_tables}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"tabular-{dataset}-{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "german"], required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    out = aggregate(args.dataset, args.tag, n_boot=args.n_boot)

    for variant, tab in out["paired_vs_erm"].items():
        print(f"\n=== {args.dataset}/{variant} (vs ERM: delta(fisher p), holm-adj in ()) ===")
        print(f"{'method':<22}{'bacc':>16}{'auroc':>16}{'worst_group':>16}{'g0':>12}{'g1':>12}")
        for method, row in tab["methods"].items():
            d, h = row["delta_mean"], tab["holm"]
            def cell(name, needs_p=True):
                delta = d.get(name)
                if delta is None:
                    return "n/a"
                if not needs_p:
                    return f"{delta:+.3f}"
                adj = h.get(name, {}).get(method)
                if adj is None:
                    return "n/a"
                return f"{delta:+.3f}({adj:.2g})"
            print(f"{method:<22}{cell('bacc'):>16}{cell('auroc'):>16}"
                  f"{cell('worst_group_bacc', needs_p=False):>16}{cell('group_0_bacc', needs_p=False):>12}{cell('group_1_bacc', needs_p=False):>12}")
    print(f"\nsaved -> {OUT_DIR / ('tabular-' + args.dataset + '-' + args.tag + '.json')}")


if __name__ == "__main__":
    main()
