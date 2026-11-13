"""Hierarchical (test-unit x seed) inference for every stored comparison vs ERM.

Replaces the Fisher-over-seeds combination used by aggregate_vision /
aggregate_tabular, which treats per-seed tests on the same test items as
independent (see stats/hierarchical.py). Reads only stored predictions; no
model is retrained and no threshold is re-selected (thresholds are the ones
chosen on internal validation and stored with each run).

Resampling units:
- HAM10000 test: lesion clusters (images of one lesion stay together).
- Fitzpatrick17k: single images, bootstrap stratified by (FST group, label).
- Adult / German: single rows, bootstrap stratified by (group, label).

Multiplicity: Holm within each (dataset, variant, metric) family across
methods (as in the original protocol), plus a global Holm over every
primary-endpoint test reported here, as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..stats.hierarchical import cell_weights, cluster_weights, hierarchical_paired
from ..stats.inference import holm_bonferroni
from .aggregate_tabular import load as load_tabular
from .aggregate_vision import PREDS_DIR, load_context

OUT = Path("results/hier_summary")
# Primary endpoints, fixed from the CHECKLIST hypotheses before this analysis:
# H1 HAM internal overall BAcc, H2 Fitzpatrick17k FST 5-6 BAcc, H3 tabular gap.
PRIMARY = {"vision": [("ham", "overall"), ("fitz", "dark")],
           "adult": [("test", "gap")], "german": [("test", "gap")]}


def _matrices(by_seed: dict, field: str, thr_field: str | None = None):
    seeds = sorted(by_seed)
    rows = []
    for s in seeds:
        p = by_seed[s]
        if thr_field is None:
            rows.append(np.asarray(p[field], dtype=float))
        else:
            rows.append((np.asarray(p[field]) > p[thr_field]).astype(float))
    return seeds, np.vstack(rows)


def _compare(groups, y, units, masks, field, thr_field, n_boot, boot_seed):
    out = {}
    variants = sorted({v for (v, _) in groups})
    methods = sorted({m for (_, m) in groups} - {"erm"})
    for variant in variants:
        erm = groups.get((variant, "erm"))
        if not erm:
            continue
        out[variant] = {}
        for i, method in enumerate(methods):
            ours = groups.get((variant, method))
            if not ours:
                continue
            common = sorted(set(ours) & set(erm))
            sa, a = _matrices({s: ours[s] for s in common}, field, thr_field)
            _, b = _matrices({s: erm[s] for s in common}, field, thr_field)
            res = hierarchical_paired(y, a, b, units, masks=masks, n_boot=n_boot,
                                      seed=boot_seed + 1000 * i + len(variant))
            res["_design"]["seeds"] = sa
            out[variant][method] = res
    return out


def _holm(block: dict, scope: str) -> None:
    """Holm across methods within each (variant, metric); writes p_holm."""
    for variant, by_method in block.items():
        metrics = sorted({k for r in by_method.values() for k in r if not k.startswith("_")})
        for metric in metrics:
            names = [m for m in sorted(by_method) if metric in by_method[m]]
            raw = [by_method[m][metric]["p_primary"] for m in names]
            adj = holm_bonferroni([p if np.isfinite(p) else 1.0 for p in raw])
            for m, a in zip(names, adj):
                by_method[m][metric]["p_holm_family"] = float(a)
                by_method[m][metric]["family"] = f"{scope}/{variant}/{metric}"


def run_vision(n_boot: int, boot_seed: int) -> dict:
    y_ham, lesion_ham, y_fitz, g_fitz, _ = load_context()
    preds = json.loads((PREDS_DIR / "vision_ham10000-main.json").read_text())
    groups: dict = {}
    for key, payload in preds.items():
        variant, method, seed = key.rsplit("::", 2)
        groups.setdefault((variant, method), {})[int(seed)] = payload
    fitz_cells = np.where(np.isnan(g_fitz), -1, g_fitz).astype(int) * 2 + y_fitz
    masks = {"dark": g_fitz == 1, "light": g_fitz == 0}
    ham = _compare(groups, y_ham, lambda r: cluster_weights(lesion_ham, r), {},
                   "ham_test_score", "threshold", n_boot, boot_seed)
    fitz = _compare(groups, y_fitz, lambda r: cell_weights(fitz_cells, r), masks,
                    "fitz_score", "threshold", n_boot, boot_seed + 1)
    _holm(ham, "vision/ham")
    _holm(fitz, "vision/fitz")
    return {"ham": ham, "fitz": fitz,
            "units": {"ham": "lesion clusters", "fitz": "images stratified by (FST group, label)"}}


def run_tabular(dataset: str, n_boot: int, boot_seed: int) -> dict:
    grouped, y, g = load_tabular(dataset, "main")
    cells = g * 2 + y
    masks = {f"group_{k}": g == k for k in (0, 1)}
    test = _compare(grouped, y, lambda r: cell_weights(cells, r), masks,
                    "pred", None, n_boot, boot_seed)
    _holm(test, f"{dataset}/test")
    return {"test": test, "units": "rows stratified by (group, label)"}


def global_holm(results: dict) -> list[dict]:
    rows = []
    for dataset, endpoints in PRIMARY.items():
        for split, metric in endpoints:
            for variant, by_method in results[dataset][split].items():
                for method, r in by_method.items():
                    if metric in r:
                        rows.append({"dataset": dataset, "split": split, "variant": variant,
                                     "method": method, "metric": metric, "delta": r[metric]["delta"],
                                     "primary": r[metric]["primary"], "ci": r[metric]["ci_primary"],
                                     "p": r[metric]["p_primary"]})
    adj = holm_bonferroni([r["p"] if np.isfinite(r["p"]) else 1.0 for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm_global"] = float(a)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--boot-seed", type=int, default=20261113)
    args = ap.parse_args(argv)
    results = {"vision": run_vision(args.n_boot, args.boot_seed),
               "adult": run_tabular("adult", args.n_boot, args.boot_seed + 7),
               "german": run_tabular("german", args.n_boot, args.boot_seed + 11)}
    results["primary_global_holm"] = global_holm(results)
    results["_meta"] = {"n_boot": args.n_boot, "boot_seed": args.boot_seed,
                        "primary_endpoints": PRIMARY,
                        "inference": "hierarchical paired bootstrap; t(S-1) on inflated bootstrap SE, percentile when seeds are deterministic"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hier-main.json").write_text(json.dumps(results, indent=1, default=float))
    sig = [r for r in results["primary_global_holm"] if r["p_holm_global"] < 0.05]
    print(f"primary tests: {len(results['primary_global_holm'])}; significant after global Holm: {len(sig)}")
    for r in sig:
        print(r)


if __name__ == "__main__":
    main()
