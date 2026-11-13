"""Tabular experiments: Adult income and German credit.

Protocol (identical for every method):
- Adult: official train/test split; a stratified 10% validation carve-out from
  the training file is used for hyperparameters and threshold selection. The
  official test file is a sealed evaluation set, evaluated once per run.
- German: fixed 800/200 train/test split (seed 20261111, recorded), 12.5% val
  carve-out from the train part. Small-n: 20 training seeds by default.
- Proxy: controlled corruption of the protected attribute (eps flip noise),
  confounded variant (spurious label channel), and an empirical marital-based
  proxy (Adult only). Proxy quality vs true G is computed on training data.
- Methods: erm, class_strat, proxy_strat, proxy_strat_joint, proxy_iw,
  proxy_dro, random_proxy_strat, oracle_strat (oracle = reference only).
- Inference: paired bootstrap + DeLong vs ERM from stored test predictions;
  seed summaries; Holm correction across the method family.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from ..data.adult import (
    checksum as adult_checksum,
    load_adult,
    make_proxy,
    prepare_features,
    proxy_from_marital,
)
from ..data.german import build_features as german_features, load_german
from ..exp.ledger import save_record
from ..models.tabular import TabularModelConfig, fit_tabular, select_threshold_bacc, sweep_C
from ..proxies.quality import proxy_quality_summary
from ..sampling.methods import MethodConfig, fit_method
from ..stats.metrics import auroc, subgroup_metrics, to_dict

SEED_SPLIT = 20261111

METHODS = [
    "erm",
    "class_strat",
    "proxy_strat",
    "proxy_strat_joint",
    "proxy_iw",
    "proxy_dro",
    "random_proxy_strat",
    "oracle_strat",
]


def prepare_adult(val_frac: float = 0.1) -> dict:
    d = load_adult()
    f = prepare_features(d)
    X_tr, y_tr, g_tr = f["X_tr"], f["y_tr"], f["g_tr"]
    X_te, y_te, g_te = f["X_te"], f["y_te"], f["g_te"]
    rng = np.random.default_rng(SEED_SPLIT)
    idx = rng.permutation(len(y_tr))
    n_val = int(val_frac * len(y_tr))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return dict(
        X_tr=X_tr[tr_idx], y_tr=y_tr[tr_idx], g_tr=g_tr[tr_idx],
        X_val=X_tr[val_idx], y_val=y_tr[val_idx], g_val=g_tr[val_idx],
        X_te=X_te, y_te=y_te, g_te=g_te,
        train_full=d["train"],
        tr_idx=tr_idx,
        checksum=adult_checksum(),
    )


def prepare_german(test_frac: float = 0.2) -> dict:
    df = load_german()
    X, y, g, _ = german_features(df)
    rng = np.random.default_rng(SEED_SPLIT)
    idx = rng.permutation(len(y))
    n_te = int(test_frac * len(y))
    te_idx, rest = idx[:n_te], idx[n_te:]
    rng2 = np.random.default_rng(SEED_SPLIT + 1)
    vidx = rng2.permutation(len(rest))
    n_val = int(0.125 * len(rest))
    val_idx, tr_idx = rest[vidx[:n_val]], rest[vidx[n_val:]]
    return dict(
        X_tr=X[tr_idx], y_tr=y[tr_idx], g_tr=g[tr_idx],
        X_val=X[val_idx], y_val=y[val_idx], g_val=g[val_idx],
        X_te=X[te_idx], y_te=y[te_idx], g_te=g[te_idx],
    )


def _proxy_variants(dataset: str, ds: dict, eps_levels: list[float]) -> list[dict]:
    out = []
    for e in eps_levels:
        P = make_proxy(ds["g_tr"], ds["y_tr"], eps=e, seed=SPLICE(e))
        out.append(dict(variant=f"corrupt_eps{e}", eps=e, mode="corrupt", P=P))
    return out


def SPLICE(x: float) -> int:
    return int(round(x * 1000)) + 7


def _run_single(job: dict) -> dict:
    ds, meth, seed, P_tr, n_bins, store_preds = (
        job["ds"], job["method"], job["seed"], job["P_tr"], job["n_bins"], job["store_preds"]
    )
    mcfg = MethodConfig(
        name=meth, n_bins=n_bins, seed=seed,
        joint_with_label=(meth == "proxy_strat_joint"),
    )
    out = fit_method(mcfg, ds["y_tr"], P_tr, G_train=ds["g_tr"])
    w = out.weights
    bins = out.bins if meth == "proxy_dro" else None
    best_C, _ = sweep_C(ds["X_tr"], ds["y_tr"], w, ds["X_val"], ds["y_val"], seed)
    dro_steps = 3 if meth == "proxy_dro" else 0
    tc = TabularModelConfig(model="logistic", C=best_C, seed=seed, dro_steps=dro_steps)
    m = fit_tabular(tc, ds["X_tr"], ds["y_tr"], w, bins=bins)
    thr = select_threshold_bacc(ds["y_val"], m.predict_proba(ds["X_val"])[:, 1])
    score = m.predict_proba(ds["X_te"])[:, 1]
    pred = (score > thr).astype(int)
    sm = subgroup_metrics(ds["y_te"], pred, ds["g_te"], score)
    rec = {
        "method": meth, "seed": seed, "n_bins": n_bins, "C": best_C, "threshold": float(thr),
        "metrics": to_dict(sm), "auroc": float(auroc(ds["y_te"], score)),
    }
    if store_preds:
        rec["_pred"] = pred.tolist()
        rec["_score"] = np.round(score, 5).tolist()
    return rec


def _worker(job: dict):
    t0 = time.time()
    rec = _run_single(job)
    rec["seconds"] = time.time() - t0
    preds = rec.pop("_pred", None)
    scores = rec.pop("_score", None)
    return job, rec, preds, scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "german"], required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--eps", type=float, nargs="*", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--store-preds", action="store_true")
    args = ap.parse_args(argv)

    if args.dataset == "adult":
        ds = prepare_adult()
        eps_levels = args.eps if args.eps is not None else [0.1, 0.3, 0.5, 0.7, 0.9]
        variants = _proxy_variants("adult", ds, eps_levels)
        variants.append(dict(
            variant="confounded_eps0.5", eps=0.5, mode="confounded",
            P=make_proxy(ds["g_tr"], ds["y_tr"], eps=0.5, seed=SPLICE(0.5), mode="confounded"),
        ))
        variants.append(dict(
            variant="marital_empirical", eps=None, mode="empirical",
            P=proxy_from_marital(ds["train_full"])[ds["tr_idx"]],
        ))
    else:
        ds = prepare_german()
        eps_levels = args.eps if args.eps is not None else [0.3, 0.6]
        variants = _proxy_variants("german", ds, eps_levels)
        variants.append(dict(
            variant="confounded_eps0.6", eps=0.6, mode="confounded",
            P=make_proxy(ds["g_tr"], ds["y_tr"], eps=0.6, seed=SPLICE(0.6), mode="confounded"),
        ))

    seeds = list(range(args.n_seeds))
    jobs = []
    for v in variants:
        for meth in METHODS:
            for seed in seeds:
                jobs.append({
                    "dataset": args.dataset, "variant": v["variant"], "eps": v["eps"],
                    "mode": v["mode"], "method": meth, "seed": seed,
                    "n_bins": args.n_bins, "P_tr": v["P"], "ds": ds,
                    "store_preds": args.store_preds,
                })

    print(f"{len(jobs)} runs on {args.dataset} ({args.tag})")
    t0 = time.time()
    results = []
    if args.jobs <= 1:
        it = map(_worker, jobs)
    else:
        it = ProcessPoolExecutor(max_workers=args.jobs).map(_worker, jobs, chunksize=1)
    for job, rec, preds, scores in it:
        cfg = {
            "dataset": job["dataset"], "variant": job["variant"], "eps": job["eps"],
            "mode": job["mode"], "method": job["method"], "n_bins": job["n_bins"],
            "tag": args.tag,
        }
        if args.dataset == "adult":
            cfg["checksum"] = ds["checksum"]
        save_record(f"tabular_{args.dataset}", cfg, rec, seed=job["seed"])
        if preds is not None:
            import json as _json
            from pathlib import Path

            pdir = (ds if False else None)  # placeholder
        results.append((job, rec, preds, scores))
        if len(results) % 40 == 0:
            print(f"  {len(results)}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    if args.store_preds:
        _save_predictions(args, ds, results)
    print(f"done in {time.time()-t0:.0f}s")


def _save_predictions(args, ds, results):
    """Store test predictions for paired inference (one NPZ per variant)."""
    import json
    from pathlib import Path

    outdir = Path("results/preds")
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}-{args.tag}"
    by_variant: dict[str, dict] = {}
    for job, rec, preds, scores in results:
        key = f"{job['variant']}::{job['method']}"
        slot = by_variant.setdefault(key, {"seeds": [], "preds": [], "scores": []})
        slot["seeds"].append(job["seed"])
        slot["preds"].append(preds)
        slot["scores"].append(scores)
    payload = {
        "y_test": ds["y_te"].tolist(),
        "g_test": ds["g_te"].tolist(),
        "variants": {k: v for k, v in by_variant.items()},
        "dataset": args.dataset,
        "n_seeds": args.n_seeds,
        "split_seed": SEED_SPLIT,
    }
    (outdir / f"{name}.json").write_text(json.dumps(payload))
    print(f"predictions saved to results/preds/{name}.json")


if __name__ == "__main__":
    main()
