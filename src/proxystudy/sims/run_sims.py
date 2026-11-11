"""Run the controlled simulation grid.

Each cell: (reliability, p_group, n, n_bins, confound, model, prevalence-gap).
Every cell runs ``n_seeds`` independent replications; for each we:
1. generate train/val/test with a seed,
2. fit every method with matched hyperparameter sweep on validation,
3. evaluate on a large fixed test draw, compute subgroup metrics,
4. store per-seed metrics.

Outputs aggregate records used by ``analyze_sims.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from ..exp.ledger import save_record
from ..models.tabular import TabularModelConfig, fit_tabular, select_threshold_bacc, sweep_C, sweep_hgb
from ..sampling.methods import MethodConfig, fit_method
from ..stats.metrics import subgroup_metrics, to_dict
from .dgp import SimParams, generate


def run_cell(
    reliability: float,
    p_group: float,
    n: int,
    n_bins: int,
    confound: float,
    confound_feature: float,
    model: str,
    prev_gap: float,
    seed: int,
    tag: str = "main",
) -> dict:
    params = SimParams(
        n=n,
        n_test=100_000,
        p_group=p_group,
        prevalence=(0.20 - prev_gap / 2, 0.20 + prev_gap / 2),
        reliability=reliability,
        confound=confound,
        zfeat=confound_feature,
        z_label_mult=1.6 if confound > 0 else 1.0,
        n_bins=n_bins,
        model=model,
        seed=seed,
    )
    ds = generate(params)
    # split train into train/val for hyperparameter selection
    rng = np.random.default_rng(seed + 10_000)
    idx = rng.permutation(params.n)
    n_val = max(int(0.2 * params.n), 200)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr, y_tr, P_tr, G_tr = ds.X[tr_idx], ds.y[tr_idx], ds.P[tr_idx], ds.G[tr_idx]
    X_val, y_val, P_val = ds.X[val_idx], ds.y[val_idx], ds.P[val_idx]
    X_te, y_te, G_te = ds.meta["X_test"], ds.meta["y_test"], ds.meta["G_test"]

    methods = [
        "erm",
        "class_strat",
        "proxy_strat",
        "proxy_strat_joint",
        "proxy_iw",
        "proxy_dro",
        "random_proxy_strat",
        "oracle_strat",
    ]
    results = {}
    for meth in methods:
        mcfg = MethodConfig(
            name=meth,
            n_bins=n_bins,
            seed=seed,
            joint_with_label=(meth == "proxy_strat_joint"),
        )
        out = fit_method(mcfg, y_tr, P_tr, G_train=G_tr)
        w = out.weights
        bins = out.bins if meth in {"proxy_dro"} else None
        dro_steps = 3 if meth == "proxy_dro" else 0
        if model == "logistic":
            best_C, _ = sweep_C(X_tr, y_tr, w, X_val, y_val, seed)
            tc = TabularModelConfig(model="logistic", C=best_C, seed=seed, dro_steps=dro_steps)
            m = fit_tabular(tc, X_tr, y_tr, w, bins=bins)
        else:
            (leaf, it), _ = sweep_hgb(X_tr, y_tr, w, X_val, y_val, seed)
            tc = TabularModelConfig(model="hgb", max_leaf_nodes=leaf, max_iter=it, seed=seed, dro_steps=dro_steps)
            m = fit_tabular(tc, X_tr, y_tr, w, bins=bins)
        val_scores = m.predict_proba(X_val)[:, 1]
        threshold = select_threshold_bacc(y_val, val_scores)
        score = m.predict_proba(X_te)[:, 1]
        pred = (score > threshold).astype(int)
        sm = subgroup_metrics(y_te, pred, G_te, score)
        results[meth] = {
            "overall_bacc": sm_to_bacc(y_te, pred),
            "threshold": threshold,
            **to_dict(sm),
        }
    # proxy quality on train (vs true G, which simulation exposes for analysis)
    from ..proxies.quality import proxy_quality_summary

    pq = proxy_quality_summary(P_tr, G_tr, n_bins=n_bins)
    return {
        "params": asdict(params),
        "metrics": results,
        "proxy_quality": {k: v for k, v in pq.items() if k != "calibration"},
    }


def sm_to_bacc(y_true, y_pred) -> float:
    from ..stats.metrics import balanced_accuracy

    return balanced_accuracy(y_true, y_pred)


def _worker(job):
    t0 = time.time()
    res = run_cell(**job)
    res["seconds"] = time.time() - t0
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke tests")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)

    if args.quick:
        grid = list(
            itertools.product(
                [0.0, 0.5, 1.0],  # reliability
                [0.25],           # p_group
                [1000],           # n
                [5],              # n_bins
                [0.0],            # confound
                [0.0],            # confound_feature
                ["logistic"],
                [0.1],            # prev_gap
            )
        )
        seeds = list(range(5))
    else:
        grid = list(
            itertools.product(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                [0.5, 0.25, 0.1],
                [500, 2000, 8000],
                [2, 5, 10],
                [0.0, 0.7],
                [0.0, 1.0],
                ["logistic"],
                [0.1],
            )
        )
        seeds = list(range(args.n_seeds))

    jobs = [
        dict(
            reliability=rel,
            p_group=pg,
            n=n,
            n_bins=nb,
            confound=cf,
            confound_feature=cfeat,
            model=mdl,
            prev_gap=pgap,
            seed=seed,
            tag=args.tag,
        )
        for rel, pg, n, nb, cf, cfeat, mdl, pgap in grid
        for seed in seeds
    ]
    # NOTE: confounded cells (cf>0) keep cfeat=0 in this grid; the harmful-
    # confounding variant runs with cfeat>0 separately in analyze phase.
    jobs = [j for j in jobs if not (j["confound"] > 0 and j["confound_feature"] > 0)]

    print(f"total jobs: {len(jobs)}")
    t0 = time.time()
    done = 0
    results_path = None
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for res in ex.map(_worker, jobs, chunksize=4):
            done += 1
            cfg = res["params"]
            seed = cfg.pop("seed")
            metrics = res["metrics"]
            proxy_q = res["proxy_quality"]
            rec_cfg = {k: v for k, v in cfg.items()}
            rec_cfg["tag"] = args.tag
            rec_cfg["proxy_quality"] = proxy_q
            rec_cfg["seconds"] = res["seconds"]
            save_record(f"sim_{args.tag}", rec_cfg, metrics, seed=seed)
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} cells done ({time.time()-t0:.0f}s)", flush=True)
    print(f"all done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
