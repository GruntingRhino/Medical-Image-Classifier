"""Vision experiments: HAM10000 training + Fitzpatrick17k external evaluation.

Protocol (identical for every method):
- HAM10000 binary melanoma target (y = dx == 'mel'); lesion-level split with
  sealed test set (seed 20261111, checksum recorded). Proxy P = center-patch
  ITA (degrees) from the luminance-proxy pipeline; the true demographic group
  is never available at training time.
- Proxy variants (data-level): raw ITA ("main"), and additive-noise corruption
  eps in {0.3, 0.6} (std scaled to the train split, clipped to train range,
  deterministic per-eps seed). The shuffled-proxy control is a METHOD
  (random_proxy_strat), mirroring the tabular experiments.
- Methods: erm, class_strat, proxy_strat, proxy_strat_joint, proxy_iw,
  proxy_dro, random_proxy_strat. oracle_strat is excluded: no true group
  exists in HAM10000 at any stage.
- Training: ImageNet-initialized ResNet-18, 5 epochs, AdamW, weighted sampler
  per method; identical budget for every method. Threshold selected on
  validation BAcc (identical protocol for every method).
- Evaluation:
  * HAM10000 test (sealed): BAcc, AUROC.
  * Fitzpatrick17k external (malignant vs benign): overall BAcc/AUROC;
    groups light (FST 1-2) vs dark (FST 5-6) -> per-group BAcc, worst-group
    BAcc, SBAG; FST 3-4 and unknown reported separately and excluded from
    worst-group statistics.
- Inference: paired bootstrap + DeLong vs ERM from stored per-run predictions
  (same seed), Holm correction across the method family (downstream script).

Oracle note: Fitzpatrick17k FST is used ONLY for evaluation; it never touches
training, method fitting, or threshold selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import fitzpatrick17k, ham10000
from ..exp.ledger import save_record
from ..models import vision
from ..models.tabular import select_threshold_bacc
from ..sampling.methods import MethodConfig, fit_method
from ..stats.metrics import auroc, balanced_accuracy, subgroup_metrics, to_dict

SEED_SPLIT = 20261111

METHODS = [
    "erm",
    "class_strat",
    "proxy_strat",
    "proxy_strat_joint",
    "proxy_iw",
    "proxy_dro",
    "random_proxy_strat",
]

PROXY_COL = "ita_center"


def prepare_ham(val_frac: float = 0.1) -> dict:
    """Lesion-level split + proxy join + cache paths. Val is a lesion-level
    carve-out inside the train split (never overlapping test)."""
    meta = ham10000.load_metadata()
    meta = ham10000.lesion_split(meta, seed=SEED_SPLIT)
    proxies = pd.read_csv(ham10000.PROC / "proxies.csv")
    meta = meta.merge(proxies, on="image_id", how="left", validate="1:1")
    assert meta[PROXY_COL].notna().all(), "missing proxies after join"

    img_path = str(ham10000.PROC / "images_224.npy")
    y = meta.y.to_numpy()
    idx = {s: np.flatnonzero((meta.split == s).to_numpy()) for s in ("train", "val", "test")}

    # lesion-level val carve-out: reassign a val_frac of TRAIN lesions to val
    rng = np.random.default_rng(SEED_SPLIT + 1)
    tr = meta[meta.split == "train"].copy()
    tr["r"] = rng.random(len(tr))
    lesions = tr.groupby("lesion_id").agg(y=("y", "max"), r=("r", "first")).reset_index()
    lesions = lesions.sort_values("r")
    n_val = int(round(val_frac * len(lesions)))
    val_lesions = set(lesions.lesion_id.iloc[:n_val])
    is_val = tr.lesion_id.isin(val_lesions).to_numpy()
    tr_idx = idx["train"]
    val_pos, train_pos = tr_idx[is_val], tr_idx[~is_val]

    return dict(
        meta=meta, y=y, img_path=img_path,
        idx_tr=train_pos, idx_val=val_pos, idx_te=idx["test"],
        P=meta[PROXY_COL].to_numpy(),
        checksum_test=ham10000.checksum_of_test_split(meta),
    )


def corrupt_proxy(P_tr: np.ndarray, eps: float, seed: int) -> np.ndarray:
    """Additive-noise corruption of the luminance proxy, train-range clipped."""
    if eps == 0:
        return P_tr.copy()
    rng = np.random.default_rng(int(round(eps * 1000)) + 7)
    noise = rng.normal(0.0, eps * float(np.std(P_tr)), size=len(P_tr))
    return np.clip(P_tr + noise, P_tr.min(), P_tr.max())


def prepare_fitz() -> dict:
    cache = fitzpatrick17k.build_eval_cache(224)
    meta = pd.read_csv(fitzpatrick17k.PROC / "eval_malignant_benign_meta.csv")
    y = meta.y.to_numpy().astype(int)
    fst = meta.fst.to_numpy(dtype=float)
    # AUDIT: fst==0 in the re-host = UNKNOWN (original -1), never light.
    g = np.where((fst >= 5) & (fst <= 6), 1, np.where((fst >= 1) & (fst <= 2), 0, np.nan))
    mid = np.isnan(g)  # FST 3-4 and unknown (fst==0): reported, excluded from worst-group
    return dict(img_path=str(cache), y=y, g=g, fst=fst, mid=mid, unknown=mid,
                checksum=fitzpatrick17k.checksum_eval())


def eval_fitz(model, fitz: dict, thr: float, device) -> dict:
    s = vision.predict_scores(model, fitz["img_path"], np.zeros(len(fitz["y"])), device=device)
    pred = (s > thr).astype(int)
    out = {
        "fitz_bacc": float(balanced_accuracy(fitz["y"], pred)),
        "fitz_auroc": float(auroc(fitz["y"], s)),
    }
    mask = ~np.isnan(fitz["g"])
    sm = subgroup_metrics(fitz["y"][mask], pred[mask], fitz["g"][mask].astype(int), s[mask])
    out["fitz_groups"] = to_dict(sm)
    if fitz["mid"].any():
        out["fitz_bacc_fst34_unknown"] = float(balanced_accuracy(fitz["y"][fitz["mid"]], pred[fitz["mid"]]))
    return {"metrics": out, "scores": s}


def run_single(job: dict, device: str) -> tuple[dict, dict]:
    ham, fitz, meth, seed, eps, epochs, batch_size, lr, n_bins = (
        job["ham"], job["fitz"], job["method"], job["seed"], job["eps"],
        job["epochs"], job["batch_size"], job["lr"], job["n_bins"],
    )
    P_tr = corrupt_proxy(ham["P"][ham["idx_tr"]], eps, seed)
    mcfg = MethodConfig(name=meth, n_bins=n_bins, seed=seed,
                        joint_with_label=(meth == "proxy_strat_joint"))
    out = fit_method(mcfg, ham["y"][ham["idx_tr"]], P_tr, G_train=None)
    w = out.weights
    bins = out.bins if meth == "proxy_dro" else None

    run_id = f"v_{meth}_{job['variant']}_s{seed}"
    res = vision.train_one(
        ham["img_path"], ham["y"][ham["idx_tr"]], ham["idx_tr"],
        ham["y"][ham["idx_val"]], ham["idx_val"],
        w, seed=seed, epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, run_id=run_id, dro_bins=bins, dro_eta=0.5, tag=job.get("tag", "main"),
    )
    model = res["model"]
    thr = select_threshold_bacc(ham["y"][ham["idx_val"]], res["scores"])
    val_bacc = float(balanced_accuracy(ham["y"][ham["idx_val"]],
                                       (res["scores"] > thr).astype(int)))

    s_ham = vision.predict_scores(model, ham["img_path"],
                                  ham["y"][ham["idx_te"]], index=ham["idx_te"], device=device)
    pred_ham = (s_ham > thr).astype(int)
    rec = {
        "method": meth, "seed": seed, "variant": job["variant"], "eps": eps,
        "n_bins": n_bins, "threshold": float(thr), "epochs": epochs,
        "val_bacc": val_bacc,
        "ham_test_checksum": ham["checksum_test"],
        "ham_bacc": float(balanced_accuracy(ham["y"][ham["idx_te"]], pred_ham)),
        "ham_auroc": float(auroc(ham["y"][ham["idx_te"]], s_ham)),
        "proxy_quality_note": "no true group in HAM10000; proxy quality vs FST validated on Fitzpatrick17k",
    }
    fe = eval_fitz(model, fitz, thr, device)
    rec.update(fe["metrics"])
    rec["fitz_checksum"] = fitz["checksum"]
    preds = {"ham_test_score": np.round(s_ham, 5).tolist(),
             "fitz_score": np.round(fe["scores"], 5).tolist(),
             "fitz_y": fitz["y"].tolist(), "fitz_g": np.nan_to_num(fitz["g"], nan=-1).astype(int).tolist(),
             "threshold": float(thr)}
    return rec, preds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="explicit seed list (overrides --n-seeds)")
    ap.add_argument("--eps", type=float, nargs="*", default=[0.0, 0.3, 0.6])
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: cap total runs")
    ap.add_argument("--store-preds", action="store_true", default=True)
    args = ap.parse_args(argv)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ham = prepare_ham()
    fitz = prepare_fitz()
    print(f"HAM train/val/test: {len(ham['idx_tr'])}/{len(ham['idx_val'])}/{len(ham['idx_te'])}"
          f" (test checksum {ham['checksum_test']})")
    print(f"Fitz eval n={len(fitz['y'])}, dark={int(np.nansum(fitz['g'] == 1))},"
          f" light={int(np.nansum(fitz['g'] == 0))}, checksum {fitz['checksum']}")

    variants = [(f"corrupt_eps{e}", e) for e in args.eps]
    seeds = list(range(args.n_seeds)) if args.seeds is None else list(args.seeds)
    jobs = [{"ham": ham, "fitz": fitz, "variant": vname, "eps": eps,
             "method": m, "seed": s, "epochs": args.epochs,
             "batch_size": args.batch_size, "lr": args.lr, "n_bins": args.n_bins,
             "tag": args.tag}
            for vname, eps in variants for m in args.methods for s in seeds]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} vision runs on device={device}")

    all_preds = {}
    preds_path = Path("results/preds") / f"vision_ham10000-{args.tag}.json"
    if args.store_preds and preds_path.exists():
        # merge into existing preds (multiple invocations across seeds)
        all_preds = json.loads(preds_path.read_text())
    t0 = time.time()
    for i, job in enumerate(jobs, 1):
        rec, preds = run_single(job, device)
        cfg = {"variant": job["variant"], "eps": job["eps"], "method": job["method"],
               "n_bins": job["n_bins"], "epochs": job["epochs"],
               "batch_size": job["batch_size"], "lr": job["lr"], "tag": args.tag}
        save_record("vision_ham10000", cfg, rec, seed=job["seed"])
        all_preds[f"{job['variant']}::{job['method']}::{job['seed']}"] = preds
        if args.store_preds:
            preds_path.parent.mkdir(parents=True, exist_ok=True)
            preds_path.write_text(json.dumps(all_preds))
        print(f"  [{i}/{len(jobs)}] {job['variant']}::{job['method']}::s{job['seed']} "
              f"ham_bacc={rec['ham_bacc']:.3f} fitz_worst={rec['fitz_groups']['worst_group_bacc']} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
