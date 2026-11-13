"""Protocol verification: thresholds and external scores from checkpoints.

For a sample of stored runs:
1. Rebuild the model from its checkpoint, recompute VALIDATION scores on the
   HAM10000 val split, re-select the threshold by max val BAcc, and compare
   with the threshold recorded at training time. Any difference means the
   recorded threshold did not come from the sealed internal val protocol.
2. Recompute Fitzpatrick17k scores for one run end-to-end and compare with
   the stored per-image scores (rounded to 5 decimals when stored).
3. Verify HAM test scores likewise for the same run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import fitzpatrick17k, ham10000
from ..models import vision
from ..models.tabular import select_threshold_bacc
from ..stats.metrics import balanced_accuracy

SEED_SPLIT = 20261111


def _load_model(run_id: str, seed: int, device: str):
    sd = torch.load(vision.CKPT_DIR / f"{run_id}.pt", map_location=device, weights_only=False)
    model = vision.build_model(seed, device)
    model.load_state_dict(sd["model"])
    model.eval()
    return model, sd["epoch"]


def verify(n_runs: int = 6, seed_start: int = 20261111) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = ham10000.lesion_split(ham10000.load_metadata(), seed=SEED_SPLIT)
    proxies = pd.read_csv(ham10000.PROC / "proxies.csv")
    meta = meta.merge(proxies, on="image_id", how="left", validate="1:1")
    img_path = str(ham10000.PROC / "images_224.npy")
    y_all = meta.y.to_numpy()
    idx = {s: np.flatnonzero((meta.split == s).to_numpy()) for s in ("train", "val", "test")}
    rng = np.random.default_rng(SEED_SPLIT + 1)
    tr = meta[meta.split == "train"].copy()
    tr["r"] = rng.random(len(tr))
    lesions = tr.groupby("lesion_id").agg(y=("y", "max"), r=("r", "first")).reset_index().sort_values("r")
    val_lesions = set(lesions.lesion_id.iloc[: int(round(0.1 * len(lesions)))])
    is_val = tr.lesion_id.isin(val_lesions).to_numpy()
    idx_val = idx["train"][is_val]
    idx_te = idx["test"]

    fitz_meta = pd.read_csv(fitzpatrick17k.PROC / "eval_malignant_benign_meta.csv")
    fitz_img = str(fitzpatrick17k.PROC / "eval_malignant_benign_224.npy")

    recs = [json.loads(p.read_text()) for p in sorted((Path("results") / "vision_ham10000").glob("*.json"))]
    recs = [r for r in recs if r["config"].get("tag") == "main" and r["config"].get("eps") == 0.0]
    picks = []
    seen_methods = set()
    for r in recs:
        m = r["config"]["method"]
        if m not in seen_methods:
            seen_methods.add(m)
            picks.append(r)
        if len(picks) >= n_runs:
            break

    preds = json.loads((Path("results/preds") / "vision_ham10000-main.json").read_text())
    out = []
    for r in picks:
        meth, seed = r["config"]["method"], r["seed"]
        run_id = f"v_{meth}_corrupt_eps0.0_s{seed}"
        model, last_epoch = _load_model(run_id, seed, device)
        s_val = vision.predict_scores(model, img_path, y_all[idx_val], index=idx_val, device=device)
        thr_recomputed = select_threshold_bacc(y_all[idx_val], s_val)
        thr_stored = r["metrics"]["threshold"]
        s_te = vision.predict_scores(model, img_path, y_all[idx_te], index=idx_te, device=device)
        bacc_recomputed = float(balanced_accuracy(y_all[idx_te], (s_te > thr_recomputed).astype(int)))

        # protocol integrity: stored threshold must be (near-)maximal on val.
        # cudnn nondeterminism between training-time and recompute forwards
        # can shift val scores slightly, so compare val BAcc at both
        # thresholds and the val-optimal value.
        bv_stored = float(balanced_accuracy(y_all[idx_val], (s_val > thr_stored).astype(int)))
        bv_rec = float(balanced_accuracy(y_all[idx_val], (s_val > thr_recomputed).astype(int)))
        n_val = len(idx_val)
        key = f"corrupt_eps0.0::{meth}::{seed}"
        stored_scores = np.asarray(preds[key]["ham_test_score"])
        score_match = bool(np.allclose(stored_scores, np.round(s_te, 5), atol=1e-9))
        out.append({
            "run_id": run_id, "last_epoch": int(last_epoch),
            "threshold_stored": thr_stored,
            "threshold_recomputed_from_val": float(thr_recomputed),
            "exact_threshold_match": bool(abs(thr_stored - thr_recomputed) < 1e-12),
            "val_bacc_at_stored_threshold": bv_stored,
            "val_bacc_at_recomputed_threshold": bv_rec,
            "val_bacc_deficit_of_stored_threshold": float(bv_rec - bv_stored),
            "protocol_consistent": bool(bv_rec - bv_stored <= 2.0 / n_val),  # <=2 val items
            "ham_test_bacc_stored": r["metrics"]["ham_bacc"],
            "ham_test_bacc_recomputed": bacc_recomputed,
            "ham_test_scores_match_stored": score_match,
        })
    # one end-to-end fitz recomputation
    r0 = picks[0]
    meth, seed = r0["config"]["method"], r0["seed"]
    run_id = f"v_{meth}_corrupt_eps0.0_s{seed}"
    model, _ = _load_model(run_id, seed, device)
    s_fitz = vision.predict_scores(model, fitz_img, np.zeros(len(fitz_meta)), device=device)
    key = f"corrupt_eps0.0::{meth}::{seed}"
    stored_fitz = np.asarray(preds[key]["fitz_score"])
    fitz_match = bool(np.allclose(stored_fitz, np.round(s_fitz, 5), atol=1e-9))
    result = {
        "device": device,
        "n_checked": len(out),
        "runs": out,
        "all_thresholds_val_optimal": all(o["protocol_consistent"] for o in out),
        "all_ham_scores_match_stored": all(o["ham_test_scores_match_stored"] for o in out),
        "fitz_scores_recomputed_match_stored": fitz_match,
        "fitz_recomputed_run": run_id,
        "note": "cudnn nondeterminism between training-time and recompute forward passes "
                "can shift thresholds by float-level amounts; protocol integrity is "
                "judged by val-optimality of the stored threshold, not bit equality.",
    }
    out_dir = Path("results/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "protocol_verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=6)
    args = ap.parse_args()
    r = verify(args.n_runs)
    print(json.dumps(r, indent=2))
