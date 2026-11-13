"""Protocol freeze: 5 vs 10 training epochs, decided on INTERNAL validation only.

Pre-registered decision rule (recorded here and in CHECKLIST.md before the
primary run): compare mean HAM validation balanced accuracy across the shared
(method, seed) cells of the 5-epoch (tag main) and 10-epoch (tag sens10ep)
arms; adopt 10 epochs only if the mean paired improvement exceeds 0.005 (the
seed-noise band observed in the main grid). External metrics are never
consulted for this decision.

val_bacc for the 10-epoch arm comes from the sens10ep records; val_bacc for
the 5-epoch arm is recomputed from the restored main checkpoints at the
stored validation-selected threshold (main records predate val_bacc
logging).

Output: results/audit/protocol_freeze.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..models import vision
from ..stats.metrics import balanced_accuracy
from .run_vision import prepare_ham

CKPT = vision.CKPT_DIR
METHODS = ("erm", "class_strat", "proxy_strat", "proxy_strat_joint",
           "proxy_iw", "proxy_dro", "random_proxy_strat")
SEEDS = (0, 1, 2)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ham = prepare_ham()
    y_val = ham["y"][ham["idx_val"]]

    recs = {}
    for f in Path("results/vision_ham10000").glob("*.json"):
        r = json.loads(f.read_text())
        c = r["config"]
        recs[(c["tag"], c["eps"], c["method"], r["seed"])] = r

    out = {"device": device, "cells": {}, "rule": "adopt 10ep iff mean paired val-BAcc gain > 0.005"}
    deltas = []
    for meth in METHODS:
        for seed in SEEDS:
            rec5 = recs.get(("main", 0.0, meth, seed))
            rec10 = recs.get(("sens10ep", 0.0, meth, seed))
            if rec5 is None:
                continue
            thr = rec5["metrics"]["threshold"]
            run_id = f"v_{meth}_corrupt_eps0.0_s{seed}"
            ckpt5 = CKPT / f"{run_id}.pt"
            if not ckpt5.exists():
                out["cells"][f"{meth}::s{seed}"] = {"error": "missing 5ep ckpt"}
                continue
            sd = torch.load(ckpt5, map_location=device, weights_only=False)
            if sd.get("epochs_requested", 5) != 5:
                out["cells"][f"{meth}::s{seed}"] = {"error": "ckpt wrong protocol"}
                continue
            model = vision.build_model(seed, device)
            model.load_state_dict(sd["model"])
            model.eval()
            s_val = vision.predict_scores(model, ham["img_path"], y_val,
                                          index=ham["idx_val"], device=device)
            val5 = float(balanced_accuracy(y_val, (s_val > thr).astype(int)))
            cell = {"val_bacc_5ep": val5, "threshold": thr}
            if rec10 is not None:
                cell["val_bacc_10ep"] = rec10["metrics"]["val_bacc"]
                deltas.append(cell["val_bacc_10ep"] - val5)
            out["cells"][f"{meth}::s{seed}"] = cell
        print(meth, "done", flush=True)

    n = len(deltas)
    out["n_paired_cells"] = n
    out["mean_val_bacc_5ep"] = float(np.mean([c["val_bacc_5ep"] for c in out["cells"].values()
                                              if "val_bacc_5ep" in c]))
    out["mean_val_bacc_10ep"] = float(np.mean([c["val_bacc_10ep"] for c in out["cells"].values()
                                               if "val_bacc_10ep" in c]))
    out["mean_delta_10minus5"] = float(np.mean(deltas)) if n else None
    out["decision"] = ("10ep" if n and out["mean_delta_10minus5"] > 0.005 else "5ep")
    Path("results/audit/protocol_freeze.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2))
    for cell, c in out["cells"].items():
        if "val_bacc_10ep" in c:
            print(f"{cell:<26} 5ep={c['val_bacc_5ep']:.4f} 10ep={c['val_bacc_10ep']:.4f} "
                  f"delta={c['val_bacc_10ep'] - c['val_bacc_5ep']:+.4f}")


if __name__ == "__main__":
    main()
