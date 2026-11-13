"""EXPLORATORY (post hoc): construct-matched external evaluation.

The frozen protocol evaluates a HAM10000 melanoma-vs-rest model on the
Fitzpatrick17k malignant-vs-benign split. Of its 1,683 positives only 429
are melanoma; the rest are mostly basal/squamous cell carcinoma, actinic
keratosis, mycosis fungoides and Kaposi sarcoma, several of which HAM10000
labels as NEGATIVE (bcc, akiec). The primary external metric therefore
mixes domain shift with a label-definition mismatch.

This script re-scores the SAME stored per-image scores at the SAME
validation-selected thresholds on the construct-matched subset
(melanoma, incl. lentigo maligna, vs benign). Nothing is retrained or
re-thresholded. It was added after the primary results were seen and is
reported as exploratory only, outside every Holm family.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..stats.hierarchical import cell_weights, hierarchical_paired
from ..stats.metrics import auroc, balanced_accuracy
from .aggregate_vision import PREDS_DIR

META = Path("data/processed/fitzpatrick17k/eval_malignant_benign_meta.csv")
OUT = Path("results/hier_summary/fitz-melanoma-exploratory.json")
MELANOMA = "melanoma|lentigo maligna"


def main(n_boot: int = 9999, boot_seed: int = 20261114):
    meta = pd.read_csv(META)
    keep = (meta.label.str.contains(MELANOMA, case=False) | (meta.y == 0)).to_numpy()
    y = meta.y.to_numpy().astype(int)[keep]
    fst = meta.fst.to_numpy(dtype=float)[keep]
    g = np.where((fst >= 5) & (fst <= 6), 1, np.where((fst >= 1) & (fst <= 2), 0, np.nan))
    cells = np.where(np.isnan(g), -1, g).astype(int) * 2 + y
    masks = {"dark": g == 1, "light": g == 0}

    preds = json.loads((PREDS_DIR / "vision_ham10000-main.json").read_text())
    groups: dict = {}
    for key, p in preds.items():
        variant, method, seed = key.rsplit("::", 2)
        score = np.asarray(p["fitz_score"])[keep]
        groups.setdefault((variant, method), {})[int(seed)] = (score, (score > p["threshold"]).astype(float))

    absolute, paired = {}, {}
    for (variant, method), by_seed in sorted(groups.items()):
        rows = []
        for s in sorted(by_seed):
            score, pred = by_seed[s]
            rows.append({"bacc": balanced_accuracy(y, pred), "auroc": auroc(y, score),
                         "bacc_dark": balanced_accuracy(y[masks["dark"]], pred[masks["dark"]]),
                         "bacc_light": balanced_accuracy(y[masks["light"]], pred[masks["light"]]),
                         "auroc_dark": auroc(y[masks["dark"]], score[masks["dark"]]),
                         "pred_pos": float(pred.mean())})
        absolute[f"{variant}::{method}"] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                                                "sd": float(np.std([r[k] for r in rows], ddof=1))}
                                            for k in rows[0]}
    for i, (variant, method) in enumerate(sorted(k for k in groups if k[1] != "erm")):
        erm = groups.get((variant, "erm"))
        common = sorted(set(erm) & set(groups[(variant, method)]))
        a = np.vstack([groups[(variant, method)][s][1] for s in common])
        b = np.vstack([erm[s][1] for s in common])
        paired[f"{variant}::{method}"] = hierarchical_paired(
            y.astype(float), a, b, lambda r: cell_weights(cells, r), masks=masks,
            n_boot=n_boot, seed=boot_seed + i)
    out = {"status": "EXPLORATORY_POST_HOC", "subset": "melanoma (incl. lentigo maligna) vs benign",
           "n": int(keep.sum()), "n_melanoma": int(y.sum()), "prevalence": float(y.mean()),
           "counts": {k: {"n": int(m.sum()), "melanoma": int(y[m].sum())} for k, m in masks.items()},
           "absolute": absolute, "paired_vs_erm": paired}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    erm = absolute["corrupt_eps0.0::erm"]
    print(f"n={out['n']} melanoma={out['n_melanoma']} prev={out['prevalence']:.3f} counts={out['counts']}")
    print("ERM eps0.0:", {k: round(v["mean"], 3) for k, v in erm.items()})
    for k, r in paired.items():
        if k.startswith("corrupt_eps0.0"):
            print(k, {m: (round(r[m]["delta"], 4), [round(c, 4) for c in r[m]["ci_primary"]], round(r[m]["p_primary"], 3))
                      for m in ("overall", "dark")})


if __name__ == "__main__":
    main()
