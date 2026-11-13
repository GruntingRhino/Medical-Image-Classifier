"""Publication figures from actual experiment files.

Outputs: paper/figures/*.png (150 dpi). Matplotlib only, no seaborn.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
OUT = Path("paper/figures")
plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.titlesize": 9})


def _load(name):
    return json.loads((RESULTS / name).read_text())


def fig_forest(tag="main"):
    """Forest plot: paired ΔFitz BAcc vs ERM across methods and proxy variants."""
    d = _load(f"vision_summary/vision-{tag}.json")
    variants = sorted(d["paired_vs_erm"])
    fig, axes = plt.subplots(1, len(variants), figsize=(7.5, 3.2), sharey=True)
    for ax, variant in zip(np.atleast_1d(axes), variants):
        fam = d["paired_vs_erm"][variant]["methods"]
        holm = d["paired_vs_erm"][variant]["holm"].get("fitz_bacc", {})
        methods = sorted(fam)
        ys = np.arange(len(methods))
        for y, m in zip(ys, methods):
            c = fam[m]["delta_seed_summary"]["fitz_bacc"]
            lo, hi = c["lo"], c["hi"]
            p = holm.get(m)
            color = "crimson" if (p is not None and p < 0.05 and c["mean"] < 0) else (
                "seagreen" if (p is not None and p < 0.05 and c["mean"] > 0) else "0.3")
            ax.plot([lo, hi], [y, y], color=color, lw=1.4)
            ax.plot(c["mean"], y, "o", color=color, ms=4)
        ax.axvline(0, color="0.6", lw=0.8, ls="--")
        ax.set_yticks(ys, methods, fontsize=7)
        ax.set_title(variant.replace("corrupt_", ""))
        ax.set_xlabel("ΔFitz BAcc vs ERM")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig_forest_fitz_delta.png")
    plt.close(fig)


def fig_fst_levels(tag="main"):
    """BAcc by FST level, ERM vs proxy methods (mean over seeds)."""
    d = _load(f"vision_summary/vision-{tag}.json")
    pfl = d.get("per_fst_level", {})
    if not pfl:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    levels = sorted((l for l in next(iter(pfl.values())) if l.isdigit()), key=int)
    show = [m for m in ("erm", "proxy_strat", "proxy_strat_joint", "proxy_dro", "class_strat") if m in pfl]
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(show):
        vals = [pfl[m][l]["bacc_mean"] for l in levels]
        ax.plot(range(len(levels)), vals, "o-", ms=3, color=cmap(i / max(1, len(show) - 1)), label=m)
    ax.set_xticks(range(len(levels)), [f"FST{l}" for l in levels], fontsize=7)
    ax.set_ylabel("Balanced accuracy (external)")
    ax.set_ylim(0.3, 0.75)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bacc_by_fst.png")
    plt.close(fig)


def fig_proxy_distributions():
    """ITA distributions: HAM train vs Fitz eval (by FST), plus per-bin placement."""
    r = _load("audit/external_degradation.json")
    pd_ = r["proxy_distribution"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    ax = axes[0]
    ita = pd_["ita_by_fst_level"]
    for l, color in zip(range(1, 7), plt.get_cmap("coolwarm")(np.linspace(0.05, 0.95, 6))):
        d = ita[str(l)]
        ax.errorbar(d["mean"], 6 - l + 1, xerr=[[d["mean"] - d["q10"]], [d["q90"] - d["mean"]]],
                    fmt="o", ms=4, color=color, capsize=2)
    ax.set_yticks(range(1, 7), [f"FST{l}" for l in range(6, 0, -1)], fontsize=7)
    ax.set_xlabel("ITA proxy (center patch)")
    ham = pd_["ham_train_ita"]
    fitz = pd_["fitz_eval_ita"]
    ax.axvline(ham["mean"], color="k", lw=1, ls="--")
    ax.axvline(fitz["mean"], color="k", lw=1, ls=":")
    ax.set_title("Fitz ITA by FST level (q10–q90); -- HAM, : Fitz mean")
    ax = axes[1]
    bins = ["bin0\n(darkest)", "bin1", "bin2", "bin3", "bin4\n(brightest)"]
    w = 0.38
    fitz_share = [pd_["fitz_fraction_per_ham_bin"][f"bin{i}"] for i in range(5)]
    d56_share = [pd_["fitz_fst56_fraction_per_ham_bin"][f"bin{i}"] for i in range(5)]
    x = np.arange(5)
    ax.bar(x - w / 2, fitz_share, w, label="all Fitz", color="0.6")
    ax.bar(x + w / 2, d56_share, w, label="FST 5-6", color="crimson")
    ax.axhline(0.2, color="k", lw=0.8, ls="--")
    ax.text(4.45, 0.205, "HAM bin size\n(uniform)", fontsize=6, va="bottom", ha="right")
    ax.set_xticks(x, bins, fontsize=6)
    ax.set_ylabel("fraction of eval images")
    ax.legend(fontsize=6)
    ax.set_title("Placement of Fitz images in HAM ITA bins")
    fig.tight_layout()
    fig.savefig(OUT / "fig_proxy_shift.png")
    plt.close(fig)


def fig_operating_points():
    """Per-group predicted-positive rate vs true prevalence (eps 0.0)."""
    r = _load("audit/external_degradation.json")
    tt = r["threshold_transfer_and_group_decomposition"]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for m in sorted(tt):
        t = tt[m]
        ax.plot([0, 1], [t["pospred_light"], t["pospred_dark"]], "o-", ms=4,
                label=m, color=plt.get_cmap("tab10")(hash(m) % 10))
    ax.plot([0, 1], [r["prevalence"]["fitz_prevalence_by_fst"]["1"], r["prevalence"]["fitz_prevalence_by_fst"]["5"]],
            "s--", color="k", ms=5, label="true prevalence")
    ax.set_xticks([0, 1], ["light (FST1-2)", "dark (FST5-6)"])
    ax.set_ylabel("positive-prediction rate")
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_operating_points.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_forest()
    fig_fst_levels()
    fig_proxy_distributions()
    fig_operating_points()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
