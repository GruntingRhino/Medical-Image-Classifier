"""Tables and figures for the hierarchical re-analysis and the seed-inference simulation.

Reads results/hier_summary/hier-main.json, results/vision_summary/vision-main.json
(for the superseded Fisher/Holm p-values) and results/sim_seed/sim_seed.json.
Writes paper/tables/table7-9 and paper/figures/fig_forest_hier.png,
fig_sim_calibration.png. No number is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TABLES = Path("paper/tables")
FIGS = Path("paper/figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
METHOD_ORDER = ["class_strat", "proxy_dro", "proxy_iw", "proxy_strat", "proxy_strat_joint",
                "random_proxy_strat", "oracle_strat"]
PROCS = [("fisher", "Fisher over seeds"), ("pooled", "Pooled seed×item"),
         ("seed_t", "t-test on seed deltas"), ("hier_pct", "Hierarchical, percentile"),
         ("hier_t", "Hierarchical, t(S−1)")]


def _j(path):
    return json.loads(Path(path).read_text())


def _cell(r):
    lo, hi = r["ci_primary"]
    return f"{r['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _p(x):
    return "<0.001" if x < 1e-3 else f"{x:.3f}"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def table_vision(h, old):
    lines = ["# Table 7. Vision: paired deltas vs ERM, hierarchical inference (lesion/image × seed)",
             "",
             "Δ = method − ERM, mean over 10 paired seeds. [95% CI], t(9) on inflated bootstrap SE.",
             "p(Holm) is Holm across methods within each variant × metric family.",
             "p(old) is the superseded Fisher-over-seeds + Holm value from vision-main.json.",
             ""]
    for variant in sorted(h["fitz"]):
        lines += [f"## {variant}", "",
                  "| Method | Δ HAM BAcc | p(Holm) | p(old) | Δ Fitz BAcc | p(Holm) | p(old) | Δ FST 5–6 BAcc | p(Holm) | p(old) | Δ worst group |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        oh = old["paired_vs_erm"].get(variant, {}).get("holm", {})
        for m in [x for x in METHOD_ORDER if x in h["fitz"][variant]]:
            ham, fz = h["ham"][variant][m], h["fitz"][variant][m]

            def o(metric, m=m):
                v = oh.get(metric, {}).get(m)
                return _p(v) if v is not None else "n/a"

            lines.append(f"| {m} | {_cell(ham['overall'])} | {_p(ham['overall']['p_holm_family'])} | {o('ham_bacc')} "
                         f"| {_cell(fz['overall'])} | {_p(fz['overall']['p_holm_family'])} | {o('fitz_bacc')} "
                         f"| {_cell(fz['dark'])} | {_p(fz['dark']['p_holm_family'])} | {o('fitz_bacc_dark')} "
                         f"| {_cell(fz['worst_group'])} |")
        lines.append("")
    return "\n".join(lines)


def table_tabular(h):
    lines = ["# Table 8. Tabular: paired deltas vs ERM, hierarchical inference", "",
             "Models are deterministic given weights (identical per-seed deltas), so the",
             "percentile bootstrap over rows stratified by (group, label) is primary.", ""]
    for ds in ("adult", "german"):
        lines += [f"## {ds}", "", "| Variant | Method | Δ BAcc | p(Holm) | Δ gap (SBAG) | p(Holm) | Δ worst group | primary |",
                  "|---|---|---|---|---|---|---|---|"]
        for variant, by_m in sorted(h[ds]["test"].items()):
            for m in [x for x in METHOD_ORDER if x in by_m]:
                r = by_m[m]
                lines.append(f"| {variant} | {m} | {_cell(r['overall'])} | {_p(r['overall']['p_holm_family'])} "
                             f"| {_cell(r['gap'])} | {_p(r['gap']['p_holm_family'])} | {_cell(r['worst_group'])} "
                             f"| {r['gap']['primary']} |")
        lines.append("")
    rows = h["primary_global_holm"]
    sig = [r for r in rows if r["p_holm_global"] < 0.05]
    lines += [f"Global Holm over all {len(rows)} primary-endpoint tests: {len(sig)} significant at 0.05.", ""]
    for r in sig:
        lines.append(f"- {r['dataset']}/{r['variant']}/{r['method']} {r['metric']}: Δ {r['delta']:+.3f}, "
                     f"p(global Holm) {_p(r['p_holm_global'])}")
    return "\n".join(lines)


def table_sim(sim):
    cells = sim["cells"]
    lines = ["# Table 9. Seed-combination procedures: rejection rate at α = 0.05", "",
             f"{cells[0]['reps']} replications per cell, {cells[0]['n_boot']} bootstrap draws, n = {cells[0]['n']} test items.",
             "Effect 0 = true null (type-I error; nominal 0.05). Monte Carlo SE ≈ 0.007 at 0.05.", "",
             "| Seeds | Seed noise τ | Effect | " + " | ".join(lbl for _, lbl in PROCS) + " |",
             "|---|---|---|" + "---|" * len(PROCS)]
    for c in sorted(cells, key=lambda c: (c["effect"], c["n_seeds"], c["tau"])):
        r = c["rejection_rate"]
        lines.append(f"| {c['n_seeds']} | {c['tau']:.1f} | {c['effect']:.2f} | " +
                     " | ".join(f"{r[k]:.3f}" for k, _ in PROCS) + " |")
    return "\n".join(lines)


def fig_forest(h, variant="corrupt_eps0.0"):
    methods = [m for m in METHOD_ORDER if m in h["fitz"][variant]]
    panels = [("HAM10000 internal BAcc", h["ham"][variant], "overall"),
              ("Fitzpatrick17k external BAcc", h["fitz"][variant], "overall"),
              ("Fitzpatrick17k FST 5–6 BAcc", h["fitz"][variant], "dark")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (title, block, metric) in zip(axes, panels):
        for yi, m in zip(y, methods):
            r = block[m][metric]
            lo, hi = r["ci_primary"]
            ax.plot([lo, hi], [yi, yi], color=SERIES[0], lw=2, solid_capstyle="round")
            ax.plot(r["delta"], yi, "o", ms=8, color=SERIES[0], mec="#fcfcfb", mew=2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Δ vs ERM (95% CI)", fontsize=8, color=INK2)
        _style(ax)
    axes[0].set_yticks(y, methods)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_forest_hier.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def fig_sim(sim):
    null = [c for c in sim["cells"] if c["effect"] == 0.0]
    seeds = sorted({c["n_seeds"] for c in null})
    fig, axes = plt.subplots(1, len(seeds), figsize=(11, 3.2), sharey=True)
    for ax, S in zip(axes, seeds):
        cs = sorted([c for c in null if c["n_seeds"] == S], key=lambda c: c["tau"])
        taus = [c["tau"] for c in cs]
        for (k, lbl), col, mk in zip(PROCS, SERIES, MARKERS):
            ax.plot(taus, [c["rejection_rate"][k] for c in cs], color=col, lw=2, marker=mk, ms=6,
                    mec="#fcfcfb", mew=1.5, label=lbl)
        ax.axhline(0.05, color=INK2, lw=1, ls="--")
        ax.set_title(f"{S} seeds", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("seed noise τ", fontsize=8, color=INK2)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_ylim(0, 1.05)
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("type-I error (nominal 0.05, dashed)", fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGS / "fig_sim_calibration.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    h = _j("results/hier_summary/hier-main.json")
    old = _j("results/vision_summary/vision-main.json")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table7_vision_hier.md").write_text(table_vision(h["vision"], old))
    (TABLES / "table8_tabular_hier.md").write_text(table_tabular(h))
    fig_forest(h["vision"])
    sim_path = Path("results/sim_seed/sim_seed.json")
    if sim_path.exists():
        sim = _j(sim_path)
        (TABLES / "table9_sim_calibration.md").write_text(table_sim(sim))
        fig_sim(sim)
    print("wrote tables 7-9 and figures")


if __name__ == "__main__":
    main()
