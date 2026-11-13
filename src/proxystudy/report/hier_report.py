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
    paper_numbers()
    print("wrote tables 7-9, figures and paper/numbers.json")



def absolute_table(variant="corrupt_eps0.0"):
    """Per-method absolute metrics (mean and SD over seeds) from stored predictions."""
    from ..exp.aggregate_vision import load_context
    from ..stats.metrics import auroc, balanced_accuracy

    y_ham, _, y_fitz, g_fitz, _ = load_context()
    preds = _j("results/preds/vision_ham10000-main.json")
    acc: dict = {}
    for key, p in preds.items():
        v, method, _seed = key.rsplit("::", 2)
        if v != variant:
            continue
        sh, sf, thr = np.asarray(p["ham_test_score"]), np.asarray(p["fitz_score"]), p["threshold"]
        pf = (sf > thr).astype(int)
        row = {"ham_bacc": balanced_accuracy(y_ham, (sh > thr).astype(int)), "ham_auroc": auroc(y_ham, sh),
               "fitz_bacc": balanced_accuracy(y_fitz, pf), "fitz_auroc": auroc(y_fitz, sf)}
        for name, m in (("dark", g_fitz == 1), ("light", g_fitz == 0)):
            row[f"fitz_bacc_{name}"] = balanced_accuracy(y_fitz[m], pf[m])
        acc.setdefault(method, []).append(row)
    return {m: {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1)),
                    "n": len(rows)} for k in rows[0]} for m, rows in sorted(acc.items())}


def paper_numbers():
    """Derived numbers quoted in the manuscript prose, written to paper/numbers.json."""
    from scipy import stats as st

    h = _j("results/hier_summary/hier-main.json")
    mel = _j("results/hier_summary/fitz-melanoma-exploratory.json")
    nums = {"mde_80": {}}
    for split, metric in (("ham", "overall"), ("fitz", "overall"), ("fitz", "dark")):
        halves = []
        for by_m in h["vision"][split].values():
            for r in by_m.values():
                x = r[metric]
                df = r["_design"]["n_seeds"] - 1
                if x["primary"] == "t":
                    se = (x["ci_primary"][1] - x["ci_primary"][0]) / 2 / st.t.ppf(0.975, df)
                    halves.append((st.t.ppf(0.975, df) + st.t.ppf(0.8, df)) * se)
        nums["mde_80"][f"{split}_{metric}"] = {"median": float(np.median(halves)),
                                                "min": float(np.min(halves)), "max": float(np.max(halves))}
    rows = h["primary_global_holm"]
    nums["global"] = {"n_tests": len(rows), "n_sig": sum(r["p_holm_global"] < 0.05 for r in rows)}
    nums["melanoma_subset"] = {k: mel[k] for k in ("n", "n_melanoma", "prevalence", "counts")}
    nums["melanoma_subset"]["erm_eps0"] = {k: v["mean"] for k, v in mel["absolute"]["corrupt_eps0.0::erm"].items()}
    old = _j("results/vision_summary/vision-main.json")["paired_vs_erm"]
    neg = {}
    for v, tab in old.items():
        for arm in ("random_proxy_strat", "proxy_strat"):
            neg[f"{v}::{arm}"] = {"old_holm_fitz": tab["holm"]["fitz_bacc"][arm],
                                  "new_holm_fitz": h["vision"]["fitz"][v][arm]["overall"]["p_holm_family"]}
    nums["negative_controls"] = neg
    for name in ("sim_seed", "sim_seed_het"):
        p = Path(f"results/sim_seed/{name}.json")
        if p.exists():
            cells = _j(p)["cells"]
            nums[name] = {"max_null_rate": {k: float(max(c["rejection_rate"][k] for c in cells
                                                         if c["effect"] == 0.0 or name == "sim_seed_het"))
                                            for k, _ in PROCS}}
    nums["absolute_eps0"] = absolute_table()
    (Path("paper") / "numbers.json").write_text(json.dumps(nums, indent=1))
    typst_tables(nums)
    return nums


def _f(x, d=3):
    return f"{x:.{d}f}"


def _d(r):
    lo, hi = r["ci_primary"]
    return f"{r['delta']:+.3f}\n[{lo:+.3f}, {hi:+.3f}]"


def typst_tables(nums):
    """Pre-formatted table rows for paper/paper.typ (generated, never typed)."""
    h = _j("results/hier_summary/hier-main.json")
    old = _j("results/vision_summary/vision-main.json")["paired_vs_erm"]
    mel = _j("results/hier_summary/fitz-melanoma-exploratory.json")
    t = {}
    order = ["erm"] + [m for m in METHOD_ORDER if m in nums["absolute_eps0"]]
    t["absolute"] = [[m] + [f"{_f(r[k]['mean'])} ({_f(r[k]['sd'])})" for k in
                            ("ham_bacc", "ham_auroc", "fitz_bacc", "fitz_auroc", "fitz_bacc_dark", "fitz_bacc_light")]
                     for m in order for r in [nums["absolute_eps0"][m]]]
    v = "corrupt_eps0.0"
    t["paired"] = []
    for m in [x for x in METHOD_ORDER if x in h["vision"]["fitz"][v]]:
        ham, fz = h["vision"]["ham"][v][m]["overall"], h["vision"]["fitz"][v][m]
        t["paired"].append([m, _d(ham), _p(ham["p_holm_family"]), _d(fz["overall"]),
                            _p(fz["overall"]["p_holm_family"]), _p(old[v]["holm"]["fitz_bacc"][m]),
                            _d(fz["dark"]), _p(fz["dark"]["p_holm_family"])])
    t["paired_eps"] = []
    for v2 in ("corrupt_eps0.3", "corrupt_eps0.6"):
        for m in [x for x in METHOD_ORDER if x in h["vision"]["fitz"][v2]]:
            fz = h["vision"]["fitz"][v2][m]
            t["paired_eps"].append([v2.replace("corrupt_", ""), m, _d(fz["overall"]),
                                    _p(fz["overall"]["p_holm_family"]), _d(fz["dark"]), _p(fz["dark"]["p_holm_family"])])
    t["tabular_sig"] = [[r["dataset"], r["variant"], r["method"], f"{r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]",
                         _p(r["p"]), _p(r["p_holm_global"])]
                        for r in h["primary_global_holm"] if r["p_holm_global"] < 0.05]
    t["melanoma"] = [[k.split("::")[1], _d(r["overall"]), _p(r["overall"]["p_primary"]), _d(r["dark"]),
                      _p(r["dark"]["p_primary"])] for k, r in mel["paired_vs_erm"].items() if k.startswith("corrupt_eps0.0")]
    for name in ("sim_seed", "sim_seed_het"):
        path = Path(f"results/sim_seed/{name}.json")
        if not path.exists():
            continue
        cells = _j(path)["cells"]
        if name == "sim_seed":
            t["sim_null"] = [[str(c["n_seeds"]), f"{c['tau']:.1f}"] + [_f(c["rejection_rate"][k]) for k, _ in PROCS]
                             for c in sorted(cells, key=lambda c: (c["n_seeds"], c["tau"])) if c["effect"] == 0.0]
            t["sim_power"] = [[str(c["n_seeds"]), f"{c['tau']:.1f}", f"{c['effect']:.2f}"] +
                              [_f(c["rejection_rate"][k]) for k, _ in PROCS]
                              for c in sorted(cells, key=lambda c: (c["n_seeds"], c["tau"], c["effect"]))
                              if c["effect"] > 0 and c["n_seeds"] == 10]
        else:
            t["sim_het"] = [[str(c["n_seeds"]), f"{c['tau']:.1f}", f"{c['effect']:.1f}"] +
                            [_f(c["rejection_rate"][k]) for k, _ in PROCS]
                            for c in sorted(cells, key=lambda c: (c["n_seeds"], c["tau"], c["effect"]))]
    (Path("paper") / "typ_tables.json").write_text(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
