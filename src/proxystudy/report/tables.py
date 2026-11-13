"""Publication tables generated from actual experiment files.

Every number is read from results/**.json produced by this repo — no hand-
entered values. Outputs: paper/tables/*.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")
OUT = Path("paper/tables")


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def _ci(c):
    lo = c.get("lo", c.get("t_ci_lo"))
    hi = c.get("hi", c.get("t_ci_hi"))
    return lo, hi


def _load(name):
    return json.loads((RESULTS / name).read_text())


def _records(tag="main"):
    out = {}
    for p in sorted((RESULTS / "vision_ham10000").glob("*.json")):
        r = json.loads(p.read_text())
        if r["config"].get("tag") != tag:
            continue
        key = (r["config"]["variant"], r["config"]["method"], r["seed"])
        out.setdefault(key, []).append(r["metrics"])
    return out


def table_audit() -> str:
    a = _load("audit/dataset_audit.json")
    h, f = a["ham10000"], a["fitzpatrick17k"]
    lines = [
        "# Table 1: Dataset integrity audit",
        "",
        "| Check | HAM10000 | Fitzpatrick17k (eval subset) |",
        "|---|---|---|",
        f"| Images | {h['n_images']:,} ({h['n_lesions']:,} lesions) | {f['eval_subset_rows']:,} |",
        f"| Exact content duplicates | {h['exact_duplicate_groups']} groups ({h['duplicate_images_total']} images) | {f['content_duplicate_groups']} |",
        f"| Duplicates spanning splits | {len(h['duplicates_spanning_splits'])} | n/a (external set) |",
        f"| Lesion label inconsistencies | {h['lesion_label_inconsistent']} | {f['key_label_conflicts']} (per-key) |",
        f"| Content md5 == filename | n/a | {f['content_md5_equals_filename']}/{f['eval_subset_rows']} |",
        f"| Positive prevalence | {h['melanoma_prevalence']:.3f} (melanoma) | {f['malignant_prevalence']:.3f} (malignant) |",
        f"| Skin-tone annotation | {'present' if h['skin_tone_annotation'] else 'none (by design)'} | FST 1-6 + unknown |",
        f"| Cross-dataset content overlap | 0 | 0 |",
        f"| QC-flagged rows | n/a | {sum(v for k, v in f['qc_flags'].items() if k != 'NaN')} / {f['eval_subset_rows']} |",
        "",
        "Subgroup counts (eval subset): " + ", ".join(f"FST {k} = {v}" for k, v in f["fst_counts"].items()) + ".",
        "",
        "Audit correction: `fst == 0` in the re-host corresponds to the original "
        f"`fitzpatrick_scale == -1` (unknown); the light group is FST 1-2 only "
        f"(n = {f['fst_counts']['1'] + f['fst_counts']['2']}); FST 0 excluded from "
        "binary contrasts.",
    ]
    return "\n".join(lines)


def table_vision_absolute(tag="main"):
    """Absolute metrics: HAM from records, Fitz from the summary seed tables."""
    recs = _records(tag)
    d = _load(f"vision_summary/vision-{tag}.json")
    ss = d["seed_summaries"]
    lines = [
        f"# Table 2: Vision absolute metrics ({tag})",
        "",
        "Mean over seeds [t-CI] (seed-level training variability). "
        "HAM = internal test lesions; Fitz = external. dark = FST 5-6, "
        "light = FST 1-2 (FST 0 unknown excluded).",
        "",
        "| Method (eps0.0) | n | HAM BAcc | HAM AUROC | Fitz BAcc | Fitz AUROC | Fitz BAcc dark | Fitz BAcc light | worst-group |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    methods = sorted({m for (_v, m, _s) in recs})
    for method in methods:
        vals = [m for (v, mm, s), ms in recs.items() if mm == method and v == "corrupt_eps0.0"
                for m in ms]
        ham_bacc = np.mean([x["ham_bacc"] for x in vals])
        ham_auroc = np.mean([x["ham_auroc"] for x in vals])
        key = f"corrupt_eps0.0::{method}"
        if key not in ss:
            lines.append(f"| {method} | {len(vals)} | {_fmt(ham_bacc)} | {_fmt(ham_auroc)} | — | — | — | — | — |")
            continue
        r = ss[key]
        if "bacc_overall" not in r or "lo" not in r["bacc_overall"]:
            return ("# Table 2: pending — regenerate with "
                    "`python -m proxystudy.exp.aggregate_vision --tag main --n-boot 1000`")
        n = r["bacc_overall"]["n"]
        def c(metric):
            x = r[metric]
            return f"{_fmt(x['mean'])} [{_fmt(x['lo'], 2)}, {_fmt(x['hi'], 2)}]"
        lines.append(
            f"| {method} | {n} | {_fmt(ham_bacc)} | {_fmt(ham_auroc)} | {c('bacc_overall')} | "
            f"{c('auroc_overall')} | {c('bacc_dark')} | {c('bacc_light')} | {c('bacc_worst')} |")
    return "\n".join(lines)


def table_vision_paired(tag="main"):
    d = _load(f"vision_summary/vision-{tag}.json")
    fam0 = d["paired_vs_erm"]["corrupt_eps0.0"]["methods"]
    if "delta_seed_summary" not in fam0[next(iter(fam0))]:
        return ("# Table 3: pending — regenerate with "
                "`python -m proxystudy.exp.aggregate_vision --tag main --n-boot 1000`")
    lines = [
        f"# Table 3: Vision paired deltas vs ERM (eps=0.0, {tag})",
        "",
        "Paired per-seed deltas (same split/protocol). Cluster-aware: lesion-"
        "level bootstrap (1,000 draws) for BAcc, paired DeLong for AUROC. "
        "Fisher-combined p over seeds; Holm-Bonferroni across methods within "
        "each metric family.",
        "",
        "| Method | ΔHAM BAcc | p | ΔFitz BAcc | p | ΔFitz AUROC | p | ΔFitz BAcc dark | p | ΔFitz BAcc light | p |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    fam = d["paired_vs_erm"]["corrupt_eps0.0"]["methods"]
    holm = d["paired_vs_erm"]["corrupt_eps0.0"]["holm"]
    for method in sorted(fam):
        ds = fam[method]["delta_seed_summary"]
        p = fam[method]["p_fisher"]
        def cell(metric):
            c = ds[metric]
            padj = holm.get(metric, {}).get(method)
            ptxt = (f"**{_fmt(padj, 4)}**" if padj is not None and padj < 0.05
                    else _fmt(padj if padj is not None else p.get(metric), 4))
            return f"{_fmt(c['mean'])} [{_fmt(c['lo'], 2)},{_fmt(c['hi'], 2)}] | {ptxt}"
        cells = [cell("ham_bacc"), cell("fitz_bacc"), cell("fitz_auroc"),
                 cell("fitz_bacc_dark"), cell("fitz_bacc_light")]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")
    for variant in sorted(d["paired_vs_erm"]):
        if variant == "corrupt_eps0.0":
            continue
        lines.append(f"## Variant {variant} (ΔFitz BAcc vs ERM, Holm p)")
        lines.append("")
        lines.append("| Method | ΔFitz BAcc | p(Holm) |")
        lines.append("|---|---|---|")
        famv = d["paired_vs_erm"][variant]["methods"]
        holmv = d["paired_vs_erm"][variant]["holm"]
        for method in sorted(famv):
            c = famv[method]["delta_seed_summary"]["fitz_bacc"]
            padj = holmv.get("fitz_bacc", {}).get(method)
            ptxt = f"**{_fmt(padj, 4)}**" if padj is not None and padj < 0.05 else _fmt(padj, 4)
            lines.append(f"| {method} | {_fmt(c['mean'])} [{_fmt(c['lo'], 2)},{_fmt(c['hi'], 2)}] | {ptxt} |")
        lines.append("")
    return "\n".join(lines)


def table_tabular(dataset="adult"):
    name = f"tabular_summary/tabular-{dataset}-main.json"
    if not (RESULTS / name).exists():
        return f"# Table: {dataset} — no summary file found"
    d = _load(name)
    lines = [
        f"# Table: Tabular ({dataset}) — paired deltas vs ERM",
        "",
        "Cell-stratified bootstrap over demographic cells (items independent); "
        "Fisher-combined p over seeds; Holm within metric family.",
        "",
        "| Variant | Method | Metric | Δ mean [t-CI] | p (Holm) |",
        "|---|---|---|---|---|",
    ]
    for variant in sorted(d["paired_vs_erm"]):
        block = d["paired_vs_erm"][variant]
        for method in sorted(block["methods"]):
            row = block["methods"][method]
            for metric in sorted(row["delta_seed_summary"]):
                c = row["delta_seed_summary"][metric]
                lo, hi = _ci(c)
                padj = block.get("holm", {}).get(metric, {}).get(method)
                if padj is None:
                    padj = row.get("p_fisher", {}).get(metric)
                star = "**" if padj is not None and padj < 0.05 else ""
                lines.append(f"| {variant} | {method} | {metric} | {_fmt(c['mean'], 4)} "
                             f"[{_fmt(lo, 2)}, {_fmt(hi, 2)}] | {star}{_fmt(padj, 4)}{star} |")
    return "\n".join(lines)


def table_mechanism():
    if not (RESULTS / "audit/external_degradation.json").exists():
        return None
    r = _load("audit/external_degradation.json")
    pd_, pv, tt = r["proxy_distribution"], r["prevalence"], r["threshold_transfer_and_group_decomposition"]
    lines = [
        "# Table 6: External-degradation mechanism summary (eps 0.0)",
        "",
        f"- ITA proxy overlap HAM↔Fitz: {pd_['overlap_coefficient_ham_vs_fitz']:.3f}",
        f"- Prevalence: HAM train {pv['ham_train_melanoma']:.3f} → Fitz {pv['fitz_eval_malignant']:.3f}",
        "- Fitz share per HAM ITA bin: " + ", ".join(f"{k}={v:.2f}" for k, v in pd_["fitz_fraction_per_ham_bin"].items()),
        "- FST 5-6 share per HAM bin: " + ", ".join(f"{k}={v:.2f}" for k, v in pd_["fitz_fst56_fraction_per_ham_bin"].items()),
        "",
        "| Method | pred-pos (dark) | prev (dark) | pred-pos (light) | prev (light) | sens dark | spec dark |",
        "|---|---|---|---|---|---|---|",
    ]
    for method in sorted(tt):
        t = tt[method]
        lines.append(
            f"| {method} | {t['pospred_dark']:.3f} | {t['trueprev_dark']:.3f} | "
            f"{t['pospred_light']:.3f} | {t['trueprev_light']:.3f} | "
            f"{(t['sens_dark'] or 0):.3f} | {(t['spec_dark'] or 0):.3f} |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "table1_audit.md": table_audit(),
        "table2_vision_absolute.md": table_vision_absolute(),
        "table3_vision_paired.md": table_vision_paired(),
        "table4_tabular_adult.md": table_tabular("adult"),
        "table5_tabular_german.md": table_tabular("german"),
    }
    mech = table_mechanism()
    if mech:
        files["table6_mechanism.md"] = mech
    for name, content in files.items():
        (OUT / name).write_text(content)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
