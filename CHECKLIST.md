# CHECKLIST — persistent progress (rebuild study)

Historical claims C1–C3 (AUDIT.md) are restated as hypotheses and tested afresh.
No historical number is ever reported as a result.

## Hypotheses

- **H1** (from C1): Proxy-guided stratified sampling improves overall balanced
  accuracy for melanoma classification (HAM10000 → binary mel vs benign) vs
  ERM, matched training budget.
- **H2** (from C2): Proxy-guided sampling improves dark-skin subgroup balanced
  accuracy, evaluated externally on Fitzpatrick17k (FST 5–6 vs 1–2).
- **H3** (from C3): Proxy-guided sampling reduces gender-based subgroup
  disparity in income prediction (Adult; German credit as secondary).

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| Phase 0 audit | DONE | `AUDIT.md` (C1–C3 unsupported) |
| Repo skeleton + unit tests | DONE | `pytest tests/` all pass (36 tests) |
| Simulation experiments | DONE | `results/sim_main/` — 3,416 records in ledger |
| Tabular Adult | DONE | `results/tabular_adult/` — 60 records (12 configs × 5 seeds) |
| Tabular German | DONE | 480 records (3 variants × 8 methods × 20 seeds); preds stored |
| HAM10000 data + cache | DONE | metadata sha256 `a430c61b…a6505`; cache sha256 `47684de4…b92`; sealed test split `06030a5e82ec0114` |
| Fitzpatrick17k data + eval cache | DONE | C.csv sha256 `bfc28411…fd219`; cache sha256 `9fcdd7a7…b365d`; eval checksum `377016a8e2c31858` (11,394 rows; images from original Groh release keyed by md5hash — 11,394/11,394 resolve) |
| Vision experiments (HAM→Fitz) | DONE | 63/63 records (3 proxy variants × 7 methods × 3 seeds), 52 min GPU |
| Vision aggregation + paired inference | DONE (code v2; summary to regenerate post-primary) | lesion-cluster bootstrap, corrected groups (fst0 = unknown excluded from light), seed t-CIs, FST-level decomposition |
| Dataset integrity audit | DONE | `results/audit/dataset_audit.json`; audit fix: `fst==0` = UNKNOWN (original `fitzpatrick_scale==-1`); light group = FST 1-2 only (n=1,758) |
| Threshold-protocol verification | DONE | `results/audit/protocol_verification.json`: stored scores reproduce bit-exactly; val batch pinned to 512 |
| 5-vs-10-epoch freeze (internal val only) | DONE | `results/audit/protocol_freeze.json`: 10ep worse on val (mean paired Δ=−0.032, 17/19 cells negative) → **5 epochs frozen**; sens10ep arm = exploratory only |
| Incident log | RESOLVED | sens10ep arm overwrote 20/21 eps0.0 main ckpts (shared run_id naming); preds/records unaffected (tag-keyed); ckpts restored by deterministic 5-epoch retrain, thresholds re-verified vs records; ckpt names now tag-suffixed for non-main arms |
| External-degradation investigation | DONE | `results/audit/external_degradation.json`: mechanism = prevalence shift (0.11→0.53) + operating-point shift + weak proxy validity (ρ(ITA,FST)=−0.31); per-FST deltas vs ERM |
| Vision 10-seed primary (seeds 3-9) | DONE | 147/147 runs; 210 main records total (21 configs × 10 seeds, 5-epoch frozen protocol) |
| Vision aggregation v2 (final, 10 seeds) | DONE | `results/vision_summary/vision-main.json`: cluster bootstrap, DeLong, Fisher+Holm, per-FST decomposition, seed t-CIs |
| Tabular paired inference vs ERM (v2, seed t-CIs) | DONE | `results/tabular_summary/tabular-{adult,german}-main.json`; no subgroup benefit; joint-strat harm in 2 configs |
| External-degradation investigation (10 seeds) | DONE | `results/audit/external_degradation.json` + `paper/tables/table6_mechanism.md` |
| Report tables + figures | DONE | `paper/tables/*.md` (6 tables), `paper/figures/*.png` (4 figures) |
| Manuscript | DONE (draft) | `paper/report.md` — all numbers trace to `results/**.json` |

## Data sources (licensing + provenance)

- HAM10000: Tschandl et al. 2018, Harvard Dataverse `doi:10.7910/DVN/DBW86T`, CC BY-NC-SA 4.0.
- Fitzpatrick17k metadata: Groh et al. 2021 (CC BY-NC-SA 3.0) via quality re-host
  (Abhishek et al. 2024, Zenodo `doi:10.5281/zenodo.12739457`); images from the
  original Groh release (Kaggle mirror `mobaswiralfarabi/fitzpatrick17k-original`)
  keyed by `md5hash`; FST labels used **only** for evaluation, never training.
- Adult / German: UCI mirrors, fetched by the loaders with checksums recorded.
- Reproduce with `scripts/download_data.sh`.

## Leakage controls

- HAM10000: lesion-level split (multi-image lesions never span splits); test
  set sealed with checksum; threshold + hyperparameters on val only.
- Fitzpatrick17k: external evaluation only; FST never used for training,
  sampling, or threshold selection.
- Proxy corruption uses per-eps deterministic seeds; random_proxy control
  destroys P–G association; all methods share architecture/optimizer/budget.

## Rules

- Every number in the manuscript must trace to a `results/**.json` record
  produced by this repo (seed, config hash, checksums recorded in the ledger).
- Append notable experiment summaries to `LEDGER.md`.
