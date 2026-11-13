# LEDGER — experiment batches (append-only)

Machine-readable provenance lives in `results/ledger.jsonl` and
`results/<experiment>/<record>.json`. This file summarizes batches in order.

## 2026-11-11 — sim_main (controlled simulations)

- Runner: `proxystudy.sims.run_sims`
- 3,520 records: DGP conditions × proxy quality × methods × 50 seeds.
- Purpose: statistical-conditions map — when does proxy-guided sampling
  help, and when does it hurt (controls: random proxy, confounded proxy,
  class-only stratification).

## 2026-11-11 — tabular_adult (Adult income, H3)

- Runner: `proxystudy.exp.run_tabular --dataset adult`
- 60 records: 12 configs (eps corruption 0.1–0.9, confounded, empirical
  marital proxy) × 5 seeds; 8 methods incl. oracle (reference only).
- Official train/test split; threshold on 10% val carve-out; test evaluated
  once per run.

## 2026-11-11 — tabular_german (German credit, H3 secondary)

- Runner: `proxystudy.exp.run_tabular --dataset german --n-seeds 20`
- 480 records: 3 proxy variants (eps 0.3 / 0.6 / confounded 0.6) × 8 methods
  × 20 seeds. Fixed 800/200 split (seed 20261111), 12.5% val carve-out.

## 2026-11-11 — vision_ham10000 (H1, H2) — COMPLETE

- Runner: `proxystudy.exp.run_vision --tag main --n-seeds 3`
- 63 records: proxy variants (raw ITA; eps-noise 0.3, 0.6) × 7 methods ×
  3 seeds. oracle_strat excluded by design (no true group at training).
- Training: ImageNet ResNet-18, 5 epochs, AdamW 1e-4, batch 64, identical
  budget; threshold = max-val-BAcc (per method).
- Evaluation: sealed HAM10000 test (checksum `06030a5e82ec0114`) +
  Fitzpatrick17k external (checksum `377016a8e2c31858`).
- Absolute seed means (raw proxy): ERM HAM BAcc 0.804±0.018, Fitz 0.558
  (dark 0.533); best proxy method proxy_strat HAM 0.819±0.013, Fitz 0.541
  (dark 0.514).
- Paired vs ERM (Fisher across seeds, Holm-adjusted): NO proxy method
  significantly improves HAM test BAcc (all Holm p ≥ 0.06, deltas +0.001
  to +0.015). ALL proxy methods significantly WORSEN Fitzpatrick17k
  external BAcc (deltas −0.013 to −0.028, Holm p < 0.002) and worst-group
  (dark) BAcc. Shuffled-proxy control ≈ ERM (sanity check passes).
- Bug fixed during analysis: paired loop initially applied the method's
  threshold to ERM scores, inflating deltas; corrected to per-method
  thresholds before final aggregation (artifact not present in final JSON).
- Headline: in this rebuilt, leakage-safe protocol, proxy-guided sampling
  does not reproduce the historical claims (C1: +5.6 pts overall, C2:
  +4.7 pts dark-skin); it slightly harms external dark-skin transfer.

## 2026-11-11 — tabular paired inference (H3)

- Re-ran `run_tabular` with `--store-preds` (records idempotent: same
  config hashes/seeds overwrite identical files) and ran
  `proxystudy.exp.aggregate_tabular`.
- Adult: proxy methods ≈ ERM everywhere (|ΔBAcc| ≤ 0.002, n.s. after
  Holm). oracle_strat slightly but significantly worse (AUROC Δ −0.002,
  p ≈ 1e-28 — DeLong power on n≈15k; magnitude negligible). Striking:
  `proxy_strat_joint` on the empirical marital proxy HURTS badly
  (ΔBAcc −0.082, p ≈ 1e-8).
- German (20 seeds): deltas ±0.013–0.018, all n.s. after Holm.
- `stats/inference.py`: DeLong placement computation rewritten to stream
  in chunks (O(n) memory) for the 15k-item Adult test set; validated
  against prior exact float64 behavior (test suite passes).

## 2026-11-11 — Primary 10-seed vision run + final aggregation

- Protocol freeze (internal val only): 10ep worse (Δval = −0.032, 17/19 cells
  negative) → 5 epochs frozen (`results/audit/protocol_freeze.json`).
- Primary: seeds 3–9 run under frozen protocol (147 runs); main tag now has
  21 configs × 10 seeds = 210 records (5 epochs each).
- Incident: sens10ep arm overwrote 20/21 eps0.0 main ckpts (shared run_id);
  preds/records unaffected (tag-keyed). Restored by deterministic retrain;
  thresholds re-verified vs records (63/63 match). Ckpt names now
  tag-suffixed for non-main arms; protocol-mismatch guard added to resume.
- Final aggregate (lesion-cluster bootstrap, DeLong, Fisher+Holm, seed
  t-CIs, per-FST levels): `results/vision_summary/vision-main.json`.
  Headline (eps0.0, paired vs ERM): no proxy arm improves external or
  dark-group BAcc beyond noise (|Δ| ≤ 0.013, CIs ±0.02); class_strat −0.019
  and proxy_dro −0.016 significantly harm external BAcc (Holm p < 1e-3).
- Mechanism audit (10 seeds): prevalence 0.111→0.528; ρ(ITA,FST) = −0.31;
  border KS 0.60–0.71; per-FST harm of class_strat at FST1-3, none at FST5-6.
- Tabular v2 summaries regenerated with seed t-CIs: Adult + German show no
  subgroup benefit; proxy_strat_joint harmful in two configs (−0.082 Adult
  marital_empirical; −0.028 German ε0.3); tabular models deterministic
  (per-seed delta sd = 0) — inference rests on item-level bootstrap.
- Deliverables: `paper/report.md` (draft), `paper/tables/*.md` (6),
  `paper/figures/*.png` (4). Tests: 36 passed.
