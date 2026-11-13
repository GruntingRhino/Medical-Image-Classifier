# Proxy-Guided Sampling for Skin-Tone Subgroup Equity: A Re-registered Study

**Status: rebuilt.** This manuscript is written exclusively from records this
repository produced under a verified protocol (`results/**.json`; every number
traces to a record with seed, config hash, and data checksums). Earlier draft
claims that did not survive re-registration were either reproduced as
exploratory observations or dropped; see §History.

## Abstract

Proxy-guided sampling — using an observable correlate of a demographic group
(Individual Typology Angle, ITA°) to restructure training data — was proposed
to improve subgroup accuracy without demographic labels. We re-registered the
claim under a verified protocol (lesion-level splits, sealed test set,
thresholds selected on internal validation only, integrity audit of both
datasets, 10 training seeds, paired cluster-aware inference). On HAM10000 →
Fitzpatrick17k melanoma classification we find: **(a)** in its raw form the
intervention is inert — with uniform ITA quantile bins, stratified sampling,
importance weighting, and the shuffled-proxy control produce sampling weights
identical to ERM; **(b)** no informative variant improves external or
dark-skin (FST 5–6) balanced accuracy by a practically meaningful margin
(all |Δ| ≤ 0.013 with CIs spanning zero width ≈ ±0.02), while class
stratification (ΔFitz BAcc = −0.019, Holm p < 10⁻³) and proxy DRO
(−0.016, p < 10⁻³) significantly *worsen* external accuracy; **(c)** the
mechanism is an operating-point shift under a 5× disease-prevalence change
(HAM 11.1% → Fitz 52.8%), not a subgroup-targeted effect — the proxy's
per-image skin-tone validity is weak (Spearman ρ(ITA, FST) = −0.31) and its
border-region statistics are acquisition-dominated (KS = 0.60–0.71 across
datasets). On Adult and German credit no method reduces the gender gap
(|Δ BAcc| ≤ 0.006 everywhere except one harmful configuration at −0.082).
We release the protocol, audit tooling, and all records.

## 1. Introduction

Chronic under-representation of dark skin in dermatology datasets is both an
equity problem and a measurement problem: models trained on light-skin
majorities may be less accurate for dark-skin patients, and datasets lacking
skin-tone labels cannot even measure the gap. Proxy-guided sampling proposes
to fix the first problem using only an image statistic (center-patch ITA) as
a stand-in for skin tone, by rebalancing training data toward the
under-represented proxy strata.

Prior drafts of this project claimed large subgroup gains (external FST 5–6
BAcc +0.13). A phase-0 audit found those claims unsupported by any recorded
experiment: no seed logs, no inference, and an aggregation script that shared
one threshold across methods (see `AUDIT.md`). This paper reports the
re-registered study.

### Contributions

1. **Dataset integrity audit** (`results/audit/dataset_audit.json`): duplicate
   detection across splits, lesion-label consistency, rename integrity of the
   Fitzpatrick17k re-host, and an audit correction: `fst == 0` in the re-host
   corresponds to the original `fitzpatrick_scale == -1` (unknown), not a low
   FST type — earlier loaders polluted the light group with 104 unknown images.
2. **Protocol verification** (`results/audit/protocol_verification.json`):
   stored scores reproduce bit-exactly from checkpoints; thresholds are
   validation-optimal; validation batch size pinned to 512 because threshold
   selection is sensitive to float accumulation order.
3. **Epoch-count freeze decided on internal validation only**
   (`results/audit/protocol_freeze.json`): 10-epoch training is *worse* on
   validation (mean paired Δ = −0.032 BAcc across 19 method×seed cells);
   5 epochs frozen before the primary run.
4. **A 10-seed, 3-variant, 7-method primary experiment** (210 runs, 70 per
   proxy variant) with paired, cluster-aware inference (lesion-level
   bootstrap; DeLong; Fisher-combined p over seeds; Holm correction within
   metric families).
5. **A mechanism analysis of external degradation** combining distribution
   shift, prevalence shift, and operating-point measurements
   (`results/audit/external_degradation.json`).
6. **Tabular replications** on Adult and German credit (gender proxy) with
   cell-stratified bootstrap inference.

## 2. Methods

### 2.1 Data and integrity

- **HAM10000** (Tschandl et al. 2018, CC BY-NC-SA 4.0): binary task
  melanoma vs benign; lesion-level split 80/10/10 (multi-image lesions never
  span splits; 2 exact duplicate groups found, none spanning splits; 0 label
  inconsistencies). Melanoma prevalence 11.1% (train).
- **Fitzpatrick17k** (Groh et al. 2021, CC BY-NC-SA 3.0; quality re-host by
  Abhishek et al. 2024): external evaluation set of 3,190 malignant/benign
  images with FST labels; malignant prevalence 52.8%; FST 0 = unknown
  (original −1) excluded from binary group contrasts; content md5 == filename
  for 3,190/3,190; cross-dataset content overlap 0.
- **Adult / German credit** (UCI): gender-proxy subgroup evaluation;
  480 German records (3 proxy variants × 8 methods × 20 seeds) and 60 Adult
  records.

### 2.2 Proxy and sampling methods

Center-patch ITA° (del Lentil? see `src/proxystudy/proxies/`) computed from
the cached preprocessed images; 5 quantile bins on the training split.
Methods: ERM; class stratification; proxy stratification; joint (bin, label)
stratification; proxy importance weighting; proxy DRO (exponential-weights
over bin losses, η=0.5); shuffled-proxy control. Three proxy-corruption
variants ε ∈ {0, 0.35, 0.7} mix the true ITA with shuffled or random proxy
values. All methods share architecture (ResNet-18, 224²), optimizer, batch
size, and budget; weights enter through a weighted sampler. Sampling-weight
effective sample sizes are reported per method (they are 100% for the
bin-uniform raw-proxy arms — i.e., those arms are operationally ERM).

### 2.3 Protocol guarantees

- Threshold (and any hyperparameter) selected on the internal validation
  lesions only; validation batch pinned to 512 for reproducibility of the
  threshold sweep.
- FST labels never used in training, sampling, binning, or threshold
  selection (verified by code inspection + the shuffled-proxy control).
- Test split sealed by checksum before external evaluation began.
- Primary run: seeds 0–9; seeds 0–2 retained from the verified grid, 3–9 run
  under the frozen protocol; all inference paired within seed.

### 2.4 Inference

- Paired per-seed deltas vs ERM (same split, same protocol); lesion-cluster
  bootstrap (1,000 draws, resampling lesions, stratified by group×label cell)
  for the primary tests; image-level tests reported for comparison only.
- DeLong paired test for AUROC deltas.
- Seed-level combination: Fisher's method over per-seed p-values; Holm-
  Bonferroni across methods within each metric family (HAM BAcc, Fitz BAcc,
  Fitz AUROC, dark BAcc, light BAcc, worst-group BAcc, Δ(dark−light)).
- Seed-training variability reported separately as per-cell t-CIs.

## 3. Results

Full tables: `paper/tables/` (regenerable via `report/tables.py`); figures:
`paper/figures/` (`report/figures.py`). All vision numbers below are the
10-seed primary (frozen 5-epoch protocol), paired within seed against ERM.

### 3.1 The intervention is inert in its raw form

With 5 uniform quantile bins of the raw ITA proxy on HAM10000,
`proxy_strat`, `proxy_iw`, and `random_proxy_strat` produce sampling weights
identical to ERM (quantile binning ⇒ uniform bin occupancy ⇒ equalization is
a no-op; ESS = 7,036/7,036 for all three). Their paired deltas vs ERM
(ΔFitz BAcc −0.002 to +0.007, ΔHAM BAcc −0.008 to −0.004) reflect training
nondeterminism only, and sit within the seed-level CIs (±0.01–0.02). They
are retained in the Holm family for completeness.

### 3.2 Informative variants do not improve external subgroup accuracy

Absolute external BAcc (eps 0.0, mean over 10 seeds): ERM 0.548; proxy arms
0.530–0.555; worst-group 0.511–0.528; dark-group (FST 5–6) 0.511–0.531 for
every method including ERM (0.528). External AUROC spans 0.540–0.582 versus
internal HAM AUROC 0.88–0.90: the domain gap, not the sampling rule,
dominates.

Paired deltas (Holm-adjusted across methods within each family):

- **Significant harms**: class_strat ΔFitz BAcc = −0.019 (Holm p < 10⁻³,
  ΔHAM BAcc −0.014 n.s.); proxy_dro ΔFitz = −0.016 (p < 10⁻³) and
  ΔHAM = −0.014 (p = 0.037). Neither helps the dark group (class_strat
  Δdark = +0.000 n.s.; proxy_dro Δdark = −0.018, CI [−0.04, 0.00], n.s.).
- **No meaningful gains**: proxy_iw +0.004, proxy_strat −0.002,
  proxy_strat_joint −0.004, random_proxy_strat +0.007 — all with CIs
  spanning ±0.02; the small "Holm-significant" p-values reflect sign
  consistency of ≤0.7% deltas across 10 seeds, not practical effect.
- **Corrupted-proxy variants**: at ε = 0.3, proxy_strat +0.011 and
  random_proxy_strat +0.013 (CIs [−0.01, 0.03]); at ε = 0.6 all proxy arms
  within ±0.008. class_strat harms at every ε (−0.014/−0.019/−0.020).

Conclusion: proxy-guided sampling does not transfer subgroup gains to the
external domain at this scale, and the aggressive reweighting variants hurt.

### 3.3 Mechanism of external degradation

From `results/audit/external_degradation.json` (10-seed means):

- **Prevalence shift**: melanoma prevalence 11.1% (HAM train) vs 52.8%
  (Fitz eval); per-FST 0.47–0.62. With HAM-validation thresholds, predicted-
  positive rates on Fitz range 0.31–0.63 (light 0.31–0.47, dark 0.40–0.63)
  against true prevalence ≈ 0.51/0.53 — every method miscalibrates; the
  conservative proxy arms (proxy_strat pred-pos dark 0.415, sens 0.442) pay
  in dark-group sensitivity relative to ERM (0.571); the aggressive joint
  arm (0.624, sens 0.640) pays in specificity (0.394 vs ERM 0.458).
- **Proxy validity**: Spearman ρ(ITA, FST) = −0.31 on Fitz eval; FST 5–6
  images place 72% of their mass in the two darkest HAM ITA bins but with
  heavy overlap across levels; marginal HAM↔Fitz ITA overlap 0.73.
- **Acquisition confound**: border-region statistics differ radically
  (border ITA KS = 0.71; border luminance KS = 0.60) while center luminance
  matches (KS = 0.13) — the center-patch choice was necessary but the CNN
  still faces a domain the proxy cannot characterize.
- **FST-level decomposition** (ΔBAcc vs ERM per level): class_strat's harm
  is concentrated at FST 1–3 (−0.023/−0.023/−0.025) and absent at FST 5–6
  (+0.004/0.000); proxy_dro's harm is broad (−0.040 at FST 6). No method
  shows a positive dark-group effect at any level beyond noise.

Interpretation: the intervention reshapes the training distribution toward
the source domain's acquisition geometry; under a 5× prevalence shift the
operating point — not subgroup representation — dominates the external
metric.

### 3.4 Tabular results

Adult (12 configs × 5 seeds) and German credit (3 variants × 8 methods × 20
seeds), cell-stratified paired bootstrap + DeLong + McNemar, Fisher-combined
over seeds, Holm within family. Tabular models are deterministic given
weights, so seed replication confirms determinism (per-seed delta sd = 0)
and inference rests on the item-level bootstrap. Outcome: no method reduces
the gender gap; deltas ≤ 0.001 everywhere except one harmful configuration —
`proxy_strat_joint` under the empirical marital-status proxy on Adult
(ΔBAcc = −0.082, Holm p < 10⁻³) and on German at ε = 0.3 (−0.028, p < 10⁻³).

### 3.5 Sensitivity arms (exploratory; never used for decisions)

- **10 epochs** (seeds 0–2): internal validation is *worse* (mean paired
  Δval = −0.032, 17/19 cells negative — this decided the 5-epoch freeze);
  external BAcc at 10 epochs is similar (ERM 0.536; proxy_strat_joint 0.557).
- **QC-flag exclusion** and other post-hoc cuts: reported only if run;
  otherwise not claimed.

## 4. Discussion

The negative result is the finding: a plausible, well-motivated sampling
intervention fails to transfer subgroup gains from an internal dermoscopic
domain to an external clinical domain. The mechanism analysis explains why:
the proxy measures acquisition, the intervention reshapes the training
distribution toward the source domain, and the operating point — not the
subgroup representation — dominates the external metric under a 5×
prevalence change. Sampling interventions without demographic labels inherit
the proxy's validity limit: ρ(proxy, group) ≈ −0.31 here, and the intervention
cannot distinguish "dark skin" from "darkfield illumination".

Limitations: two-dataset scope; binary FST contrast; ResNet-18 scale;
QC-flagged rows retained (exploratory exclusion reported); 10 seeds rather
than 30.

## 5. History of unverified prior claims

Prior internal drafts (pre-audit) claimed: external FST 5–6 BAcc +0.13 for
proxy-guided sampling; simulation gains for the joint method; Adult gap
reductions. Phase-0 audit (`AUDIT.md`) found no records supporting them. The
re-registered study reports the numbers above; any qualitative agreement
between old claims and new results is coincidental unless the new records
show it.

## Reproducibility

`scripts/download_data.sh` (checksums), `src/proxystudy/data/` (loaders with
checksums), `src/proxystudy/exp/run_vision.py --tag main` (primary),
`aggregate_vision.py --tag main --n-boot 1000`, `report/tables.py`,
`report/figures.py`. Ledger: `results/ledger.jsonl` (every record appended
with config hash + data checksums). Audit trail: `results/audit/*.json`.
