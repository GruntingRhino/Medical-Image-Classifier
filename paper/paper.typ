// Manuscript. Compile from the repository root:
//   uv run --no-project --with typst==0.13.2 python -c \
//     "import typst; typst.compile('paper/paper.typ', output='paper/paper.pdf', root='.')"
// Every table is read from paper/typ_tables.json and paper/numbers.json,
// which src/proxystudy/report/hier_report.py generates from results/**.json.

#let T = json("typ_tables.json")
#let N = json("numbers.json")
#let pct(x) = str(calc.round(x * 100, digits: 1)) + "%"
#let r3(x) = str(calc.round(x, digits: 3))
#let cells(rows) = rows.flatten().map(c => c.split("\n").map(x => [#x]).join(linebreak()))

#set document(title: "Proxy-guided sampling and skin-tone equity: a re-registered audit", author: "Abhay Sivaram")
#set page(paper: "us-letter", margin: (x: 1in, y: 1in), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set block(above: 1.4em, below: 0.8em)
#show figure.caption: set text(size: 9pt)
#show table: set text(size: 8pt)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none }, inset: (x: 4pt, y: 3pt))
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 15pt, weight: "bold")[Proxy-Guided Sampling Does Not Improve Skin-Tone Equity\ in Melanoma Classification]
  #v(0.3em)
  #text(size: 11.5pt)[A re-registered audit, and why combining training seeds with Fisher's method fails]
  #v(0.8em)
  Abhay Sivaram \
  #text(size: 9pt)[Independent research · November 2026 · Preprint, not peer reviewed]
]

#v(0.6em)
#block(inset: (x: 2.2em))[
  #set text(size: 9.5pt)
  *Abstract.* Medical image classifiers can perform worse for patients with darker skin, but skin-tone labels are rarely recorded. A natural fix is to rebalance training data using an observable _proxy_ for skin tone, such as image colour. An earlier version of this project reported large gains from doing so (+5.6 balanced-accuracy points in melanoma detection, and a 27% smaller gender gap in income prediction). We audit those claims, find that none is supported by a controlled experiment, and re-test the idea under a pre-specified protocol: HAM10000 dermoscopy for training, external evaluation on Fitzpatrick17k clinical photographs with dermatologist skin-type labels, 10 training seeds per method, and paired, cluster-aware inference. We report four findings. (i) The originally described algorithm is a mathematical no-op: equalising quantile bins leaves the training set unchanged. (ii) No proxy-based method changed internal, external or darkest-skin (Fitzpatrick V–VI) balanced accuracy detectably; the study could detect effects of about #pct(N.mde_80.fitz_overall.median) overall and #pct(N.mde_80.fitz_dark.median) for Fitzpatrick V–VI. (iii) External transfer itself fails (AUROC ≈ 0.57), which no sampling rule repairs, and Fitzpatrick17k contains only #N.melanoma_subset.counts.dark.melanoma melanoma images of Fitzpatrick type V–VI. (iv) Of #N.global.n_tests primary-endpoint tests, #N.global.n_sig survive family-wise correction, and all three are _harms_: proxy methods widened the gender gap. Along the way we found that the analysis pipeline combined per-seed p-values with Fisher's method. Because every seed is scored on the same test patients, those p-values are dependent; in simulation this procedure rejected a true null up to #pct(N.sim_seed.max_null_rate.fisher) of the time, and on real data it declared arms that are identical to the baseline "significant" at $p approx 10^(-9)$. We give a hierarchical bootstrap over patients and seeds that stays at or below nominal error, and we distinguish the two questions a seed analysis can answer.
]

= Introduction

Dermatology AI is trained mostly on light skin, and performs worse on dark skin @daneshjou2022 @groh2021. Correcting this usually requires knowing each patient's skin type, which hospitals rarely record. "Fairness without demographics" methods therefore try to improve the worst-off group without group labels, either by reweighting high-loss examples @hashimoto2018 @lahoti2020 or by using an observable proxy correlated with the group @chen2019 @kallus2022. In dermatology the obvious proxy is skin colour measured from the image, summarised by the Individual Typology Angle (ITA) @chardon1991 @kinyanjui2020.

An earlier version of this project proposed _P-Stratified Sampling_: bin the training data by the proxy, then oversample each bin to equal size. It reported a melanoma balanced accuracy of 0.897 versus 0.841 for standard training, and a 27% reduction in a gender accuracy gap on the Adult income data, with a "provable" bound. This paper replaces that version. Its contributions are:

+ *An audit of the original claims* (§2). All three are unsupported: the headline medical row came from a different pipeline run than the baselines it was compared with, compared models at different thresholds, and defined the "darker" group by the proxy itself.
+ *A simple impossibility result* (§3.3): with quantile bins, the original algorithm does nothing.
+ *A re-registered, multi-seed external evaluation* (§4–5) with seven methods, three levels of proxy corruption and two tabular replications, reported in full, including null results and their detectable effect sizes.
+ *A statistical correction with general relevance* (§4): combining per-seed tests with Fisher's method is invalid when seeds share a test set. We quantify the failure in simulation, demonstrate it on real data with built-in negative controls, and give a calibrated alternative.
+ *A mechanism analysis* (§5.4) explaining why external performance collapses, including a label-definition mismatch between the two datasets that the frozen protocol missed.

= Audit of the original claims

The original code and results were located on the author's machine and inspected file by file (`AUDIT.md`).

#figure(
  table(columns: (auto, 1fr), align: (left, left),
    [*Claim*], [*Why it is unsupported*],
    [Melanoma BA 0.841 → 0.897 (C1)], [The "ours" row was copied from a different training run (`clinical_grade_metrics.json`, joint label × brightness stratification) rather than produced by the controlled baseline script. The significance test compared models at thresholds 0.35 and 0.82.],
    [Darker-group BA 0.824 → 0.871 (C2)], [The "darker" group was defined by image brightness, the proxy itself. The test split contained only 4 images of Fitzpatrick type V–VI.],
    [27% gap reduction on Adult (C3)], [One training run; a p-value from bootstrapping the test set only; the sex/hours analysis was added after the race analysis was not significant. Multi-seed replication finds no reduction (§5.3).],
    [Algorithm and code], [Quantile-bin equalisation is a no-op (§3.3). The public repository contained only a licence file, and `run_all.sh` called scripts that did not exist.],
  ),
  caption: [Original claims and audit findings.],
)

= Methods

== Data
*HAM10000* @tschandl2018: 10,015 dermoscopic images of 7,470 lesions; task melanoma versus all other diagnoses (prevalence 11.1%). Splits are by lesion (80/10/10), so images of one lesion never cross splits; the test split was sealed by checksum. *Fitzpatrick17k* @groh2021, using the quality-controlled re-host of Abhishek et al.: 3,190 clinical photographs labelled malignant or benign, with dermatologist Fitzpatrick skin type (FST). Type "0" in the re-host is the original "unknown" and is excluded from group contrasts. We compare FST V–VI (_dark_, n = 255) with FST I–II (_light_, n = 1,758). Skin type is used only for evaluation, never for training, sampling or threshold choice. *Adult* and *German credit* (UCI) provide the tabular replication, with gender as the group.

== Proxy and methods
The proxy is the ITA of the central image patch, $"ITA" = arctan((L^* - 50) \/ b^*)$ in CIE Lab, binned into 5 training-set quantiles. Seven methods share one architecture (ImageNet-pretrained ResNet-18, 224 px, AdamW, learning rate $10^(-4)$, 5 epochs fixed in advance on validation data), differing only in how training examples are weighted: *ERM* (no reweighting); *class_strat* (balance labels); *proxy_strat* (equalise proxy bins: the original method); *proxy_strat_joint* (equalise proxy-bin × label cells); *proxy_iw* (inverse bin-frequency weights); *proxy_dro* (exponentiated-gradient weights on bin losses, $eta = 0.5$, after group DRO @sagawa2020); and *random_proxy_strat*, a negative control that uses a shuffled proxy. The proxy is also corrupted at rates $epsilon in {0, 0.3, 0.6}$. The decision threshold for each run is the one maximising balanced accuracy on internal validation lesions; it is then applied unchanged to both test sets.

== Quantile equalisation is a no-op
#block(inset: (x: 1em))[
  *Proposition 1.* Let a training set of size $n$ be split into $m$ bins at the empirical $k\/m$ quantiles of a proxy with no ties. Then every bin contains $floor(n\/m)$ or $ceil(n\/m)$ examples, so oversampling each bin to the size of the largest repeats every bin once, and the resampled training set equals the original.

  _Proof._ By definition of empirical quantiles, each bin receives the same number of order statistics up to rounding. The original algorithm repeats bin $b$ $ceil(n_max \/ |S_b|) = 1$ times. #h(1fr) $square$
]
On HAM10000 the effective sample size under `proxy_strat`, `proxy_iw` and `random_proxy_strat` was 7,036 of 7,036: these arms train on exactly the ERM distribution, and any difference from ERM is training noise. This makes them natural negative controls (§4.4). Bins only differ in size when the proxy has ties (as for the discrete education variable in the original Adult analysis) or when bins are defined jointly with the label.

= Statistical inference

== Two questions a seed analysis can answer
Each method is trained with seeds $s = 1, dots, S$, and every seed is scored on the _same_ test set. Write $Delta_s$ for the difference in balanced accuracy between a method and ERM trained with the same seed. Two estimands are possible:

- *Conditional:* is the method better than ERM _on these test patients_, averaging over training randomness? Uncertainty comes from seeds only; a one-sample $t$-test on ${Delta_s}$ answers it.
- *Population:* is the method better _for new patients_ from the same population? Uncertainty comes from both seeds and patients.

Clinical claims are population claims, so the population estimand is primary here. Neither question is answered by testing each seed separately on bootstrap resamples of the test set and combining the $S$ p-values with Fisher's method, which was the pipeline's original approach. Fisher's method assumes independent experiments, but the per-seed tests share every test patient, and each per-seed test asks whether two _fixed_ fitted models differ, which is almost always true because of training noise.

== Hierarchical paired bootstrap
In each of $B = 9999$ draws we (1) resample test units: whole lesions for HAM10000, where one lesion can have several images, and images within (group × label) cells for Fitzpatrick17k and the tabular data; (2) resample seed indices with replacement, keeping each method paired with ERM at the same seed; and (3) record the seed-averaged difference $macron(Delta)^*$. Balanced accuracy under resampling is computed from multiplicity weights, so a draw costs one matrix–vector product per method. With few seeds the bootstrap underestimates between-seed variance, so the primary test uses a $t_(S-1)$ reference on the bootstrap standard error inflated by $sqrt(S\/(S-1))$:
$ t = macron(Delta) / (hat(sigma)^* sqrt(S\/(S-1))), quad p = 2 P(T_(S-1) > |t|). $
When training is deterministic (identical $Delta_s$ for every seed, as for the tabular models), seed variance is known to be zero and the percentile bootstrap is used. Multiplicity is controlled with Holm's method @holm1979 across the six methods within each (dataset, proxy variant, metric) family, and, as a sensitivity analysis, across all #N.global.n_tests primary-endpoint tests derived from the registered hypotheses H1–H3: HAM10000 overall and Fitzpatrick V–VI balanced accuracy for vision, and the gender gap for the tabular data.

== Calibration study
We simulated two training procedures scored on one shared test set of 400 items, varying the number of seeds ($S in {3, 5, 10, 20}$) and how much fitted models disagree on individual items (seed noise $tau$). Under the *strong null* the procedures are identical in distribution. Under a *heterogeneous null* each procedure helps a different random half of patients, so the population difference is zero but the difference on any finite test set is not. Each cell has 1,000 replications.

#figure(
  table(columns: 7, align: (center,) * 7,
    [*Seeds*], [*$tau$*], [*Fisher*], [*Pooled*], [*Seed $t$*], [*Hier. pct.*], [*Hier. $t$*],
    ..T.sim_null.flatten(),
  ),
  caption: [Type-I error at nominal $alpha = 0.05$ under the strong null. "Pooled" treats every (seed, item) prediction as an independent patient. Monte Carlo standard error ≈ 0.007.],
) <tab-sim>

#figure(image("figures/fig_sim_calibration.png", width: 100%),
  caption: [Type-I error by seed noise for each procedure (strong null). Dashed line: nominal 0.05.]) <fig-sim>

Fisher's method fails badly as soon as seeds disagree (@tab-sim, @fig-sim), and pooling predictions across seeds is little better. The seed-level $t$-test is calibrated for the conditional question, and the hierarchical $t$ test stays at or below nominal level in every cell. It is conservative, and that costs power: at 10 seeds and $tau = 0.3$, a true improvement of 0.05 in score margin is detected #T.sim_power.at(2).at(7) of the time by the hierarchical test versus #T.sim_power.at(2).at(5) by the seed $t$-test. #if "sim_het" in T [Under the heterogeneous null (@tab-het), however, the seed $t$-test's rejection rate climbs above 5% as patient-level heterogeneity and the number of seeds grow, reaching #N.sim_seed_het.max_null_rate.seed_t with 20 seeds. It is correctly detecting that this particular test set favours one procedure, but that is not a population effect. The hierarchical $t$ test stays at or below nominal level in every heterogeneous cell. With mild heterogeneity the two tests agree, so the choice matters most when many seeds make the seed-level test very precise.]

#if "sim_het" in T [
#figure(
  table(columns: 8, align: (center,) * 8,
    [*Seeds*], [*$tau$*], [*$h$*], [*Fisher*], [*Pooled*], [*Seed $t$*], [*Hier. pct.*], [*Hier. $t$*],
    ..T.sim_het.flatten(),
  ),
  caption: [Rejection rate under the heterogeneous null (population difference zero; subtype effect size $h$).],
) <tab-het>
]

== Real-data negative controls
`random_proxy_strat` and `proxy_strat` train on the ERM distribution (Proposition 1), so their true effect is zero. The original Fisher-plus-Holm analysis nevertheless declared them significantly different from ERM on Fitzpatrick17k, with adjusted p-values as small as $10^(-9)$ (@tab-paired, column "old"). The hierarchical analysis gives adjusted $p = 1.0$ for both.

= Results

== Absolute performance
#figure(
  table(columns: 7, align: (left,) + (center,) * 6,
    [*Method*], [*HAM BA*], [*HAM AUROC*], [*Fitz BA*], [*Fitz AUROC*], [*Fitz V–VI BA*], [*Fitz I–II BA*],
    ..T.absolute.flatten(),
  ),
  caption: [Absolute performance, mean (SD) over 10 seeds, uncorrupted proxy. HAM = internal test lesions; Fitz = external Fitzpatrick17k.],
) <tab-abs>

Internally, every method reaches balanced accuracy (BA) of 0.81–0.82 and AUROC of 0.88–0.90. Externally, all methods are close to chance (BA 0.53–0.56, AUROC 0.54–0.58), and every method does worse on FST V–VI than on FST I–II (@tab-abs).

== Paired comparisons with ERM
#figure(
  table(columns: 8, align: (left,) + (center,) * 7,
    [*Method*], [*Δ HAM BA*], [*p (Holm)*], [*Δ Fitz BA*], [*p (Holm)*], [*p (old)*], [*Δ Fitz V–VI BA*], [*p (Holm)*],
    ..cells(T.paired),
  ),
  caption: [Paired differences from ERM [95% CI], uncorrupted proxy, hierarchical inference. "old" = superseded Fisher-over-seeds + Holm p-value for Δ Fitz BA.],
) <tab-paired>

#figure(image("figures/fig_forest_hier.png", width: 100%),
  caption: [Paired differences from ERM with 95% CIs (hierarchical, 10 seeds), uncorrupted proxy.]) <fig-forest>

No method's effect survives correction on any vision endpoint (@tab-paired, @fig-forest). The same holds at proxy corruption $epsilon = 0.3$ and $0.6$ (supplementary table 7). The only consistent pattern is a small harm from class stratification and proxy DRO, about −1 to −2 BA points both internally and externally. That harm is nominally significant in several analyses but never after correction. For FST V–VI, every confidence interval spans roughly ±0.05. The minimum effect detectable with 80% power was #pct(N.mde_80.fitz_dark.median) for FST V–VI and #pct(N.mde_80.fitz_overall.median) for external BA overall. Effects of the size the original paper claimed (+4.7 points for the darker group) would therefore probably have been missed for FST V–VI, but effects of that size overall would not. As a sensitivity check, the seed-level $t$-test (the conditional question) gives the same conclusion: no proxy method improves any endpoint, and only one harm survives Holm correction (proxy DRO on HAM10000 at $epsilon = 0.6$).

== Tabular replication
Logistic models on Adult (5 seeds) and German credit (20 seeds) are deterministic given the sample weights, so inference is over test rows. Across all proxy variants, no method narrowed the gender gap. Of the #N.global.n_tests primary-endpoint tests (vision and tabular), #N.global.n_sig survive Holm correction across all of them. All three are increases in the Adult gender gap of 2.4–3.7 points:

#figure(
  table(columns: 6, align: (left,) * 3 + (center,) * 3,
    [*Data*], [*Proxy variant*], [*Method*], [*Δ gap [95% CI]*], [*p*], [*p (global Holm)*],
    ..T.tabular_sig.flatten(),
  ),
  caption: [The only effects surviving global correction: proxy methods _widening_ the gender gap. "confounded" proxies carry a spurious label channel; "corrupt_eps0.9" is 90% noise; "marital_empirical" uses marital status.],
)

When the proxy carries label information (confounded) or almost no group information (heavily corrupted), reweighting by it harms the subgroup it was meant to help. This is the confounded-proxy failure mode the original paper described in synthetic data, now observed on real data.

== Why external performance collapses
*Domain gap.* HAM10000 images are dermoscopic: magnified, polarised and lit uniformly. Fitzpatrick17k images are ordinary clinical photographs. A model at 0.90 AUROC internally falls to about 0.57 externally whatever its sampling rule.

*Label-definition mismatch (found in this work).* The model is trained to separate melanoma from everything else, so HAM10000's basal-cell carcinomas and actinic keratoses are _negatives_. Fitzpatrick17k's "malignant" class, however, is mostly non-melanoma: of 1,683 positives only #N.melanoma_subset.n_melanoma are melanoma, alongside 499 basal-cell and 372 squamous-cell carcinomas. The frozen protocol therefore evaluates a different task than the model was trained for. As an exploratory, post-hoc check we re-scored the stored predictions, at the same validation thresholds, on the construct-matched subset (melanoma vs benign, n = #N.melanoma_subset.n). ERM improves only slightly, to BA #r3(N.melanoma_subset.erm_eps0.bacc) and AUROC #r3(N.melanoma_subset.erm_eps0.auroc). The mismatch is real but is not the main cause; the domain gap is. On this subset no method improves on ERM. Class stratification again shows a harm, here −0.035 BA (unadjusted $p = 0.016$), and is again the method most consistently below ERM.

*Too few dark-skin melanomas.* The construct-matched subset contains #N.melanoma_subset.counts.dark.melanoma melanomas in FST V–VI (out of #N.melanoma_subset.counts.dark.n images), against #N.melanoma_subset.counts.light.melanoma in FST I–II. No study using this benchmark can estimate dark-skin melanoma sensitivity precisely.

*The proxy is partly confounded with the label.* Centre-patch ITA measures the lesion as much as the skin. In HAM10000 the darkest proxy quintile has 18.8% melanoma versus 8–11% in the other quintiles. Reweighting by proxy therefore also changes class balance, which moves the operating point; this matches the observed shifts in predicted-positive rates (supplementary table 6). Its rank correlation with dermatologist skin type on Fitzpatrick17k is only −0.31.

= Discussion

The idea that a cheap image-colour proxy can stand in for skin-type labels does not survive a controlled test here. Three obstacles stack up. The raw method does nothing (Proposition 1). The proxy is weakly related to skin type and partly related to the diagnosis. And external performance is limited by a domain gap that no reweighting can close. The informative variants that do change the training distribution either have no detectable effect or, in tabular data with poor proxies, make the gap worse.

The statistical lesson is more general than the application. Machine-learning papers routinely train several seeds and evaluate them on one test set @bouthillier2021. Treating those seeds as independent experiments, whether by Fisher's method, by pooling predictions, or by counting each (seed, patient) pair as a separate patient, can turn training noise into "highly significant" effects. The fact that two arms which _cannot_ differ from the baseline were reported at $p approx 10^(-9)$ shows how easy this error is to make, and how convincing its output looks. The problem is closely related to the variance underestimation that Nadeau and Bengio identified for overlapping training sets @nadeau2003. The fix is to state the estimand first (these patients, or new patients) and then resample the units that estimand generalises over.

*Limitations.* The study uses one source and one target dataset, a ResNet-18 at a fixed 5-epoch budget, and 10 seeds; its power for FST V–VI effects is low (detectable effect ≈ #pct(N.mde_80.fitz_dark.median)). The construct-matched analysis was added after the primary results were seen and is exploratory. The inference method was also changed after the primary results had been computed: the original Fisher-based analysis was replaced once its negative controls failed (§4.4). The change made every result less significant, never more, and both sets of p-values are reported. The hierarchical $t$ correction is conservative; a better-calibrated small-sample method would gain power. The tabular proxies are synthetic corruptions except for the marital-status proxy. HAM10000 has no skin-type labels, so the proxy could only be validated externally. Code, analysis and drafting were carried out with substantial assistance from AI coding agents (Anthropic's Claude and an OpenCode-based agent); the author directed the work, checked the results and is responsible for the content.

= Conclusion
Proxy-guided sampling, as proposed in this project's earlier version, does not improve skin-tone equity in melanoma classification, and the earlier positive results were artefacts of an uncontrolled comparison. The most useful outputs of the re-test are the negative result itself, a quantified warning about how seed-level inference goes wrong, and a calibrated alternative.

*Reproducibility.* All numbers in this paper are generated from `results/**.json` by `src/proxystudy/report/hier_report.py`. The full experiment ledger, protocol audits and every negative result are in the repository. Raw images are not redistributed; `scripts/download_data.sh` fetches them with checksums.

#bibliography("refs.bib", style: "ieee")
