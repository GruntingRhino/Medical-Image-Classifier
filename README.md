# Proxy-Guided Sampling and Skin-Tone Equity in Melanoma Classification

**Paper:** [`paper/paper.pdf`](paper/paper.pdf) — *Proxy-Guided Sampling Does Not Improve Skin-Tone
Equity in Melanoma Classification: a re-registered audit, and why combining training seeds with
Fisher's method fails.* Abhay Sivaram, preprint (not peer reviewed), November 2026.

> **Correction notice.** An earlier version of this project (PDF dated 2026-11-07) reported
> +5.6 balanced-accuracy points for melanoma detection and a 27% smaller gender accuracy gap from
> "P-Stratified Sampling". Those results did not come from controlled comparisons and are
> **withdrawn**. [`AUDIT.md`](AUDIT.md) documents why, and this repository contains the re-test.

## Findings

1. **The original algorithm is a no-op.** Equalising quantile bins of a continuous proxy leaves the
   training set unchanged (paper, Proposition 1).
2. **No proxy-based method detectably improved accuracy** for melanoma classification trained on
   HAM10000 and evaluated externally on Fitzpatrick17k, overall or for Fitzpatrick skin types V–VI.
   This holds across 7 methods, 3 proxy-corruption levels and 10 seeds each. The detectable effect
   was about 2.6 points overall and 6.3 points for types V–VI.
3. **External transfer fails** (AUROC ≈ 0.57), mainly because of the dermoscopy → clinical-photo
   domain gap. Fitzpatrick17k has only 38 melanoma images of skin types V–VI.
4. **On tabular data, proxy methods can widen the gap.** Only 3 of 106 primary tests survive
   family-wise correction, and all three are harms.
5. **Statistics:** combining per-seed p-values with Fisher's method is invalid when every seed is
   scored on the same test set. In simulation it rejected a true null up to 100% of the time; on
   real data it flagged arms identical to the baseline at p ≈ 10⁻⁹.
   [`stats/hierarchical.py`](src/proxystudy/stats/hierarchical.py) implements a hierarchical
   (patient × seed) paired bootstrap that stays at or below nominal error.

## Layout

```
src/proxystudy/   data loaders + audits, proxies, sampling methods, models,
                  stats (metrics, bootstrap, hierarchical inference), experiments, report
scripts/          data download, seed-inference calibration simulation
tests/            unit and statistical-validity tests
results/          every experiment record (JSON) and summaries; ledger.jsonl
paper/            paper.typ (source), paper.pdf, generated tables and figures
AUDIT.md          audit of the original claims
CHECKLIST.md      protocol, phase status and reproduction commands
LEDGER.md         dated log of every experiment, incident and correction
```

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev,vision]"
.venv/bin/python -m pytest
scripts/download_data.sh          # fetches datasets and checks checksums (see below)
```

All experiment, analysis and paper-build commands are listed in `CHECKLIST.md`. Each result
record stores the code revision, config hash, dataset checksums and seed. Images, model
checkpoints and per-image predictions are not included (about 30 GB); their checksums are in
`results/manifests/`. GPU runs assume one consumer GPU with ≤ 16 GB.

## Data

Datasets are not redistributed. Download them from their sources under their licences:
HAM10000 (Tschandl et al. 2018, CC BY-NC-SA 4.0), Fitzpatrick17k (Groh et al. 2021, CC BY-NC-SA
3.0; quality-controlled re-host by Abhishek et al. 2024), and UCI Adult and German credit. Skin type
is used only for evaluation, never for training.

> **Authorship & AI-use.** I designed and built this project. Anthropic's Claude (with an
> OpenCode-based agent) was used as a coding and drafting assistant under my direction — to
> scaffold code, run analyses, and draft prose. All research, design, and methodological
> decisions are mine, and I reviewed and stand behind the code and results. All figures are
> derived from public data by this code. Research only; not a medical device.

## License

Code: MIT (see `LICENSE`). Datasets keep their own licences.
