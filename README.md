# Proxy-Guided Sampling Under Demographic Uncertainty

Statistical reliability, fairness, and applications to medical AI.

Rebuilt, audited, and extended from an earlier (unverifiable) study that claimed
proxy-based stratified sampling improves subgroup performance without access to
protected attributes at training time.

## Research question

Under what statistical conditions can *imperfect* demographic proxies improve
subgroup classification performance, without access to protected attributes
during training — and when do they fail or cause harm?

## Layout

```
src/proxystudy/        core library
  data/                dataset acquisition, validation, splitting, leakage checks
  proxies/             proxy extraction, corruption, quality metrics
  sampling/            sampling & weighting methods (ERM baselines and controls)
  models/              tabular and vision models
  stats/               metrics, bootstrap, hypothesis tests, power
  sims/                controlled simulation experiments
  exp/                 experiment orchestration, ledger, aggregation
  report/              tables and figures from saved results
configs/               experiment configs (YAML)
scripts/               download / run scripts, Kaggle notebook
tests/                 unit + statistical validation tests
results/               experiment outputs (JSON), aggregated artifacts
paper/                 manuscript, figures, references
AUDIT.md               Phase 0 audit of the original study
CHECKLIST.md           persistent progress checklist
LEDGER.md              experiment ledger (append-only)
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev,vision]"
.venv/bin/python -m pytest
```

GPU experiments assume a single consumer GPU (<=16 GB VRAM). Every experiment
command is documented in `CHECKLIST.md`; each result JSON records the code
revision, config hash, dataset checksum, and seed(s).
