"""Calibration of seed-combination procedures for paired model comparisons.

Every training seed is evaluated on the same test set. This simulation asks
how often each way of combining seeds rejects a true null (two training
procedures identical in distribution), and how much power it has when one
procedure is genuinely better.

Procedures compared (all two-sided, alpha = 0.05):
  fisher      per-seed paired bootstrap p-values combined with Fisher's method
  pooled      every (seed, item) prediction treated as an independent unit
  seed_t      one-sample t-test on per-seed deltas (ignores test sampling)
  hier_pct    hierarchical bootstrap over items and seeds, percentile p
  hier_t      hierarchical bootstrap, t(S-1) on inflated SE (primary method)

Usage: python scripts/sim_seed_inference.py --reps 500 --out results/sim_seed/sim_seed.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats as st

from proxystudy.stats.hierarchical import cell_weights, hierarchical_paired


def world(rng, n, n_seeds, tau, effect):
    """Shared test set; seeds differ via a random slope on an item feature.

    ``effect`` shifts procedure A's scores toward the correct label, so
    effect = 0 is the null and effect > 0 makes A genuinely better.
    """
    y = (rng.random(n) < 0.3).astype(float)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)

    def fit(shift):
        u = rng.normal(0, tau, size=(n_seeds, 1))
        e = rng.normal(0, 0.3, size=(n_seeds, n))
        return ((1.2 * y + 0.5 * z + u * x + e + shift * (2 * y - 1)) > 0.6).astype(float)

    return y, fit(effect), fit(0.0)


def one_cell(args):
    n, n_seeds, tau, effect, reps, n_boot, seed = args
    rng = np.random.default_rng(seed)
    rej = dict.fromkeys(("fisher", "pooled", "seed_t", "hier_pct", "hier_t"), 0)
    alpha = 0.05
    for r in range(reps):
        y, a, b = world(rng, n, n_seeds, tau, effect)

        def units(g, y=y):
            return cell_weights(y, g)

        h = hierarchical_paired(y, a, b, units, n_boot=n_boot, seed=r)["overall"]
        rej["hier_pct"] += h["p"] < alpha
        rej["hier_t"] += h["p_t"] < alpha
        per_seed_p = [hierarchical_paired(y, a[s:s + 1], b[s:s + 1], units, n_boot=n_boot,
                                          seed=10_000 + r * n_seeds + s,
                                          resample_seeds=False)["overall"]["p"]
                      for s in range(n_seeds)]
        p = np.clip(per_seed_p, 1e-12, 1)
        rej["fisher"] += st.chi2.sf(-2 * np.log(p).sum(), 2 * n_seeds) < alpha
        yt = np.tile(y, n_seeds)
        pooled = hierarchical_paired(yt, a.reshape(1, -1), b.reshape(1, -1),
                                     lambda g, yt=yt: cell_weights(yt, g), n_boot=n_boot,
                                     seed=20_000 + r, resample_seeds=False)["overall"]
        rej["pooled"] += pooled["p"] < alpha
        d = np.asarray(h["per_seed_delta"])
        if np.ptp(d) > 0:
            rej["seed_t"] += st.ttest_1samp(d, 0.0).pvalue < alpha
        else:
            rej["seed_t"] += bool(d[0] != 0)
    rates = {k: v / reps for k, v in rej.items()}
    mc_se = {k: float(np.sqrt(max(v * (1 - v), 1e-12) / reps)) for k, v in rates.items()}
    return {"n": n, "n_seeds": n_seeds, "tau": tau, "effect": effect, "reps": reps,
            "n_boot": n_boot, "rejection_rate": rates, "mc_se": mc_se}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", default="results/sim_seed/sim_seed.json")
    args = ap.parse_args(argv)
    grid = list(product((3, 5, 10, 20), (0.0, 0.3, 0.6, 1.0), (0.0, 0.05, 0.1)))
    jobs = [(args.n, s, t, e, args.reps, args.n_boot, args.seed + i) for i, (s, t, e) in enumerate(grid)]
    with ProcessPoolExecutor(args.workers) as ex:
        cells = list(ex.map(one_cell, jobs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"design": vars(args), "cells": cells}, indent=1))
    for c in cells:
        if c["effect"] == 0.0:
            r = c["rejection_rate"]
            print(f"S={c['n_seeds']:>2} tau={c['tau']:.1f} null: " +
                  "  ".join(f"{k}={v:.3f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
