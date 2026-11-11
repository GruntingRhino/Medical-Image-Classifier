"""Validation of core metrics against scikit-learn on randomized data."""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from proxystudy.stats.metrics import (
    auroc,
    balanced_accuracy,
    subgroup_metrics,
    to_dict,
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_balanced_accuracy_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=500)
    p = rng.integers(0, 2, size=500)
    assert abs(balanced_accuracy(y, p) - balanced_accuracy_score(y, p)) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_auroc_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=800)
    s = rng.normal(size=800) + 0.3 * y
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-10


def test_auroc_handles_ties():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.7, 0.2, 0.9])
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_subgroup_metrics_and_sbag():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, size=1000)
    y = rng.integers(0, 2, size=1000)
    # group 1 perfectly predicted
    pred = np.where(g == 1, y, 0)
    sm = subgroup_metrics(y, pred, g)
    assert sm.group_bacc[1] == 1.0
    assert sm.worst_group_bacc == sm.group_bacc[0]
    assert abs(sm.sbag - (sm.group_bacc[1] - sm.group_bacc[0])) < 1e-12


def test_undefined_group_flagged():
    # group 2 has only positives -> specificity undefined
    y = np.array([1, 0, 1, 1])
    pred = np.array([1, 0, 1, 0])
    g = np.array([0, 0, 2, 2])
    sm = subgroup_metrics(y, pred, g)
    assert 2 in sm.undefined_groups
    assert np.isnan(sm.group_bacc[2])
    # worst-group must ignore the undefined group
    assert sm.worst_group_bacc == sm.group_bacc[0]
    d = to_dict(sm)
    assert d["group_bacc"]["2"] is None


def test_subgroup_with_scores():
    rng = np.random.default_rng(7)
    g = rng.integers(0, 2, size=600)
    y = rng.integers(0, 2, size=600)
    s = rng.normal(size=600) + 0.5 * y
    sm = subgroup_metrics(y, (s > 0.5).astype(int), g, s)
    for val in sm.group_auroc.values():
        assert not np.isnan(val)
        assert 0.0 <= val <= 1.0
