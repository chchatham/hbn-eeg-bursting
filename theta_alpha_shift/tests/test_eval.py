"""Tests for evaluation module."""

import numpy as np
import pytest

from theta_alpha_shift.eval import EvalResult
from theta_alpha_shift.eval.confusion import (
    run_evaluation_grid,
    compute_confusion,
    results_to_array,
)
from theta_alpha_shift.eval.discrimination import (
    _roc_auc,
    compute_auc,
    compute_bias_variance,
    summarize_discrimination,
)


class TestEvalResult:
    def test_dataclass_fields(self):
        r = EvalResult(
            method_name="test",
            regime="chirp",
            age=8,
            trial=0,
            headline_stat=1.0,
            headline_stat_name="n_peaks",
        )
        assert r.method_name == "test"
        assert r.regime == "chirp"
        assert r.metadata == {}


class TestROCAUC:
    def test_perfect_separation(self):
        auc = _roc_auc([10, 11, 12], [1, 2, 3])
        assert auc == 1.0

    def test_no_separation(self):
        auc = _roc_auc([5, 5, 5], [5, 5, 5])
        assert auc == 0.5

    def test_partial_separation(self):
        auc = _roc_auc([6, 7, 8], [4, 5, 6])
        assert 0.5 < auc <= 1.0

    def test_empty_input(self):
        auc = _roc_auc([], [1, 2])
        assert np.isnan(auc)

    def test_always_above_half(self):
        auc = _roc_auc([1, 2, 3], [10, 11, 12])
        assert auc >= 0.5


class TestEvalGrid:
    def test_small_grid(self):
        results = run_evaluation_grid(
            method_names=["threshold_bursts"],
            ages=[8],
            n_trials=1,
            verbose=False,
        )
        assert len(results) == 5
        for r in results:
            assert isinstance(r, EvalResult)
            assert r.method_name == "threshold_bursts"
            assert r.age == 8

    def test_confusion_matrix_shape(self):
        results = run_evaluation_grid(
            method_names=["threshold_bursts"],
            ages=[8, 11],
            n_trials=2,
            verbose=False,
        )
        matrix, std_matrix, regimes, ages = compute_confusion(results, "threshold_bursts")
        assert matrix.shape == (5, 2)
        assert len(regimes) == 5
        assert ages == [8, 11]


class TestDiscrimination:
    def test_compute_auc(self):
        results = run_evaluation_grid(
            method_names=["threshold_bursts"],
            ages=[8],
            n_trials=3,
            verbose=False,
        )
        auc_overall, auc_per_age = compute_auc(results, "threshold_bursts")
        assert 0.0 <= auc_overall <= 1.0
        assert 8 in auc_per_age

    def test_bias_variance(self):
        results = run_evaluation_grid(
            method_names=["threshold_bursts"],
            ages=[8, 11],
            n_trials=3,
            verbose=False,
        )
        metrics = compute_bias_variance(results, "threshold_bursts")
        assert "chirp" in metrics
        assert "mixture" in metrics
        assert "mean_trajectory" in metrics["chirp"]
        assert len(metrics["chirp"]["mean_trajectory"]) == 2

    def test_summarize(self):
        results = run_evaluation_grid(
            method_names=["threshold_bursts"],
            ages=[8],
            n_trials=2,
            verbose=False,
        )
        summary = summarize_discrimination(results)
        assert len(summary) == 1
        assert summary[0]["method"] == "threshold_bursts"
        assert "auc_overall" in summary[0]
