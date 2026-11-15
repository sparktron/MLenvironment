import numpy as np
import pytest

from rl_framework.core.training.statistics import (
    bootstrap_mean_interval,
    interquartile_mean,
    summarize_metrics,
    wilson_interval,
)


def test_interquartile_mean_trims_outliers() -> None:
    assert interquartile_mean([1, 2, 3, 4]) == pytest.approx(2.5)
    assert interquartile_mean([0, 10, 10, 10, 1000]) == pytest.approx(10.0)
    # Fractional weights: n=5 keeps 1.25..3.75 of the sorted positions.
    assert interquartile_mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)


def test_bootstrap_interval_is_deterministic_and_brackets_mean() -> None:
    values = [1.0, 2.0, 4.0, 8.0, 3.0]
    first = bootstrap_mean_interval(values)
    assert first == bootstrap_mean_interval(values)
    assert first[0] < np.mean(values) < first[1]
    assert bootstrap_mean_interval([5.0]) == (5.0, 5.0)


def test_wilson_interval_shows_twenty_episode_uncertainty() -> None:
    low, high = wilson_interval(10, 20)
    assert low < 0.3 and high > 0.7
    low, high = wilson_interval(0, 100)
    assert low == 0.0 and high < 0.04


def test_summarize_metrics_skips_non_numeric_and_missing_keys() -> None:
    summary = summarize_metrics(
        [{"a": 1.0, "b": "x", "c": 1.0}, {"a": 3.0, "b": "y"}]
    )
    assert set(summary) == {"a"}
    assert summary["a"]["mean"] == 2.0
