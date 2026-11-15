"""Small-sample statistics for multi-seed results.

Three seeds reported as mean and population standard deviation cannot carry a
pass/fail claim. These helpers report what a reader needs to judge one: the
sample standard deviation, the median, the interquartile mean (robust to one
outlier seed), and a percentile bootstrap interval for the mean.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from numpy.typing import ArrayLike

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
# Fixed so the same per-seed numbers always produce the same interval.
BOOTSTRAP_SEED = 20261114


def interquartile_mean(values: ArrayLike) -> float:
    """Mean of the middle 50% of values, with fractional end weights.

    Matches the IQM used by rliable: the sorted sample is trimmed by 25% at
    each end, weighting partially-included points by the fraction kept.
    """
    data = np.sort(np.asarray(values, dtype=np.float64))
    n = len(data)
    if n == 0:
        raise ValueError("interquartile_mean needs at least one value")
    lower, upper = 0.25 * n, 0.75 * n
    weights = np.clip(
        np.minimum(np.arange(1, n + 1), upper) - np.maximum(np.arange(n), lower),
        0.0,
        1.0,
    )
    return float(np.dot(weights, data) / weights.sum())


def bootstrap_mean_interval(
    values: ArrayLike,
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    With very few seeds this interval is too narrow (it cannot exceed the
    observed range), which is a reason to run more seeds, not to trust it.
    """
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        raise ValueError("bootstrap_mean_interval needs at least one value")
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(resamples, len(data)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def wilson_interval(
    successes: float, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a rate such as a fall rate.

    A 20-episode fall rate near 50% has an interval of roughly +/-20 points;
    this makes that visible next to the point estimate.
    """
    if trials <= 0:
        raise ValueError("wilson_interval needs at least one trial")
    z = {0.90: 1.6448536, 0.95: 1.959964, 0.99: 2.5758293}.get(round(confidence, 2))
    if z is None:
        raise ValueError("wilson_interval supports confidence 0.90, 0.95 or 0.99")
    rate = successes / trials
    denom = 1.0 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denom
    half = (
        z
        * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
        / denom
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(values: ArrayLike) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    low, high = bootstrap_mean_interval(data)
    return {
        "n": float(len(data)),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "median": float(np.median(data)),
        "iqm": interquartile_mean(data),
        "ci95_low": low,
        "ci95_high": high,
        "min": float(data.min()),
        "max": float(data.max()),
    }


def summarize_metrics(
    per_seed: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Summarise every finite numeric metric present in all seeds."""
    rows = list(per_seed)
    if not rows:
        return {}
    shared = set(rows[0])
    for row in rows[1:]:
        shared &= set(row)
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(shared):
        values = [row[key] for row in rows]
        if all(
            not isinstance(value, bool)
            and isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value))
            for value in values
        ):
            summary[key] = summarize([float(value) for value in values])
    return summary
