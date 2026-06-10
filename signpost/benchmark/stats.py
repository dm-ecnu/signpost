from __future__ import annotations

"""Small statistics helpers for experiment metric aggregation."""

import random
from math import ceil
from statistics import mean, median
from typing import Any, Callable, Iterable


def to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort numeric conversion used for heterogeneous JSON logs."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_values(values: Iterable[Any]) -> dict[str, float]:
    """Return sum/mean/median/p90/p95/min/max/count for a numeric series."""

    nums = [to_float(value) for value in values if value is not None]
    if not nums:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(nums)
    return {
        "count": len(ordered),
        "sum": float(sum(ordered)),
        "mean": float(mean(ordered)),
        "median": float(median(ordered)),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile for small experiment samples."""

    if not sorted_values:
        return 0.0
    rank = max(1, ceil((pct / 100.0) * len(sorted_values)))
    return float(sorted_values[min(rank - 1, len(sorted_values) - 1)])


def safe_div(numerator: float, denominator: float) -> float | None:
    """Return None for undefined ratios instead of emitting misleading zeros."""

    if denominator == 0:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# Bootstrap inference
# ---------------------------------------------------------------------------

_DEFAULT_SEED = 20240101


def bootstrap_ci(
    values: Iterable[Any],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    statistic: Callable[[list[float]], float] = mean,
    seed: int = _DEFAULT_SEED,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for a univariate statistic.

    Parameters
    ----------
    values:
        Raw per-question scores.  ``None`` entries are dropped.
    n_boot:
        Number of bootstrap resamples.
    alpha:
        Two-tailed error level; returns a ``1 - alpha`` CI.
    statistic:
        Any callable that accepts a ``list[float]`` and returns a ``float``.
        Defaults to ``statistics.mean``.
    seed:
        Fixed seed for reproducibility.  Pass a different integer to obtain
        independent replicates.

    Returns
    -------
    dict with keys ``mean``, ``lo``, ``hi``.  ``mean`` is the statistic applied
    to the original (non-resampled) data; ``lo``/``hi`` are the lower and upper
    percentile-bootstrap bounds.
    """

    nums = [to_float(v) for v in values if v is not None]
    if not nums:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}

    point_estimate = statistic(nums)
    n = len(nums)
    rng = random.Random(seed)
    boot_stats: list[float] = []
    for _ in range(n_boot):
        resample = [rng.choice(nums) for _ in range(n)]
        boot_stats.append(statistic(resample))

    boot_stats.sort()
    lo_idx = max(0, int(ceil(n_boot * (alpha / 2.0))) - 1)
    hi_idx = min(n_boot - 1, int(n_boot * (1.0 - alpha / 2.0)) - 1)
    return {
        "mean": point_estimate,
        "lo": boot_stats[lo_idx],
        "hi": boot_stats[hi_idx],
    }


def paired_bootstrap_diff(
    values_a: Iterable[Any],
    values_b: Iterable[Any],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = _DEFAULT_SEED,
) -> dict[str, float | int]:
    """Paired percentile bootstrap for the mean difference (A - B).

    The pairing is positional: element ``i`` of *values_a* is compared to
    element ``i`` of *values_b*.  Pairs where either value is ``None`` are
    dropped.

    Returns
    -------
    dict with keys:
    - ``mean_diff`` — point estimate ``mean(A) - mean(B)``
    - ``lo``, ``hi`` — ``1 - alpha`` percentile-bootstrap CI of the difference
    - ``p_value`` — two-sided bootstrap p-value for H₀: mean(A) = mean(B)
    - ``n_pairs`` — number of non-null pairs used
    """

    pairs = [
        (to_float(a), to_float(b))
        for a, b in zip(values_a, values_b)
        if a is not None and b is not None
    ]
    if not pairs:
        return {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0, "n_pairs": 0}

    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    n = len(pairs)
    point_diff = mean(a_vals) - mean(b_vals)

    rng = random.Random(seed)
    boot_diffs: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_a = mean([a_vals[i] for i in idx])
        boot_b = mean([b_vals[i] for i in idx])
        boot_diffs.append(boot_a - boot_b)

    boot_diffs.sort()
    lo_idx = max(0, int(ceil(n_boot * (alpha / 2.0))) - 1)
    hi_idx = min(n_boot - 1, int(n_boot * (1.0 - alpha / 2.0)) - 1)

    # Two-sided bootstrap p-value: proportion of resamples at least as extreme
    # as the observed difference under the null (shift each resample toward 0).
    abs_point = abs(point_diff)
    extreme = sum(1 for d in boot_diffs if abs(d - point_diff) >= abs_point)
    p_value = extreme / n_boot

    return {
        "mean_diff": point_diff,
        "lo": boot_diffs[lo_idx],
        "hi": boot_diffs[hi_idx],
        "p_value": p_value,
        "n_pairs": n,
    }


def summarize_with_ci(
    values: Iterable[Any],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = _DEFAULT_SEED,
) -> dict[str, float]:
    """Extend ``summarize_values`` with percentile bootstrap CI for the mean.

    Returns all keys from :func:`summarize_values` plus ``ci_lo`` and
    ``ci_hi`` (the ``1 - alpha`` bootstrap CI bounds on the mean).
    """

    nums_list = [v for v in values if v is not None]
    base = summarize_values(nums_list)
    ci = bootstrap_ci(nums_list, n_boot=n_boot, alpha=alpha, seed=seed)
    return {**base, "ci_lo": ci["lo"], "ci_hi": ci["hi"]}
