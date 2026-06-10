"""Tests for bootstrap CI and significance helpers in signpost.benchmark.stats."""

from __future__ import annotations

import math
from statistics import mean

import pytest

from signpost.benchmark.stats import (
    bootstrap_ci,
    paired_bootstrap_diff,
    summarize_with_ci,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """bootstrap_ci(values, ...) -> {mean, lo, hi}"""

    def test_ci_brackets_mean(self):
        """lo <= mean <= hi for a typical sample."""
        values = [0.4, 0.6, 0.5, 0.7, 0.3, 0.8, 0.55, 0.45, 0.65, 0.35]
        result = bootstrap_ci(values)
        assert result["lo"] <= result["mean"] <= result["hi"]

    def test_mean_equals_sample_mean(self):
        """The 'mean' key must equal the actual sample mean."""
        values = [0.1, 0.4, 0.9, 0.6, 0.5]
        result = bootstrap_ci(values)
        assert math.isclose(result["mean"], mean(values), rel_tol=1e-9)

    def test_ci_narrows_as_n_grows(self):
        """CI width should decrease as the sample grows (law of large numbers)."""
        small = [0.5 + (i % 3) * 0.1 for i in range(10)]
        large = small * 20  # 200 elements, same distribution
        ci_small = bootstrap_ci(small, seed=42)
        ci_large = bootstrap_ci(large, seed=42)
        width_small = ci_small["hi"] - ci_small["lo"]
        width_large = ci_large["hi"] - ci_large["lo"]
        assert width_large < width_small, (
            f"Expected CI to narrow: small={width_small:.4f}, large={width_large:.4f}"
        )

    def test_deterministic_under_fixed_seed(self):
        """Two calls with the same seed must return identical results."""
        values = list(range(50))
        r1 = bootstrap_ci(values, seed=99)
        r2 = bootstrap_ci(values, seed=99)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        """Different seeds should (almost always) produce different CI bounds."""
        values = [float(i) for i in range(30)]
        r1 = bootstrap_ci(values, seed=1)
        r2 = bootstrap_ci(values, seed=2)
        # Means must be identical (point estimate is not stochastic)
        assert r1["mean"] == r2["mean"]
        # Bounds will differ with overwhelming probability for n=30
        assert r1["lo"] != r2["lo"] or r1["hi"] != r2["hi"]

    def test_empty_input_returns_zeros(self):
        result = bootstrap_ci([])
        assert result == {"mean": 0.0, "lo": 0.0, "hi": 0.0}

    def test_all_none_input_returns_zeros(self):
        result = bootstrap_ci([None, None])
        assert result == {"mean": 0.0, "lo": 0.0, "hi": 0.0}

    def test_single_value(self):
        """CI for a single repeated value should collapse to that value."""
        result = bootstrap_ci([0.75])
        assert math.isclose(result["mean"], 0.75)
        assert math.isclose(result["lo"], 0.75)
        assert math.isclose(result["hi"], 0.75)

    def test_constant_series(self):
        """All-same values: CI must collapse to that constant."""
        values = [0.9] * 50
        result = bootstrap_ci(values)
        assert math.isclose(result["mean"], 0.9)
        assert math.isclose(result["lo"], 0.9)
        assert math.isclose(result["hi"], 0.9)

    def test_custom_statistic(self):
        """statistic= kwarg is respected (e.g. max)."""
        values = [0.1, 0.5, 0.9]
        result = bootstrap_ci(values, statistic=max)
        # Point estimate should be max of the original sample
        assert math.isclose(result["mean"], 0.9)
        # lo/hi must bracket the point estimate
        assert result["lo"] <= result["mean"] <= result["hi"]

    def test_alpha_widens_with_smaller_alpha(self):
        """alpha=0.01 should give a wider CI than alpha=0.10."""
        values = [float(i) / 100 for i in range(100)]
        ci_narrow = bootstrap_ci(values, alpha=0.10, seed=7)
        ci_wide = bootstrap_ci(values, alpha=0.01, seed=7)
        width_narrow = ci_narrow["hi"] - ci_narrow["lo"]
        width_wide = ci_wide["hi"] - ci_wide["lo"]
        assert width_wide >= width_narrow


# ---------------------------------------------------------------------------
# paired_bootstrap_diff
# ---------------------------------------------------------------------------


class TestPairedBootstrapDiff:
    """paired_bootstrap_diff(a, b, ...) -> {mean_diff, lo, hi, p_value, n_pairs}"""

    def test_clearly_better_a_small_pvalue_and_ci_excludes_zero(self):
        """When A is clearly better than B, p-value should be small and CI > 0."""
        a = [0.9] * 100
        b = [0.6] * 100
        result = paired_bootstrap_diff(a, b)
        assert result["p_value"] < 0.01, f"p_value={result['p_value']}"
        assert result["lo"] > 0.0, f"CI lo={result['lo']} should exclude 0"
        assert result["hi"] > 0.0

    def test_identical_series_ci_spans_zero(self):
        """When A == B, CI should span 0 and p-value should be large."""
        values = [float(i % 5) / 5 for i in range(80)]
        result = paired_bootstrap_diff(values, values)
        assert result["lo"] <= 0.0 <= result["hi"], (
            f"CI [{result['lo']:.4f}, {result['hi']:.4f}] should span 0 for A==B"
        )
        assert result["p_value"] > 0.10, f"p_value={result['p_value']} should be large for A==B"

    def test_mean_diff_is_correct(self):
        """mean_diff should equal mean(A) - mean(B)."""
        a = [0.7, 0.8, 0.6, 0.9]
        b = [0.5, 0.6, 0.4, 0.7]
        result = paired_bootstrap_diff(a, b)
        expected = mean(a) - mean(b)
        assert math.isclose(result["mean_diff"], expected, rel_tol=1e-9)

    def test_n_pairs_counts_correctly(self):
        """n_pairs should equal len(non-null pairs)."""
        a = [0.5, None, 0.7, 0.8]
        b = [0.4, 0.6, None, 0.7]
        result = paired_bootstrap_diff(a, b)
        # Only positions 0 and 3 are non-null in both
        assert result["n_pairs"] == 2

    def test_ci_brackets_mean_diff(self):
        """lo <= mean_diff <= hi (allowing float rounding at the boundary)."""
        a = [0.65 + (i % 4) * 0.05 for i in range(40)]
        b = [0.50 + (i % 4) * 0.05 for i in range(40)]
        result = paired_bootstrap_diff(a, b)
        tol = 1e-12
        assert result["lo"] <= result["mean_diff"] + tol
        assert result["hi"] >= result["mean_diff"] - tol

    def test_deterministic_under_fixed_seed(self):
        """Two calls with the same seed must return identical results."""
        a = [float(i) / 20 for i in range(20)]
        b = [float(i) / 25 for i in range(20)]
        r1 = paired_bootstrap_diff(a, b, seed=42)
        r2 = paired_bootstrap_diff(a, b, seed=42)
        assert r1 == r2

    def test_empty_input(self):
        result = paired_bootstrap_diff([], [])
        assert result["n_pairs"] == 0
        assert result["p_value"] == 1.0

    def test_symmetry_of_direction(self):
        """Swapping A and B should negate mean_diff."""
        a = [0.8, 0.7, 0.9, 0.75]
        b = [0.5, 0.4, 0.6, 0.55]
        r_ab = paired_bootstrap_diff(a, b, seed=5)
        r_ba = paired_bootstrap_diff(b, a, seed=5)
        assert math.isclose(r_ab["mean_diff"], -r_ba["mean_diff"], rel_tol=1e-9)

    def test_small_noise_does_not_reject_null(self):
        """Tiny, noisy differences should not produce small p-values."""
        import random as _random
        rng = _random.Random(2025)
        a = [0.5 + rng.gauss(0, 0.002) for _ in range(50)]
        b = [0.5 + rng.gauss(0, 0.002) for _ in range(50)]
        result = paired_bootstrap_diff(a, b, seed=3)
        # With near-zero signal, p-value should be > 0.05 most of the time
        # (this is probabilistic; fixed seed makes it deterministic here)
        assert result["p_value"] > 0.05, f"Unexpected rejection: p={result['p_value']}"


# ---------------------------------------------------------------------------
# summarize_with_ci
# ---------------------------------------------------------------------------


class TestSummarizeWithCI:
    """summarize_with_ci extends summarize_values with ci_lo / ci_hi."""

    def test_has_all_base_keys(self):
        """All keys from summarize_values must be present."""
        base_keys = {"count", "sum", "mean", "median", "p90", "p95", "min", "max"}
        result = summarize_with_ci([0.3, 0.5, 0.7])
        assert base_keys.issubset(result.keys())

    def test_has_ci_keys(self):
        result = summarize_with_ci([0.3, 0.5, 0.7])
        assert "ci_lo" in result
        assert "ci_hi" in result

    def test_ci_brackets_mean(self):
        values = [0.4, 0.6, 0.5, 0.7, 0.3, 0.8]
        result = summarize_with_ci(values)
        assert result["ci_lo"] <= result["mean"] <= result["ci_hi"]

    def test_empty_input(self):
        result = summarize_with_ci([])
        assert result["ci_lo"] == 0.0
        assert result["ci_hi"] == 0.0

    def test_deterministic_under_fixed_seed(self):
        values = [float(i) / 10 for i in range(20)]
        r1 = summarize_with_ci(values, seed=77)
        r2 = summarize_with_ci(values, seed=77)
        assert r1 == r2
