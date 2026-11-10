"""The Phase 12 statistics, against values computed by hand or with exact formulas."""

from __future__ import annotations

import math

import pytest

from experiments.analysis.stats import (
    benjamini_hochberg,
    cluster_bootstrap_diff,
    mcnemar_exact,
    mde_paired,
    wilson,
)


def test_mcnemar_exact_matches_the_binomial() -> None:
    # 2 * P(X <= 1), X ~ Bin(10, 0.5) = 2 * 11/1024
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)


def test_benjamini_hochberg_known_example() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.02, 0.04, 0.04
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert benjamini_hochberg([0.9]) == [0.9]


def test_cluster_bootstrap_centres_on_the_paired_difference() -> None:
    clusters = {f"t{i}": [(1.0, 0.0), (1.0, 1.0)] for i in range(40)}  # +50 pp everywhere
    point, lo, hi = cluster_bootstrap_diff(clusters, resamples=2000, seed=1)
    assert point == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    mixed = {f"t{i}": [(1.0, 0.0)] if i % 2 else [(0.0, 0.0)] for i in range(200)}
    point, lo, hi = cluster_bootstrap_diff(mixed, resamples=4000, seed=1)
    assert point == pytest.approx(0.5) and lo < 0.5 < hi and hi - lo < 0.2
    assert math.isnan(cluster_bootstrap_diff({})[0])


def test_wilson_and_mde() -> None:
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert mde_paired(360, 0.2) == pytest.approx((1.959964 + 0.841621) * math.sqrt(0.2 / 360))
