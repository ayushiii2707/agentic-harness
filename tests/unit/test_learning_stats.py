"""The exact Beta comparison that replaced Monte-Carlo sampling in the policy."""

from __future__ import annotations

import random

import pytest

from adaptive_harness.learning.stats import betainc, normal_approx, prob_greater


def test_betainc_matches_closed_forms() -> None:
    assert betainc(1, 1, 0.3) == pytest.approx(0.3)
    assert betainc(2, 3, 0.4) == pytest.approx(1 - 0.6**4 - 4 * 0.4 * 0.6**3)
    assert betainc(5, 1, 0.7) == pytest.approx(0.7**5)
    assert betainc(3, 4, 0.0) == 0.0 and betainc(3, 4, 1.0) == 1.0


@pytest.mark.parametrize(
    ("a1", "b1", "a2", "b2", "gap"),
    [
        (10, 10, 10, 10, 0.0),
        (31, 11, 7, 35, 0.02),
        (40, 2, 35, 7, 0.02),
        (300, 100, 280, 120, 0.02),
        (2, 1, 1, 2, 0.1),
    ],
)
def test_prob_greater_agrees_with_monte_carlo(a1: float, b1: float, a2: float, b2: float, gap: float) -> None:
    rng = random.Random(0)
    n = 100_000
    mc = sum(rng.betavariate(a1, b1) > rng.betavariate(a2, b2) + gap for _ in range(n)) / n
    assert prob_greater(a1, b1, a2, b2, gap) == pytest.approx(mc, abs=0.006)  # ~4 MC standard errors


def test_symmetry_determinism_and_screen() -> None:
    assert prob_greater(7, 3, 7, 3) == pytest.approx(0.5, abs=1e-6)
    assert prob_greater(40, 2, 35, 7, 0.02) == prob_greater(40, 2, 35, 7, 0.02)
    # the screen is within the margin the policy allows (0.15) on arms with >= 30 outcomes
    for a1, b1, a2, b2 in [(40, 2, 35, 7), (29, 3, 20, 12), (16, 16, 12, 20)]:
        assert abs(normal_approx(a1, b1, a2, b2, 0.02) - prob_greater(a1, b1, a2, b2, 0.02)) < 0.05
    with pytest.raises(ValueError):
        prob_greater(1, 1, 1, 1, -0.1)
