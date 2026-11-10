"""Deterministic Beta-posterior comparisons (replaces Monte-Carlo sampling in the policy).

``prob_greater(a1, b1, a2, b2, gap)`` = P(X > Y + gap) for independent X ~ Beta(a1, b1), Y ~ Beta(a2, b2):

    P = ∫ f_Y(y) · (1 - I_{y+gap}(a1, b1)) dy,

with I the regularised incomplete beta function (continued fraction, Numerical Recipes §6.4). The integral uses
composite Simpson's rule on Y's mass (mean ± 9 sd, clipped to [0, 1 - gap]). Its accuracy against exact values
and against 200k-sample Monte Carlo is tested in tests/unit/test_learning_stats.py.
"""

from __future__ import annotations

import math

_EPS = 1e-14
_TINY = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b), i.e. the CDF of Beta(a, b) at x."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _log_pdf(a: float, b: float, y: float, lbeta: float) -> float:
    return (a - 1.0) * math.log(y) + (b - 1.0) * math.log1p(-y) - lbeta


def prob_greater(a1: float, b1: float, a2: float, b2: float, gap: float = 0.0, intervals: int = 120) -> float:
    """P(X > Y + gap) for X ~ Beta(a1, b1), Y ~ Beta(a2, b2), independent."""
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if gap >= 1.0:
        return 0.0
    mean = a2 / (a2 + b2)
    sd = math.sqrt(a2 * b2 / ((a2 + b2) ** 2 * (a2 + b2 + 1.0)))
    lo = max(1e-12, mean - 9.0 * sd)
    hi = min(1.0 - max(gap, 0.0) - 1e-12, mean + 9.0 * sd)
    if hi <= lo:
        return 0.0 if gap > 0 else 1.0
    lbeta = math.lgamma(a2) + math.lgamma(b2) - math.lgamma(a2 + b2)
    n = intervals + (intervals % 2)
    step = (hi - lo) / n
    total = 0.0
    for i in range(n + 1):
        y = lo + i * step
        w = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
        total += w * math.exp(_log_pdf(a2, b2, y, lbeta)) * (1.0 - betainc(a1, b1, y + gap))
    return min(1.0, max(0.0, total * step / 3.0))


def normal_approx(a1: float, b1: float, a2: float, b2: float, gap: float = 0.0) -> float:
    """Cheap normal approximation of ``prob_greater``, used only to skip challengers that are clearly far below the
    threshold (an error there can only keep identity, never cause a switch)."""

    def mv(a: float, b: float) -> tuple[float, float]:
        s = a + b
        return a / s, a * b / (s * s * (s + 1.0))

    m1, v1 = mv(a1, b1)
    m2, v2 = mv(a2, b2)
    z = (m1 - m2 - gap) / math.sqrt(v1 + v2)
    return 0.5 * math.erfc(-z / math.sqrt(2.0))
