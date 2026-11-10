"""Frequentist statistics for the pre-registered comparison (DESIGN §6.8), implemented independently (no SciPy).

- ``mcnemar_exact``: two-sided exact McNemar test on the discordant pairs of a paired binary comparison.
- ``benjamini_hochberg``: adjusted p-values (q-values) for a family of tests.
- ``cluster_bootstrap_diff``: percentile CI for a paired difference, resampling clusters (tasks) with all their
  repetitions, because repetitions of one task are not independent.
- ``wilson``: score interval for a single proportion.
- ``mde_paired``: smallest difference detectable at given power, from the observed discordance.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence


def _binom_cdf(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), computed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    log_half_n = n * math.log(0.5)
    for i in range(k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n)
    return min(1.0, total)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. ``b``: pairs where only the first arm succeeded; ``c``: only the second."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * _binom_cdf(min(b, c), n))


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        adjusted[i] = min(1.0, running)
    return adjusted


def cluster_bootstrap_diff(
    clusters: Mapping[str, Sequence[tuple[float, float]]],
    resamples: int = 10_000,
    seed: int = 12,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean paired difference (first - second) and its percentile CI, resampling clusters with replacement.

    ``clusters`` maps a task id to its (first, second) outcome pairs, one per repetition.
    """
    keys = sorted(clusters)
    if not keys:
        return (math.nan, math.nan, math.nan)
    sums = [sum(a - b for a, b in clusters[k]) for k in keys]
    counts = [len(clusters[k]) for k in keys]
    point = sum(sums) / sum(counts)
    rng = random.Random(seed)
    n = len(keys)
    stats = []
    for _ in range(resamples):
        s = c = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            s += sums[j]
            c += counts[j]
        stats.append(s / c)
    stats.sort()
    lo = stats[int((1 - level) / 2 * resamples)]
    hi = stats[min(resamples - 1, int((1 + level) / 2 * resamples) - 1)]
    return point, lo, hi


def wilson(successes: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mde_paired(n_pairs: int, discordance: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Approximate minimum detectable difference for a paired binary test (Connor's normal approximation).

    Uses the observed share of discordant pairs; with none observed, an assumed 20% is used.
    """
    if n_pairs <= 0:
        return math.nan
    pd = discordance if discordance > 0 else 0.20
    z_b = {0.80: 0.841621, 0.90: 1.281552}[power]
    z_a = {0.05: 1.959964, 0.01: 2.575829}[alpha]
    return (z_a + z_b) * math.sqrt(pd / n_pairs)
