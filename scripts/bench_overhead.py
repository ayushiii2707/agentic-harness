"""Overhead of the harness itself (DESIGN O5): added latency per call, excluding model time.

    python scripts/bench_overhead.py

Uses a zero-latency mock model and a temporary store, with evidence present so the policy really compares arms.
Targets: p50 < 5 ms per call on the deterministic path; profile lookup < 1 ms.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig, PolicyConfig
from adaptive_harness.providers import MockProvider

PROMPT = "Write a Python function that checks whether a number is prime."


def measure(n: int = 300) -> dict[str, float]:
    db = Path(tempfile.mkdtemp()) / "h.db"
    model = MockProvider(lambda r: "ok")
    warm = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=model,
        config=HarnessConfig(
            persistence=PersistenceConfig(path=db), policy=PolicyConfig(exploration_rate=1.0)
        ),
        policy_seed=0,
    )
    for _ in range(300):  # evidence for every candidate arm, so exploitation runs the full comparison
        warm.generate(PROMPT, check=lambda t: True)
    h = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=model,
        config=HarnessConfig(persistence=PersistenceConfig(path=db)),
    )
    calls, checked, profile = [], [], []
    for _ in range(n):
        t = time.perf_counter()
        h.generate(PROMPT)
        calls.append(time.perf_counter() - t)
        t = time.perf_counter()
        h.generate(PROMPT, check=lambda t: True)
        checked.append(time.perf_counter() - t)
        t = time.perf_counter()
        h._current_profile()
        profile.append(time.perf_counter() - t)
    ms = lambda xs: 1000 * statistics.median(xs)  # noqa: E731
    return {
        "call_p50_ms": ms(calls),
        "checked_call_p50_ms": ms(checked),
        "profile_lookup_p50_ms": ms(profile),
    }


if __name__ == "__main__":
    for k, v in measure().items():
        print(f"{k}: {v:.3f}")
