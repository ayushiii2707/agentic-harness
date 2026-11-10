"""Tune the drift threshold: false alarms over 500 stationary outcomes, and detection of a real shift.

    python scripts/simulate_drift.py

Simulated Bernoulli outcomes only; this measures the test, not any model.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_harness.learning.drift import check_drift

WINDOW, REFERENCE, RUNS, N = 30, 120, 400, 500


def first_alarm(outcomes: list[float], z: float) -> int | None:
    for i in range(1, len(outcomes) + 1):
        if check_drift(outcomes[:i], window=WINDOW, reference=REFERENCE, z_crit=z).drifted:
            return i
    return None


def main() -> None:
    rng = random.Random(0)
    for z in (3.0, 3.5, 4.0):
        false = 0
        for p in (0.2, 0.5, 0.8):
            for _ in range(RUNS // 3):
                xs = [float(rng.random() < p) for _ in range(N)]
                false += first_alarm(xs, z) is not None
        detected, delays = 0, []
        for _ in range(RUNS // 4):
            xs = [float(rng.random() < 0.8) for _ in range(250)] + [
                float(rng.random() < 0.4) for _ in range(250)
            ]
            hit = first_alarm(xs, z)
            if hit is not None and hit > 250:
                detected += 1
                delays.append(hit - 250)
        delay = sorted(delays)[len(delays) // 2] if delays else None
        print(
            f"z_crit={z}: false alarms {false}/{3 * (RUNS // 3)} over {N} stationary outcomes; "
            f"0.8->0.4 shift detected {detected}/{RUNS // 4}, median delay {delay}"
        )


if __name__ == "__main__":
    main()
