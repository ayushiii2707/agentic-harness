"""End-to-end check of the Phase 11 paths against a local Ollama model (free; nothing leaves the machine).

    python scripts/e2e_ollama.py qwen2.5-coder:7b

Streaming with a check, a background check, and one automatic calibration run (local models need no budget).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_harness import AdaptiveHarness, CalibrationConfig, HarnessConfig, HTTPConfig, PersistenceConfig


def main(model: str) -> None:
    db = Path(tempfile.mkdtemp()) / "h.db"
    cfg = HarnessConfig(
        persistence=PersistenceConfig(path=db),
        http=HTTPConfig(timeout_s=400, max_attempts=1),
        calibration=CalibrationConfig(auto=True, max_calls=3, max_tokens=200, categories=("math",)),
    )
    h = AdaptiveHarness(provider="ollama", model=model, config=cfg)
    t = time.time()
    chunks = list(
        h.stream(
            "What is 12 times 12? Reply with the number only.", check=lambda s: "144" in s, max_tokens=20
        )
    )
    text = "".join(c.delta for c in chunks)
    print(f"stream: {len(chunks)} events, text {text.strip()!r}, {time.time() - t:.1f}s")
    r = h.generate(
        "What is 7 plus 8? Reply with the number only.",
        check=lambda s: "15" in s,
        background=True,
        max_tokens=20,
    )
    print(f"background: returned with status {r.trace.evaluation_status!r}, text {r.text.strip()!r}")
    print("waiting for the background check and the automatic calibration ...")
    done = h.flush(1200)
    rep = h.orchestrator.last_report
    print(
        f"flush done={done}; auto-calibration calls={rep.calls if rep else None}, error={h.orchestrator.last_error}"
    )
    assert h.store is not None
    for row in h.store.query(
        "SELECT t.experiment_id IS NOT NULL AS cal, e.status FROM evaluation e JOIN trial t ON t.id=e.trial_id"
    ):
        print("  evaluation:", "calibration" if row["cal"] else "user call", row["status"])
    print("profile status:", h._current_profile()[1])
    h.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:7b")
