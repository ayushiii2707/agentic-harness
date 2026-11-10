"""Run calibration against a real provider and write the report.

    python scripts/calibrate.py groq openai/gpt-oss-20b --suites math --max-calls 40 --pause 2.5
    python scripts/calibrate.py groq openai/gpt-oss-20b --suites math --max-calls 40 --resume <experiment id>

Paid providers: pass --max-cost-per-call (USD, reserved before each call) and set ADAPTIVE_HARNESS_BUDGET_USD.
Free tiers and local models: --max-cost-per-call 0 (the default).
--llm-classifier PROVIDER:MODEL labels prompts the lexicon classifier calls `general` (use the same setting when
serving, so evidence is looked up under the same labels).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_harness import AdaptiveHarness, ClassifierConfig, HarnessConfig, PersistenceConfig  # noqa: E402
from adaptive_harness.calibration import Calibrator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("provider")
    ap.add_argument("model")
    ap.add_argument("--suites", nargs="+", default=["math"])
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--max-cost-per-call", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--db", default=str(ROOT / ".harness" / "calibration.db"))
    ap.add_argument("--resume")
    ap.add_argument("--llm-classifier", help="PROVIDER:MODEL for the opt-in LLM fallback classifier")
    a = ap.parse_args()
    clf = None
    if a.llm_classifier:
        provider, _, model = a.llm_classifier.partition(":")
        clf = ClassifierConfig(provider, model)
    cfg = HarnessConfig.from_env(
        HarnessConfig(persistence=PersistenceConfig(path=Path(a.db)), llm_classifier=clf)
    )
    h = AdaptiveHarness(provider=a.provider, model=a.model, config=cfg)
    cal = Calibrator(
        h,
        max_calls=a.max_calls,
        max_cost_per_call_usd=a.max_cost_per_call,
        max_tokens=a.max_tokens,
        pause_s=a.pause,
    )
    report = cal.run(tuple(a.suites), experiment_id=a.resume)
    text = report.to_markdown()
    out = ROOT / "docs" / "reports" / f"calibration_{report.experiment_id}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
