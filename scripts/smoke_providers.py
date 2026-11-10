"""Live smoke test for provider adapters: one tiny call per selected provider.

Free by default: only local Ollama and free-tier services run unless --allow-paid is given. Every paid call
reserves an upper-bound cost in the persistent budget ledger first (cap: ADAPTIVE_HARNESS_BUDGET_USD), so the
project's hard spending cap holds across runs.

    python scripts/smoke_providers.py                     # free targets only
    python scripts/smoke_providers.py --only groq gemini  # a subset
    python scripts/smoke_providers.py --allow-paid --only anthropic

Prints a JSON summary. Keys never appear in the output (errors are redacted).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_harness import GenerationSettings, HarnessConfig, Message  # noqa: E402
from adaptive_harness._keys import redact  # noqa: E402
from adaptive_harness.budget import BudgetLedger  # noqa: E402
from adaptive_harness.errors import BudgetExceededError, HarnessError  # noqa: E402
from adaptive_harness.providers import create_adapter  # noqa: E402
from adaptive_harness.store import Store  # noqa: E402
from adaptive_harness.types import CompletionRequest  # noqa: E402

PROMPT = "Reply with exactly the word: pong"
MAX_TOKENS = 512  # reasoning models spend output tokens on hidden reasoning before the answer


PER_CALL_CAP_USD = 0.01  # owner's limit for one paid smoke call


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    free: bool
    max_cost_usd: float  # reserved before the call; must be >= the worst case at MAX_TOKENS
    usd_per_m_in: float = 0.0  # list prices, used only when the provider reports no cost
    usd_per_m_out: float = 0.0
    note: str = ""

    def estimate(self, input_tokens: int | None, output_tokens: int | None) -> float:
        return ((input_tokens or 0) * self.usd_per_m_in + (output_tokens or 0) * self.usd_per_m_out) / 1e6


TARGETS = [
    Target("ollama", "qwen2.5-coder:7b", True, 0.0, note="local"),
    # The free hosted targets are two of the three models LeBlanc v4 studied (via OpenRouter there).
    Target("groq", "openai/gpt-oss-20b", True, 0.0, note="free tier"),
    Target("gemini", "gemini-3.5-flash-lite", True, 0.0, note="free tier"),
    # Ollama cloud answered HTTP 402 "add usage credits" on 2026-11-10: not free for this account.
    Target("ollama_cloud", "deepseek-v4.1-flash:1017", False, 0.005, note="needs credits"),
    # Paid list prices (USD per million tokens): gpt-4o-mini 0.15/0.60; claude-haiku-5-5 0.10/0.50;
    # openai/gpt-oss-20b on OpenRouter 0.03/0.13 (from its /models listing, 2026-11-10).
    Target("openai", "gpt-4o-mini", False, 0.005, 0.15, 0.60),
    Target("anthropic", "claude-haiku-5-5", False, 0.005, 0.10, 0.50),
    Target("openrouter", "openai/gpt-oss-20b", False, 0.005, 0.03, 0.13),
]
assert all(t.max_cost_usd <= PER_CALL_CAP_USD for t in TARGETS)


def run_one(t: Target, ledger: BudgetLedger, allow_paid: bool) -> dict[str, object]:
    out: dict[str, object] = {"provider": t.provider, "model": t.model, "free": t.free}
    if not t.free and not allow_paid:
        out["status"] = "skipped (paid; pass --allow-paid)"
        return out
    reservation = None
    try:
        if not t.free:
            reservation = ledger.reserve(t.max_cost_usd, note=f"smoke {t.provider}:{t.model}")
        adapter = create_adapter(t.provider)
        req = CompletionRequest.build(
            t.model, [Message("user", PROMPT)], GenerationSettings(max_tokens=MAX_TOKENS)
        )
        r = adapter.complete(req)
        info = adapter.model_info(t.model)
        reported = r.cost_usd is not None
        actual = r.cost_usd if reported else t.estimate(r.usage.input_tokens, r.usage.output_tokens)
        if reservation is not None:
            ledger.commit(
                reservation, actual, note="smoke" + ("" if reported else " (estimated from list price)")
            )
        out["cost_charged_to_ledger_usd"] = round(actual or 0.0, 8)
        out["cost_source"] = "provider" if reported else ("list price estimate" if not t.free else "free")
        out.update(
            status="ok",
            text=r.text.strip()[:60],
            finish=r.finish_reason.value,
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            reasoning_tokens=r.usage.reasoning_tokens,
            cost_usd=r.cost_usd,
            latency_s=round(r.latency_s, 3),
            model_reported=r.model_reported,
            fingerprint=dict(r.fingerprint),
            dropped=list(r.dropped_settings),
            metadata={f.key: f.value for f in info.facts if f.key not in ("capabilities",)},
        )
        adapter.close()
    except BudgetExceededError as e:
        out.update(status="refused by budget", error=str(e))
    except HarnessError as e:
        if reservation is not None:
            ledger.release(reservation, note="call failed")
        out.update(status="error", error_type=type(e).__name__, error=redact(str(e))[:300])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="provider names to run")
    ap.add_argument("--allow-paid", action="store_true")
    args = ap.parse_args()
    cfg = HarnessConfig.from_env()
    store = Store(ROOT / ".harness" / "dev.db")
    ledger = BudgetLedger(store, total_usd=cfg.budget.total_usd)
    targets = [t for t in TARGETS if not args.only or t.provider in args.only]
    results = []
    for t in targets:
        results.append(run_one(t, ledger, args.allow_paid))
        time.sleep(0.5)
    summary = {
        "results": results,
        "budget": {
            "cap_usd": ledger.total_usd,
            "spent_usd": round(ledger.spent_usd(), 6),
            "remaining_usd": round(ledger.remaining_usd(), 6),
        },
    }
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
