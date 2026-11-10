from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from adaptive_harness import (
    AdaptiveHarness,
    ConfigurationError,
    HarnessConfig,
    PersistenceConfig,
    ProviderError,
)
from adaptive_harness.providers import MockProvider


def make(tmp_path: Path, storage: str = "none", **kw: object) -> AdaptiveHarness:
    cfg = HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db", prompt_storage=storage))  # type: ignore[arg-type]
    return AdaptiveHarness(provider="mock", model="mock-1", config=cfg, **kw)  # type: ignore[arg-type]


def test_generate_round_trip_and_trace(tmp_path: Path) -> None:
    h = make(tmp_path)
    r = h.generate("hello")
    assert str(r) == "[mock] hello"
    assert r.trace.strategy_id == "identity" and r.trace.optimized is False
    assert r.trace.reason.startswith("identity:")  # a fresh model has no evidence, so the prompt is unchanged
    assert r.result.simulated is True


def test_prompt_storage_none_keeps_only_hashes(tmp_path: Path) -> None:
    h = make(tmp_path, "none")
    h.generate("my secret plan")
    row = h.store.query_one("SELECT * FROM trial")  # type: ignore[union-attr]
    assert row is not None
    assert row["original_prompt"] is None and row["response_text"] is None
    assert row["prompt_hash"] and row["simulated"] == 1 and row["status"] == "done"


def test_prompt_storage_redacted(tmp_path: Path) -> None:
    h = make(tmp_path, "redacted")
    h.generate("email jane@example.com")
    row = h.store.query_one("SELECT original_prompt FROM trial")  # type: ignore[union-attr]
    assert row is not None and "jane@example.com" not in row["original_prompt"]


def test_chat_accepts_dicts_and_system(tmp_path: Path) -> None:
    mock = MockProvider()
    h = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=mock,
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db")),
    )
    h.chat([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], max_tokens=5)
    req = mock.requests[-1]
    assert [m.role for m in req.messages] == ["system", "user"] and req.settings.max_tokens == 5
    with pytest.raises(TypeError):
        h.generate("x", temprature=0.1)


def test_provider_error_is_recorded_and_raised(tmp_path: Path) -> None:
    mock = MockProvider([ProviderError("down", provider="mock")])
    h = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=mock,
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db")),
    )
    with pytest.raises(ProviderError):
        h.generate("x")
    row = h.store.query_one("SELECT status, error FROM trial")  # type: ignore[union-attr]
    assert row is not None and row["status"] == "failed" and "down" in row["error"]


def test_store_failure_does_not_block_calls(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cfg = HarnessConfig(persistence=PersistenceConfig(path=blocker / "sub" / "h.db"))
    h = AdaptiveHarness(provider="mock", model="m", config=cfg)
    assert h.store is None
    assert str(h.generate("still works")) == "[mock] still works"


def test_async_and_stream(tmp_path: Path) -> None:
    h = make(tmp_path)
    r = asyncio.run(h.agenerate("async"))
    assert r.text == "[mock] async"
    events = list(h.stream("streamed"))
    assert "".join(e.delta for e in events) == "[mock] streamed"
    assert events[-1].result is not None


def test_unknown_provider() -> None:
    with pytest.raises(ConfigurationError):
        AdaptiveHarness(
            provider="nope", model="x", config=HarnessConfig(persistence=PersistenceConfig(enabled=False))
        )
