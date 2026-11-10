from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness import ConfigurationError, HarnessConfig, Message
from adaptive_harness.hashing import canonical_hash, text_hash
from adaptive_harness.redaction import redact_text
from adaptive_harness.types import CompletionRequest, Usage


def test_defaults_are_safe() -> None:
    cfg = HarnessConfig()
    assert cfg.budget.total_usd == 0.0
    assert cfg.persistence.prompt_storage == "none"
    assert cfg.optimize is True


def test_from_env_reads_budget_and_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADAPTIVE_HARNESS_BUDGET_USD", "10")
    monkeypatch.setenv("ADAPTIVE_HARNESS_PROMPT_STORAGE", "redacted")
    cfg = HarnessConfig.from_env()
    assert cfg.budget.total_usd == 10.0 and cfg.persistence.prompt_storage == "redacted"
    monkeypatch.setenv("ADAPTIVE_HARNESS_BUDGET_USD", "lots")
    with pytest.raises(ConfigurationError):
        HarnessConfig.from_env()


def test_from_toml_and_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "adaptive_harness.toml"
    p.write_text('optimize = false\n[budget]\ntotal_usd = 2.5\n[persistence]\nprompt_storage = "full"\n')
    cfg = HarnessConfig.from_toml(p)
    assert cfg.optimize is False and cfg.budget.total_usd == 2.5 and cfg.persistence.prompt_storage == "full"
    p.write_text("[budget]\ntotal = 1\n")
    with pytest.raises(ConfigurationError):
        HarnessConfig.from_toml(p)
    with pytest.raises(ConfigurationError):
        HarnessConfig.from_dict({"nope": 1})


def test_invalid_values_rejected() -> None:
    with pytest.raises(ConfigurationError):
        HarnessConfig.from_dict({"budget": {"total_usd": -1}})
    with pytest.raises(ConfigurationError):
        HarnessConfig.from_dict({"persistence": {"prompt_storage": "everything"}})


def test_canonical_hash_ignores_key_order_and_line_endings() -> None:
    assert canonical_hash({"a": 1, "b": [1, 2]}) == canonical_hash({"b": [1, 2], "a": 1})
    assert text_hash("line1\r\nline2\r\n") == text_hash("line1\nline2\n")


def test_message_and_request_validation() -> None:
    with pytest.raises(ValueError):
        Message("tool", "x")  # type: ignore[arg-type]
    req = CompletionRequest.build("m", [{"role": "user", "content": "hi"}])
    assert req.messages == (Message("user", "hi"),)
    with pytest.raises(ValueError):
        CompletionRequest.build("m", [])
    assert Usage().total_tokens is None and Usage(3, 4).total_tokens == 7


def test_redaction_masks_personal_data() -> None:
    out = redact_text(
        "mail me at jane.doe@example.com from 10.0.0.1, card 4111 1111 1111 1111, password: hunter2"
    )
    assert "jane.doe" not in out and "10.0.0.1" not in out and "4111" not in out and "hunter2" not in out
