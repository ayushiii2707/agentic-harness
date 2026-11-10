from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pytest

from adaptive_harness._keys import Secret, get_secret, get_setting, redact
from adaptive_harness._logging import get_logger

FAKE = "sk-proj-THISISAFAKEKEYFORTESTS1234567890abcdef"


def test_secret_never_prints_its_value() -> None:
    s = Secret(FAKE)
    assert FAKE not in repr(s) and FAKE not in str(s) and FAKE not in f"{s}"
    assert s.reveal() == FAKE
    with pytest.raises(TypeError):
        pickle.dumps(s)


def test_redact_removes_registered_and_pattern_secrets() -> None:
    Secret("plainvalue-registered-0001")
    text = f"key={FAKE} other=plainvalue-registered-0001 groq=gsk_{'a' * 40} Authorization: Bearer abcdefghijklmnopqrstu"
    out = redact(text)
    assert FAKE not in out and "plainvalue-registered-0001" not in out and "gsk_" not in out
    assert "abcdefghijklmnopqrstu" not in out and "Bearer ***" in out


def test_logger_redacts_messages_args_and_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    Secret(FAKE)
    log = get_logger("test")
    with caplog.at_level(logging.DEBUG, logger="adaptive_harness"):
        log.warning("calling with %s", FAKE)
        try:
            raise RuntimeError(f"boom {FAKE}")
        except RuntimeError:
            log.exception("failed")
    joined = "\n".join(r.getMessage() + (r.exc_text or "") for r in caplog.records)
    assert FAKE not in joined
    assert "***" in joined


def test_get_secret_order_explicit_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("MY_KEY=from-file-value\nMY_SETTING=7 # comment\n", encoding="utf-8")
    assert get_secret("MY_KEY", env_file=env).reveal() == "from-file-value"  # type: ignore[union-attr]
    monkeypatch.setenv("MY_KEY", "from-env-value")
    assert get_secret("MY_KEY", env_file=env).reveal() == "from-env-value"  # type: ignore[union-attr]
    assert get_secret("MY_KEY", explicit="explicit-value").reveal() == "explicit-value"  # type: ignore[union-attr]
    assert get_secret("MISSING", env_file=env) is None
    assert get_setting("MY_SETTING", env_file=env) == "7"


def test_empty_secret_rejected() -> None:
    with pytest.raises(ValueError):
        Secret("   ")
