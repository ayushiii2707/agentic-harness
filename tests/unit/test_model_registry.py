from __future__ import annotations

from pathlib import Path

from adaptive_harness import AdaptiveHarness, FinishReason, HarnessConfig, PersistenceConfig, Usage
from adaptive_harness.models import ModelKey, ModelRegistry, deployment_for, fingerprint_of, resolved_version
from adaptive_harness.providers import MockProvider
from adaptive_harness.providers.base import MetadataFact, ModelInfo
from adaptive_harness.store import Store
from adaptive_harness.types import CompletionResult

KEY = ModelKey("groq", "openai/gpt-oss-20b", "hosted")


def result(model: str | None = "openai/gpt-oss-20b", **fp: str) -> CompletionResult:
    return CompletionResult(
        text="ok",
        finish_reason=FinishReason.STOP,
        usage=Usage(1, 1),
        model_reported=model,
        latency_s=0.1,
        provider="groq",
        fingerprint=fp,
    )


def test_identity_helpers() -> None:
    assert deployment_for("http://localhost:11434/v1", local=True) == "local:localhost:11434"
    assert deployment_for("https://api.groq.com/openai/v1", local=False) == "hosted"
    assert ModelKey("gemini", "gemini-flash-latest", "hosted").alias
    assert not KEY.alias
    r = result(model_version="v2", system_fingerprint="fp_1")
    assert resolved_version(r) == "v2"
    # system_fingerprint is not part of the identity fingerprint
    assert fingerprint_of(r) == fingerprint_of(result(model_version="v2", system_fingerprint="fp_2"))
    assert fingerprint_of(result(model=None)) is None


def test_first_observation_creates_model_and_cold_profile() -> None:
    reg = ModelRegistry(Store(":memory:"))
    obs = reg.observe(KEY, result())
    assert obs.changed is False and obs.resolved_version == "openai/gpt-oss-20b"
    again = reg.observe(KEY, result())
    assert (again.model_ref, again.profile_id, again.changed) == (obs.model_ref, obs.profile_id, False)
    p = reg.profile(obs.model_ref)
    assert p is not None and p["status"] == "cold" and p["profile_version"] == 1


def test_new_version_marks_previous_profile_stale() -> None:
    store = Store(":memory:")
    reg = ModelRegistry(store)
    old = reg.observe(KEY, result(model_version="v1"))
    new = reg.observe(KEY, result(model_version="v2"))
    assert new.changed and new.model_ref != old.model_ref
    old_profile = reg.profile(old.model_ref)
    assert old_profile is not None and old_profile["status"] == "stale"
    assert old_profile["change_events"][0]["kind"] == "fingerprint"
    new_profile = reg.profile(new.model_ref)
    assert new_profile is not None and new_profile["status"] == "cold"
    assert reg.latest_model_ref(KEY) == new.model_ref


def test_a_changed_serving_host_is_a_fact_not_a_model_change() -> None:
    """Regression (evaluation F2): routers switch hosts; that used to mark the profile stale and stop learning."""
    reg = ModelRegistry(Store(":memory:"))
    a = reg.observe(KEY, result(serving_host="DeepInfra"))
    b = reg.observe(KEY, result(serving_host="OtherHost"))
    assert b.model_ref == a.model_ref and not b.changed
    p = reg.profile(a.model_ref)
    assert p is not None and p["status"] != "stale"
    hosts = {
        f["value"]
        for f in reg.store.query("SELECT value_json AS value FROM model_fact WHERE key='serving_host'")
    }
    assert hosts == {'"DeepInfra"', '"OtherHost"'}


def test_facts_keep_provenance_and_are_not_duplicated() -> None:
    reg = ModelRegistry(Store(":memory:"))
    ref = reg.observe(KEY, result(system_fingerprint="fp_1")).model_ref
    info = ModelInfo(
        "m",
        (
            MetadataFact("context_window", 131072, "verified", "/models"),
            MetadataFact("deployment", "hosted", "inferred", "preset"),
        ),
    )
    assert reg.record_facts(ref, info) == 2
    assert reg.record_facts(ref, info) == 0  # unchanged values are not stored twice
    facts = {f["key"]: (f["value"], f["provenance"]) for f in reg.facts(ref)}
    assert facts["context_window"] == (131072, "verified")
    assert facts["deployment"] == ("hosted", "inferred")
    assert facts["system_fingerprint"] == ("fp_1", "provider_reported")


def test_harness_links_trials_to_model_and_exposes_profile(tmp_path: Path) -> None:
    cfg = HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db"))
    h = AdaptiveHarness(
        provider="mock", model="m1", config=cfg, adapter=MockProvider(model_reported="m1-0613")
    )
    assert h.profile() is None  # nothing observed yet; no network call is made to identify the model
    h.generate("hi")
    p = h.profile()
    assert p is not None and p["resolved_version"] == "m1-0613" and p["status"] == "cold"
    row = h.store.query_one("SELECT model_ref FROM trial")  # type: ignore[union-attr]
    assert row is not None and row["model_ref"] == p["model_ref"]
