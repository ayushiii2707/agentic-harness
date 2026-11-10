"""Model identity (DESIGN §5.1).

A model is identified by (provider, model id, deployment) plus the *resolved version* the provider reports when it
answers. The fingerprint is a hash of the fields that say which model actually served a call. It deliberately
excludes ``system_fingerprint``: OpenAI-style backends change that value with routine configuration changes, so
it is kept as an observed fact rather than treated as a new model.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from ..hashing import canonical_hash
from ..types import CompletionResult

# Fields that say *which model* served a call. ``serving_host`` (e.g. which OpenRouter backend) is deliberately not
# one of them: routers switch hosts between calls, and treating that as a model change disabled learning
# (evaluation finding F2). The host is still recorded as an observed fact; a behaviour change it causes is caught
# by drift detection, which is evidence-based.
FINGERPRINT_FIELDS = ("model_reported", "model_version")
_ALIAS_MARKERS = ("latest",)  # e.g. gemini-flash-latest, llama3:latest


@dataclass(frozen=True)
class ModelKey:
    provider: str
    model_id: str
    deployment: str  # "hosted" or "local:<host:port>"

    @property
    def alias(self) -> bool:
        """True for moving aliases (e.g. ``*-latest``) whose underlying model can change without notice."""
        low = self.model_id.lower()
        return any(m in low for m in _ALIAS_MARKERS)


def deployment_for(base_url: str | None, local: bool) -> str:
    if not local:
        return "hosted"
    host = urlparse(base_url).netloc if base_url else ""
    return f"local:{host or 'unknown'}"


def resolved_version(result: CompletionResult) -> str | None:
    """The most specific model version the provider reported for this call, if any."""
    return result.fingerprint.get("model_version") or result.model_reported


def fingerprint_of(result: CompletionResult) -> str | None:
    fields = {k: result.fingerprint[k] for k in FINGERPRINT_FIELDS if result.fingerprint.get(k)}
    if not fields and result.model_reported:
        fields = {"model_reported": result.model_reported}
    return canonical_hash(fields) if fields else None
