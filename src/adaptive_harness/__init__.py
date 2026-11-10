"""adaptive_harness: a model-agnostic, evidence-driven prompt adaptation harness."""

from ._version import __version__
from .api import AdaptiveHarness, HarnessResponse, Trace
from .config import BudgetConfig, HarnessConfig, HTTPConfig, PersistenceConfig, PolicyConfig, RewriterConfig
from .errors import (
    AuthenticationError,
    BudgetExceededError,
    ConfigurationError,
    ContextLengthError,
    HarnessError,
    InvalidRequestError,
    MalformedResponseError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    StoreError,
)
from .types import CompletionRequest, CompletionResult, FinishReason, GenerationSettings, Message, Usage

__all__ = [
    "AdaptiveHarness",
    "AuthenticationError",
    "BudgetConfig",
    "BudgetExceededError",
    "CompletionRequest",
    "CompletionResult",
    "ConfigurationError",
    "ContextLengthError",
    "FinishReason",
    "GenerationSettings",
    "HTTPConfig",
    "HarnessConfig",
    "HarnessError",
    "HarnessResponse",
    "InvalidRequestError",
    "MalformedResponseError",
    "Message",
    "PersistenceConfig",
    "PolicyConfig",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QuotaExceededError",
    "RateLimitError",
    "RewriterConfig",
    "StoreError",
    "Trace",
    "Usage",
    "__version__",
]
