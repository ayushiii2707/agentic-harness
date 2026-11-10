"""Exception hierarchy.

Every error the package raises derives from :class:`HarnessError`. Provider errors are normalised so the rest of
the harness never depends on a particular provider's status codes or error bodies.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all errors raised by adaptive_harness."""


class ConfigurationError(HarnessError):
    """Invalid or incomplete configuration (unknown provider, missing key, bad value)."""


class StoreError(HarnessError):
    """The evidence store could not be opened, migrated or written."""


class BudgetExceededError(HarnessError):
    """A paid operation was refused because it would exceed the configured budget."""

    def __init__(self, message: str, *, requested_usd: float, remaining_usd: float) -> None:
        super().__init__(message)
        self.requested_usd = requested_usd
        self.remaining_usd = remaining_usd


# --------------------------------------------------------------------------- provider errors
class ProviderError(HarnessError):
    """A model provider call failed. ``retryable`` says whether retrying the same request can succeed."""

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id


class AuthenticationError(ProviderError):
    """The credential is missing, invalid or lacks permission (401/403)."""


class QuotaExceededError(ProviderError):
    """The account has no credit or has hit a hard quota (e.g. 402). Not retryable."""


class RateLimitError(ProviderError):
    """Too many requests (429). Retryable after ``retry_after`` seconds when the provider says so."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str = "",
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code, request_id=request_id)
        self.retry_after = retry_after


class ContextLengthError(ProviderError):
    """The request is longer than the model's context window."""


class InvalidRequestError(ProviderError):
    """The provider rejected the request as malformed or unsupported (400/404/422)."""


class ProviderUnavailableError(ProviderError):
    """Server error, overload or connection failure (5xx, network). Retryable."""

    retryable = True


class ProviderTimeoutError(ProviderError):
    """The request timed out. Retryable."""

    retryable = True


class MalformedResponseError(ProviderError):
    """The provider answered, but the body could not be parsed into a result."""
