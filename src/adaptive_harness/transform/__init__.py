"""Strategy application with preservation guards; failures fall back to the original prompt."""

from .engine import Applied, apply_strategy
from .preservation import PreservationResult, check_rewrite

__all__ = ["Applied", "PreservationResult", "apply_strategy", "check_rewrite"]
