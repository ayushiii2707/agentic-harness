"""Strategy application with preservation guards; failures fall back to the original prompt."""

from ..auxiliary import AuxModel
from .engine import Applied, apply_strategy
from .llm_rewrite import LlmRewrite, build_llm_rewrite
from .preservation import PreservationResult, check_rewrite, exact_problems

__all__ = [
    "Applied",
    "AuxModel",
    "LlmRewrite",
    "PreservationResult",
    "apply_strategy",
    "build_llm_rewrite",
    "check_rewrite",
    "exact_problems",
]
