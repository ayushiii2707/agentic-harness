"""Strategy choice (DESIGN §6.4).

Exploit (default): send ``identity`` unless some strategy s has
    - at least ``min_trials`` evaluated outcomes, as does identity, and
    - P(theta_s > theta_identity + min_gain) >= 1 - (1 - confidence) / k,
where k is the number of challengers (a Bonferroni correction, so that several strategies each getting a lucky
streak do not add up to a false switch). The probability is estimated by Monte-Carlo sampling from the Beta
posteriors. With no evidence the rule never holds, so a fresh model always gets the original prompt.

Explore (opt-in, ``exploration_rate`` > 0): only on calls whose outcome will be observed (a check was supplied).
With probability ``exploration_rate`` the strategy is drawn **uniformly** from the candidates, identity included.
Uniform randomisation keeps the comparison a fair experiment and keeps the identity baseline well sampled; the
first version used Thompson sampling, which starved identity and produced a 37% false-switch rate in simulation
(see docs/DEVLOG.md, "decision loop").
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import PolicyConfig
from ..strategies import Strategy
from .evidence import ArmStats


@dataclass(frozen=True)
class Decision:
    strategy: Strategy
    reason: str
    explored: bool = False
    p_better: float | None = None


class Policy:
    def __init__(self, config: PolicyConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()

    def _arm(self, s: Strategy, arms: dict[tuple[str, str], ArmStats]) -> ArmStats:
        return arms.get((s.info.id, s.info.version), ArmStats(s.info.id, s.info.version))

    def p_better(self, a: ArmStats, b: ArmStats) -> float:
        """P(theta_a > theta_b + min_gain) under independent Beta posteriors."""
        n = self.config.mc_samples
        g = self.config.min_gain
        beat = sum(
            self.rng.betavariate(a.alpha, a.beta) > self.rng.betavariate(b.alpha, b.beta) + g
            for _ in range(n)
        )
        return beat / n

    def choose(
        self,
        candidates: Sequence[Strategy],
        arms: dict[tuple[str, str], ArmStats],
        *,
        can_explore: bool,
        exploit_allowed: bool,
    ) -> Decision:
        if not candidates or candidates[0].info.id != "identity":
            raise ValueError("candidates must start with identity")
        identity = candidates[0]
        others = list(candidates[1:])

        if can_explore and others and self.rng.random() < self.config.exploration_rate:
            pick = self.rng.choice(list(candidates))
            return Decision(pick, "exploration: uniform random choice among candidates", explored=True)

        if not exploit_allowed:
            return Decision(identity, "identity: no current profile evidence for this model")
        if not others:
            return Decision(identity, "identity: no other strategy applies to this task")

        base = self._arm(identity, arms)
        if base.n_trials < self.config.min_trials:
            return Decision(
                identity,
                f"identity: baseline has {base.n_trials} < {self.config.min_trials} evaluated outcomes",
            )
        required = 1.0 - (1.0 - self.config.confidence) / len(others)
        best_p, best = 0.0, None
        for s in others:
            arm = self._arm(s, arms)
            if arm.n_trials < self.config.min_trials:
                continue
            p = self.p_better(arm, base)
            if p > best_p:
                best_p, best = p, s
        if best is not None and best_p >= required:
            return Decision(
                best,
                f"evidence: P(beats identity by >= {self.config.min_gain:.2f}) = {best_p:.3f} "
                f"(required {required:.3f})",
                p_better=best_p,
            )
        detail = f" (best {best.info.id}: {best_p:.3f})" if best is not None else ""
        return Decision(
            identity,
            f"identity: no strategy beats it with probability >= {required:.3f}{detail}",
            p_better=best_p if best is not None else None,
        )
