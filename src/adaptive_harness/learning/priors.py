"""External prior evidence (DESIGN §6.6), stored as data in ``prior_evidence``, never as code paths.

The only records shipped are from our own prior study, LeBlanc v4: the secure-and-working rate (an output
passes its functional tests *and* its security test) on 56 Python CodeSecEval tasks x 3 repetitions = 168 outputs
per (model, condition). ``plain`` is the original task; ``generic`` adds the exact sentence used by
``security_generic``. LeBlanc appended it to the task, the harness sends it as a system message, so the
intervention is close but not identical; that is one more reason the weight is small.

The records apply only to cells with the same model ID (vendor prefix ignored) and category
``security_sensitive_code``. Their host as tested is not recorded here, so their weight is halved (DESIGN §6.6):
1.5 pseudo-observations, against the 30 own outcomes the policy needs before it compares anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..store import Store


@dataclass(frozen=True)
class PriorRecord:
    citation: str
    model_id_as_tested: str
    category: str
    strategy: str
    metric: str
    effect: float
    evidence_date: str
    host: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    verdict: str | None = None
    weight: float = 3.0


_CITE = (
    "LeBlanc v4, secure-and-working rate by condition (RQ1/RQ5 tables; 56 tasks x 3 reps = 168 outputs/arm)"
)
_METRIC = "secure_and_working_rate"
_CAT = "security_sensitive_code"


def _pair(
    model: str, plain: float, generic: float, ci: tuple[float, float]
) -> tuple[PriorRecord, PriorRecord]:
    verdict = "not significant (LeBlanc); prior only"
    return (
        PriorRecord(_CITE, model, _CAT, "identity", _METRIC, plain, "2026-11-10", None, *ci, verdict),
        PriorRecord(
            _CITE, model, _CAT, "security_generic", _METRIC, generic, "2026-11-10", None, None, None, verdict
        ),
    )


LEBLANC_V4: tuple[PriorRecord, ...] = (
    *_pair("gpt-oss-20b", 0.361, 0.313, (0.233, 0.489)),
    *_pair("gemini-3.5-flash-lite", 0.274, 0.326, (0.156, 0.400)),
    *_pair("deepseek-v4.1-flash", 0.351, 0.448, (0.220, 0.478)),
)


def load_priors(store: Store, records: tuple[PriorRecord, ...] = LEBLANC_V4) -> int:
    """Insert records not already present (same citation, model, category, strategy, metric). Returns how many."""
    added = 0
    with store.transaction() as con:
        for r in records:
            exists = con.execute(
                "SELECT 1 FROM prior_evidence WHERE citation=? AND model_id_as_tested=? AND category=? "
                "AND strategy=? AND metric=?",
                (r.citation, r.model_id_as_tested, r.category, r.strategy, r.metric),
            ).fetchone()
            if exists:
                continue
            con.execute(
                "INSERT INTO prior_evidence(citation, model_id_as_tested, host, category, strategy, metric, effect, "
                "ci_low, ci_high, verdict, evidence_date, weight) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    r.citation, r.model_id_as_tested, r.host, r.category, r.strategy, r.metric, r.effect,
                    r.ci_low, r.ci_high, r.verdict, r.evidence_date, r.weight,
                ),
            )  # fmt: skip
            added += 1
    return added
