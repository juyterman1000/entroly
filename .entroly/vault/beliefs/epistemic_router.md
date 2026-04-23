---
claim_id: 486873ba-3232-403a-8af8-f85cc694803f
entity: epistemic_router
status: inferred
confidence: 0.75
sources:
  - entroly\epistemic_router.py:39
  - entroly\epistemic_router.py:56
  - entroly\epistemic_router.py:65
  - entroly\epistemic_router.py:77
  - entroly\epistemic_router.py:88
  - entroly\epistemic_router.py:197
  - entroly\epistemic_router.py:98
  - entroly\epistemic_router.py:172
  - entroly\epistemic_router.py:180
  - entroly\epistemic_router.py:209
last_checked: 2026-04-23T03:07:07.806221+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: epistemic_router

**Language:** python
**Lines of code:** 663

## Types
- `class EpistemicIntent(str, Enum)` — What kind of answer is needed.
- `class EpistemicFlow(str, Enum)` — Which canonical flow to execute.
- `class RiskLevel(str, Enum)` — How dangerous is the answer domain.
- `class BeliefCoverage()` — Result of checking belief coverage in the vault.
- `class RoutingDecision()` — The router's output: which flow to execute and why.
- `class EpistemicRouter()` — Epistemic Ingress Controller. Inspects 4 signals (intent, coverage, freshness, risk) and selects one of 5 canonical flows. Primary home: Action Layer (server.py + query_refiner.py) Inputs from:  Belie

## Functions
- `def to_dict(self) -> dict[str, Any]`
- `def classify_intent(query: str) -> EpistemicIntent` — Classify a query into an epistemic intent using keyword patterns.
- `def assess_risk(query: str, intent: EpistemicIntent) -> RiskLevel` — Assess the risk level of a query domain.
- `def __init__(
        self,
        vault_path: str | None = None,
        miss_threshold: int = 3,
        freshness_hours: float = 24.0,
        min_confidence: float = 0.6,
    )`
- `def route(
        self,
        query: str,
        is_event: bool = False,
        event_type: str | None = None,
    ) -> RoutingDecision`
- `def record_miss(self, query: str) -> None` — Explicitly record a miss (system couldn't answer satisfactorily).
- `def record_outcome(
        self,
        flow: str,
        success: bool,
        confidence: float = 0.0,
        component_bus: Any = None,
    ) -> None`
- `def stats(self) -> dict[str, Any]` — Return routing statistics for observability.

## Dependencies
- `Enum`
- `__future__`
- `dataclasses`
- `enum`
- `logging`
- `pathlib`
- `re`
- `str`
- `time`
- `typing`
- `uuid`
