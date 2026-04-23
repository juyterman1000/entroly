---
claim_id: b54c8323-2341-4945-95ed-56c0ac569895
entity: flow_orchestrator
status: inferred
confidence: 0.75
sources:
  - entroly\flow_orchestrator.py:33
  - entroly\flow_orchestrator.py:57
  - entroly\flow_orchestrator.py:44
  - entroly\flow_orchestrator.py:65
  - entroly\flow_orchestrator.py:84
last_checked: 2026-04-23T03:07:07.810113+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: flow_orchestrator

**Language:** python
**Lines of code:** 470

## Types
- `class FlowResult()` — Result of executing a canonical flow.
- `class FlowOrchestrator()` — Executes the 5 canonical epistemic flows. Takes a RoutingDecision from the EpistemicRouter and chains the appropriate pipeline steps together.

## Functions
- `def to_dict(self) -> dict[str, Any]`
- `def __init__(
        self,
        vault: VaultManager,
        router: EpistemicRouter,
        compiler: BeliefCompiler,
        verifier: VerificationEngine,
        change_pipe: ChangePipeline,
        evolution: EvolutionLogger,
        source_dir: str | None = None,
    )`
- `def execute(
        self,
        query: str,
        decision: RoutingDecision | None = None,
        diff_text: str = "",
        is_event: bool = False,
        event_type: str = "",
    ) -> FlowResult`

## Dependencies
- `.belief_compiler`
- `.change_pipeline`
- `.epistemic_router`
- `.evolution_logger`
- `.vault`
- `.verification_engine`
- `__future__`
- `dataclasses`
- `logging`
- `time`
- `typing`
