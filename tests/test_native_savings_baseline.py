"""Every optimizer path must measure savings against the same 32K baseline.

The native optimizer reports ``tokens_saved = corpus - selected``. The QCCR
path caps that at :func:`naive_context_baseline`, but an empty query (or one
QCCR cannot condition on) reaches the native optimizer, whose figure was passed
through unchanged. Measured on the Entroly repository: one MCP
``optimize_context`` call delivered 1,338 tokens and reported
``tokens_saved: 5,558,864``, which ``entroly value`` then displayed as
"Tokens reduced: 5,558,864" and "$16.6766" of value.
"""

from __future__ import annotations

import pytest

from entroly.config import EntrolyConfig
from entroly.engine import _NAIVE_CONTEXT_BASELINE_TOKENS
from entroly.server import EntrolyEngine

BUDGET = 2_000
FRAGMENTS = 60
TOKENS_PER_FRAGMENT = 1_000


def _module(index: int) -> str:
    body = "\n".join(
        f"    total_{index}_{line} = shelf_{index}.weigh(bin_{line}) * rate_{line}"
        for line in range(60)
    )
    text = f"def audit_shelf_{index}(shelf_{index}):\n{body}\n    return total_{index}_0\n"
    return text[: TOKENS_PER_FRAGMENT * 4]


@pytest.fixture
def engine() -> EntrolyEngine:
    engine = EntrolyEngine(EntrolyConfig(use_persistent_index=False))
    for index in range(FRAGMENTS):
        content = _module(index)
        engine.ingest_fragment(content, f"pkg/shelf_{index}.py", len(content) // 4)
    return engine


@pytest.mark.parametrize("query", ["", "What is this?"])
def test_reported_saving_is_held_to_the_naive_context_baseline(engine, query):
    corpus = FRAGMENTS * TOKENS_PER_FRAGMENT
    assert corpus > _NAIVE_CONTEXT_BASELINE_TOKENS  # the cap has to matter

    result = engine.optimize_context(token_budget=BUDGET, query=query)

    selected = result.get("selected_fragments") or result.get("selected") or []
    delivered = sum(int(f.get("token_count", 0) or 0) for f in selected)
    sent = max(delivered, int(result.get("total_tokens", 0) or 0))
    ceiling = max(0, _NAIVE_CONTEXT_BASELINE_TOKENS - sent)
    assert result.get("tokens_saved", 0) <= ceiling, (
        f"query={query!r}: tokens_saved={result.get('tokens_saved')} for {sent} "
        f"tokens sent; the {_NAIVE_CONTEXT_BASELINE_TOKENS:,}-token baseline allows "
        f"at most {ceiling}"
    )
