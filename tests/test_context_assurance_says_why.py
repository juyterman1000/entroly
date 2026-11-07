"""A fail-closed assurance error must say what could not be verified.

The documented `assure_context` example, run after `pip install entroly`,
raised "context assurance failed closed: uncertain" and nothing else. The
reason was in the certificate: the budget check was "unavailable" because the
exact tokenizer (tiktoken) is not installed by default, so only a
character heuristic could count the selection. Failing closed is right; not
saying why sent users into the certificate JSON to find out.
"""

from __future__ import annotations

import pytest

import entroly.context_assurance as context_assurance
from entroly import AssuranceScope, assure_context
from entroly.context_assurance import ContextAssuranceError

SCOPE = AssuranceScope(project="repo:app", session="task:123", agent="agent:1")
DOCS = [("facts.txt", "The service retry budget is seven attempts.")]


def test_without_an_exact_tokenizer_the_error_names_it(monkeypatch) -> None:
    monkeypatch.setattr(context_assurance, "_encoding", lambda: None)

    with pytest.raises(ContextAssuranceError) as error:
        assure_context(DOCS, query="service retry budget", budget=200, scope=SCOPE)

    message = str(error.value)
    assert message.startswith("context assurance failed closed: uncertain")
    assert "budget unverified" in message and "o200k_base" in message, message


def test_with_an_exact_tokenizer_the_documented_example_passes() -> None:
    from entroly.tokens import _encoding

    if _encoding() is None:
        pytest.skip("o200k_base is not cached locally; Entroly never downloads it")

    result = assure_context(DOCS, query="service retry budget", budget=200, scope=SCOPE)

    assert result["selected_context"]
