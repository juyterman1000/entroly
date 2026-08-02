"""`sdk.optimize` must behave identically on the native and pure-Python engines.

Why this file exists
--------------------

`optimize_context` names its result key differently depending on which engine
answered: the pure-Python path returns only ``selected_fragments``, while the
native path additionally normalises a ``selected`` alias. ``sdk.optimize`` read
``selected`` alone, so it returned an EMPTY selection at every budget for every
user *without* the Rust extension -- the default ``pip install entroly``
experience -- while working perfectly in development, where the native engine
is present.

The existing SDK tests did not catch it for exactly that reason: CI and dev
machines have the native engine, so the fallback branch -- the one most users
actually run -- was never exercised. These tests pin both engines.
"""

from __future__ import annotations

from entroly import sdk

FRAGMENTS = [
    {
        "source": "auth.py",
        "content": (
            "def login(user, pw):\n"
            "    token = verify_password(user, pw)\n"
            "    return issue_session_token(token)"
        ),
    },
    {
        "source": "billing.py",
        "content": (
            "def charge_card(customer, amount):\n"
            "    return StripeGateway().charge(customer.card, amount)"
        ),
    },
]


def test_optimize_reads_the_key_the_pure_python_engine_sets(tmp_path, monkeypatch):
    """Pin the exact defect, without needing a pure-Python environment.

    Monkeypatching `_RUST_AVAILABLE` is NOT a faithful simulation: whenever
    `entroly_core` is importable, `ContextFragment` still resolves to the Rust
    class, producing a hybrid state that cannot occur in reality. So instead of
    faking the engine, fake its *response shape*: return exactly what the
    pure-Python path returns -- `selected_fragments` and no `selected` alias --
    and assert the SDK still produces usable context.

    This runs identically with or without the native extension, which matters
    because CI has the extension and would otherwise never cover the branch
    most users actually run.
    """
    from entroly.server import EntrolyEngine

    monkeypatch.chdir(tmp_path)

    def pure_python_shaped(self, token_budget=0, query="", **kwargs):
        return {
            "selected_fragments": [
                {"source": "auth.py", "content": FRAGMENTS[0]["content"], "token_count": 20}
            ],
            "total_tokens": 20,
            "selected_count": 1,
        }

    monkeypatch.setattr(EntrolyEngine, "optimize_context", pure_python_shaped)
    result = sdk.optimize(FRAGMENTS, budget=200, query="how does user login work")
    assert result["fragments_total"] == len(FRAGMENTS)
    assert result["fragments_selected"] > 0, (
        "optimize() selected nothing -- a user gets empty context back. "
        f"result={result!r}"
    )
    assert result["context_text"], "context_text is empty; nothing to inject"
    assert result["total_tokens"] > 0
    assert result["selected"][0]["source"] == "auth.py"


def test_optimize_returns_context_on_default_engine(tmp_path, monkeypatch):
    """Whatever engine is installed here, the contract is the same."""
    monkeypatch.chdir(tmp_path)
    result = sdk.optimize(FRAGMENTS, budget=200, query="how does user login work")
    assert result["fragments_total"] == len(FRAGMENTS)
    assert result["fragments_selected"] > 0, f"selected nothing: {result!r}"
    assert result["context_text"], "context_text is empty; nothing to inject"
    assert result["total_tokens"] > 0


def test_optimize_never_selects_more_than_it_was_given(tmp_path, monkeypatch):
    """Guards against a persisted on-disk index leaking into a stateless call.

    Run inside a directory that already carries an entroly index, `optimize`
    returned fragments belonging to that project rather than the caller's, and
    reported `selected=5/3` -- more selected than supplied. The signature reads
    as a pure function of its arguments, so it must behave like one.
    """
    monkeypatch.chdir(tmp_path)
    result = sdk.optimize(FRAGMENTS, budget=8000, query="login")
    assert result["fragments_selected"] <= result["fragments_total"]
    sources = {
        f.get("source") for f in result["selected"] if isinstance(f, dict)
    }
    assert sources <= {f["source"] for f in FRAGMENTS}, (
        f"optimize() returned fragments the caller never supplied: {sources}"
    )
