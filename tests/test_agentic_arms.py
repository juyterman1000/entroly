"""Tests for the agentic-task arm construction and engine isolation.

These cover the two ways this measurement can be quietly wrong:

1. an arm that does not actually differ from the control, so a "compression"
   result was never produced by compressing anything;
2. an engine warm start that mixes the developer's own repository into the
   selection, so the numbers describe a corpus the benchmark never supplied.

Both were live defects when this harness was written; (2) was observed against
the real engine and is reproduced here against a stub.
"""

from __future__ import annotations

import os

import pytest

from benchmarks.agentic_arms import (
    Arm,
    BuiltContext,
    Fragment,
    build_closed_loop,
    build_compressed,
    build_for_arm,
    build_raw,
    estimate_tokens,
)
from benchmarks.engine_isolation import (
    EngineNotIsolatedError,
    assert_engine_isolated,
    isolated_engine_dir,
)

FRAGMENTS = [
    Fragment(source="auth.py", content="def login(user):\n    return check(user)"),
    Fragment(source="billing.py", content="# invoices, taxes, unrelated to login"),
    Fragment(source="crypto.py", content="def check(u):\n    return u.pw == stored"),
]


def _stub_optimize(selected_sources: list[str]):
    """An optimize() stand-in returning a fixed selection in engine shape."""

    def _optimize(payload, budget, query):  # noqa: ANN001, ARG001
        by_source = {item["source"]: item for item in payload}
        return {
            "selected": [
                by_source[s] for s in selected_sources if s in by_source
            ],
            "fragments_total": len(payload),
        }

    return _optimize


# --------------------------------------------------------------------------
# Arm construction
# --------------------------------------------------------------------------

def test_raw_arm_includes_everything_and_omits_nothing() -> None:
    built = build_raw(FRAGMENTS)
    assert built.arm is Arm.RAW
    assert len(built.included) == len(FRAGMENTS)
    assert built.omitted == []
    for fragment in FRAGMENTS:
        assert fragment.content in built.text


def test_compressed_arm_actually_drops_something() -> None:
    """A compression arm identical to RAW would make the comparison vacuous."""
    built = build_compressed(
        FRAGMENTS,
        query="fix the login bug",
        budget=40,
        optimize_fn=_stub_optimize(["auth.py", "crypto.py"]),
    )
    assert built.arm is Arm.COMPRESS
    assert [f.source for f in built.included] == ["auth.py", "crypto.py"]
    assert [f.source for f in built.omitted] == ["billing.py"]
    assert "invoices" not in built.text
    assert built.estimated_tokens < build_raw(FRAGMENTS).estimated_tokens


def test_compressed_arm_refuses_to_silently_become_the_raw_arm() -> None:
    """An empty selection must fail, not fall back to full context."""
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        build_compressed(
            FRAGMENTS,
            query="q",
            budget=40,
            optimize_fn=_stub_optimize([]),
        )


def test_compressed_arm_rejects_a_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        build_compressed(
            FRAGMENTS, query="q", budget=0, optimize_fn=_stub_optimize(["auth.py"])
        )


def test_unrecognised_optimize_shape_raises_rather_than_returning_empty() -> None:
    """A shape change must surface, not read as 'compression dropped all'."""

    def _bad_optimize(payload, budget, query):  # noqa: ANN001, ARG001
        return {"unexpected_key": []}

    with pytest.raises(KeyError, match="could not find selected fragments"):
        build_compressed(
            FRAGMENTS, query="q", budget=40, optimize_fn=_bad_optimize
        )


def test_foreign_fragments_from_a_warm_index_are_not_admitted() -> None:
    """The engine may return sources that were never supplied; drop them.

    This is the shape of the real contamination: `optimize()` returned two
    files belonging to the developer's own repository. Whatever else happens,
    they must not end up in the measured context.
    """
    def _optimize(payload, budget, query):  # noqa: ANN001, ARG001
        by_source = {item["source"]: item for item in payload}
        return {
            "selected": [
                by_source["auth.py"],
                {"source": "file:entroly-wasm/bin/entroly-wasm.js", "content": "x"},
                {"source": "file:entroly/npm-alias/index.d.ts", "content": "y"},
            ],
            "fragments_total": len(payload),
        }

    built = build_compressed(
        FRAGMENTS, query="fix the login bug", budget=40, optimize_fn=_optimize
    )
    sources = [f.source for f in built.included]
    assert sources == ["auth.py"]
    assert not any("entroly-wasm" in s for s in sources)
    assert not any("npm-alias" in s for s in sources)


def test_closed_loop_starts_identical_to_compress() -> None:
    """Unpaired starting contexts would invalidate the arm comparison."""
    optimize_fn = _stub_optimize(["auth.py"])
    compress = build_for_arm(
        Arm.COMPRESS, FRAGMENTS, query="q", budget=40, optimize_fn=optimize_fn
    )
    closed = build_for_arm(
        Arm.CLOSED_LOOP, FRAGMENTS, query="q", budget=40, optimize_fn=optimize_fn
    )
    assert closed.text == compress.text
    assert [f.source for f in closed.included] == [f.source for f in compress.included]


def test_closed_loop_recovery_adds_omitted_spans_without_duplicating() -> None:
    base = build_compressed(
        FRAGMENTS, query="q", budget=40, optimize_fn=_stub_optimize(["auth.py"])
    )
    recovered = build_closed_loop(base, recovered=[FRAGMENTS[2], FRAGMENTS[0]])

    sources = [f.source for f in recovered.included]
    assert sources == ["auth.py", "crypto.py"], "already-included must not repeat"
    assert [f.source for f in recovered.omitted] == ["billing.py"]


def test_raw_arm_requires_no_engine() -> None:
    """RAW must never depend on optimize(), or the control is not a control."""
    built = build_for_arm(Arm.RAW, FRAGMENTS, query="q", budget=40, optimize_fn=None)
    assert built.arm is Arm.RAW


def test_compress_arm_without_an_engine_is_an_error() -> None:
    with pytest.raises(ValueError, match="requires optimize_fn"):
        build_for_arm(
            Arm.COMPRESS, FRAGMENTS, query="q", budget=40, optimize_fn=None
        )


def test_fragment_digest_identifies_omitted_content() -> None:
    a = Fragment(source="x.py", content="same")
    b = Fragment(source="y.py", content="same")
    c = Fragment(source="x.py", content="different")
    assert a.digest == b.digest, "digest is over content, not path"
    assert a.digest != c.digest


def test_built_context_serialises_what_was_dropped() -> None:
    built = build_compressed(
        FRAGMENTS, query="q", budget=40, optimize_fn=_stub_optimize(["auth.py"])
    )
    payload = built.to_dict()
    assert payload["omitted_sources"] == ["billing.py", "crypto.py"]
    assert len(payload["omitted_digests"]) == 2


def test_estimate_tokens_is_never_zero_for_nonempty_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1


# --------------------------------------------------------------------------
# Engine isolation
# --------------------------------------------------------------------------

def test_unset_entroly_dir_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    with pytest.raises(EngineNotIsolatedError, match="would warm-start"):
        assert_engine_isolated()


def test_directory_holding_an_existing_index_is_rejected(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A populated ENTROLY_DIR is the contaminating case; it must not pass."""
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "index.json.gz").write_bytes(b"\x1f\x8b")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    with pytest.raises(EngineNotIsolatedError, match="warm start would mix"):
        assert_engine_isolated()


def test_fresh_directory_passes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    assert assert_engine_isolated() == tmp_path


def test_isolated_context_sets_and_restores_the_variable() -> None:
    before = os.environ.get("ENTROLY_DIR")
    with isolated_engine_dir() as path:
        assert os.environ["ENTROLY_DIR"] == str(path)
        assert_engine_isolated()  # must satisfy its own precondition
    assert os.environ.get("ENTROLY_DIR") == before


def test_isolated_context_restores_even_after_an_error() -> None:
    before = os.environ.get("ENTROLY_DIR")
    with pytest.raises(ValueError):
        with isolated_engine_dir():
            raise ValueError("boom")
    assert os.environ.get("ENTROLY_DIR") == before


def test_isolation_does_not_evict_imported_modules_by_default() -> None:
    """Module surgery must be opt-in; it corrupts anything holding engine state.

    Dropping `entroly.*` from sys.modules makes a later import return *new*
    class objects, so a caller still holding the old ones has two incompatible
    copies. Doing this by default broke eight unrelated tests that shared a
    process with this file.
    """
    import sys

    import entroly  # noqa: F401  (imported for identity comparison)

    before = sys.modules.get("entroly")
    assert before is not None

    with isolated_engine_dir():
        assert sys.modules.get("entroly") is before, (
            "isolated_engine_dir() evicted entroly from sys.modules without "
            "being asked to; that corrupts other code in this process"
        )

    assert sys.modules.get("entroly") is before
