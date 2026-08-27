"""Source bytes that are not valid UTF-8 must not break snapshot persistence.

The repository intelligence input path decodes source with
``errors="surrogateescape"`` (``parsers.py``, ``lsp_orchestrator.py``,
``interprocedural_flow.py``, ``adaptive_program_graph.py``). A file carrying a
single non-UTF-8 byte -- a latin-1 name in a string literal is the ordinary
case -- therefore yields Python text containing lone surrogates.

``json.dumps`` serialises those as ``\\udcXX``. The bytes stay pure ASCII, so
they pass every byte-level check the cross-runtime parity test makes, but a
lone surrogate is not valid JSON under RFC 8259. ``serde_json`` rejects it, so
the Rust verifier that owns commitment validity cannot read what Python wrote.

Python files are accidentally shielded: ``ast.parse`` raises
``UnicodeEncodeError`` first and the file is recorded with a ``parse_error`` and
no symbols. Languages that do not go through ``ast`` have no such gate, which is
why the fixture below is JavaScript.

Marked ``xfail(strict=True)``: the defect is real and unfixed, and strict mode
turns the test into a failure the moment the behaviour changes, so a fix cannot
land silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly.repository_intelligence import RepositoryIntelligenceService
from entroly.work_context_snapshot_store import WorkContextSnapshotStore
from entroly.work_graph_store import WorkGraphStore


# Written as a bytes literal on purpose: the whole point is a byte sequence that
# is not valid UTF-8, which no str literal can express.
_LEGACY_JS = (
    b"function parseLegacy(value) {\n"
    b"  const author = 'Bj\xf6rn';\n"
    b"  return value.trim() + author;\n"
    b"}\n"
    b"module.exports = { parseLegacy };\n"
)

_CALLER_JS = (
    "const { parseLegacy } = require('./legacy');\n\n"
    "function go(value) {\n  return parseLegacy(value);\n}\n"
)

_SURROGATE_ESCAPE = "\\udc"


def _build_payload(repo: Path) -> dict:
    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "legacy.js").write_bytes(_LEGACY_JS)
    (repo / "src" / "caller.js").write_text(_CALLER_JS, encoding="utf-8")

    service = RepositoryIntelligenceService(repo)
    index, _digest, _generation = service._snapshot()

    # The file indexes cleanly -- nothing warns the caller that its bytes will
    # later be unrepresentable.
    assert index.files["src/legacy.js"].parse_error is None
    target = next(
        symbol for symbol in index.symbols.values() if symbol.name == "parseLegacy"
    )
    return service.context(
        "legacy parsing author",
        token_budget=512,
        max_hops=1,
        proposal_scores=[{"symbol_id": target.symbol_id, "score": 1.0}],
        proposal_provider="source-byte-regression",
    )


def test_non_utf8_source_byte_reaches_the_payload(tmp_path: Path) -> None:
    """Guards the premise: the surrogate really does survive into the payload.

    Kept separate from the xfail below so a change that merely stops the
    surrogate reaching the payload cannot be mistaken for a snapshot fix.
    """
    payload = _build_payload(tmp_path / "repo")
    assert _SURROGATE_ESCAPE in json.dumps(payload, ensure_ascii=True)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A non-UTF-8 byte inside a non-Python symbol body makes the snapshot "
        "unpersistable: Python emits a lone surrogate and the Rust verifier "
        "rejects it as invalid JSON. Fixing this is a commitment-format "
        "decision, so it is tracked rather than papered over."
    ),
)
def test_snapshot_survives_non_utf8_source_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    payload = _build_payload(repo)

    store = WorkContextSnapshotStore(
        WorkGraphStore("repo:source-byte-regression", root=tmp_path / "state")
    )
    token = store.put_json(payload)

    restored = store.get_json(token)
    assert restored == payload
