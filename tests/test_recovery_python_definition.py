"""Recover a fitting Python definition, not just its matching signature."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from entroly.compression_retrieval_store import (
    _MAX_PYTHON_EXCERPT_SOURCE_CHARS,
    _bounded_exact_excerpt,
    _query_local_python_definition,
)
from entroly.compression_retrieval_store_secure import CompressionRetrievalStore
from entroly.tokens import count_tokens


DEFINITION = (
    "def is_allowed(user):\n"
    "    if user.suspended:\n"
    "        return False\n"
    "    return user.role == 7\n"
)


def _source(definition=DEFINITION):
    return (
        "".join(f"# unrelated note {i} about routine work\n" for i in range(250))
        + "\n"
        + definition
        + "\n"
        + "".join(f"# archive {i}\n" for i in range(100))
    )


def _store(store, source):
    return store.put(
        original_text=source,
        compressed_text="[omitted]",
        receipt={
            "original_tokens": count_tokens(source),
            "compressed_tokens": 3,
            "omitted_spans": [
                {"start_line": 1, "end_line": len(source.splitlines())}
            ],
        },
    )


@pytest.mark.parametrize("budget", [96, 160, 256])
@pytest.mark.parametrize("record", [False, True])
def test_restarted_public_store_recovers_complete_fitting_definition(
    tmp_path, budget, record
):
    path = tmp_path / "recovery.json"
    original = _source()
    created = _store(CompressionRetrievalStore(path, scope_id="alpha"), original)
    restarted = CompressionRetrievalStore(path, scope_id="alpha")

    matches = restarted.search_exact_excerpts(
        "is_allowed",
        max_tokens_per_span=budget,
        record_retrieval=record,
        retrieval_id="definition-recovery" if record else None,
    )

    assert len(matches) == 1
    excerpt = matches[0]
    assert DEFINITION in excerpt.content
    assert "exact excerpt gap" in excerpt.content
    assert count_tokens(excerpt.content) <= budget
    assert excerpt.content_sha256 == hashlib.sha256(excerpt.content.encode()).hexdigest()
    full = restarted.get_span(created.receipt_id, created.spans[0].span_id)
    assert full.content == original
    assert full.retrieval_count == int(record)
    if record:
        assert full.retrieved_tokens == count_tokens(excerpt.content)
        restarted.search_exact_excerpts(
            "is_allowed", max_tokens_per_span=budget,
            record_retrieval=True, retrieval_id="definition-recovery",
        )
        again = restarted.get_span(created.receipt_id, created.spans[0].span_id)
        assert again.retrieval_count == 1
        assert again.retrieved_tokens == full.retrieved_tokens
    assert not CompressionRetrievalStore(path, scope_id="beta").search_exact_excerpts(
        "is_allowed", max_tokens_per_span=budget
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    ("definition", "query"),
    [
        ("@requires_review\n" + DEFINITION, "is_allowed"),
        ("async " + DEFINITION, "is_allowed"),
        ("class Policy:\n    permitted = True\n", "Policy"),
        (
            "class Policy:\n"
            "    @requires_review\n"
            "    def is_allowed(self, user):\n"
            "        return user.role == '管理员'\n",
            "is_allowed",
        ),
    ],
)
def test_definition_retains_decorators_unicode_and_python_newlines(
    definition, query, newline
):
    definition = definition.replace("\n", newline)
    # Unlike Python's line numbering, str.splitlines splits U+2028 even
    # inside a string literal. That used to be a source-offset trap.
    source = "separator = 'a\u2028b'\n" + _source(definition)
    excerpt, sliced = _bounded_exact_excerpt(source, query, 160)

    exact = definition
    if query == "is_allowed" and definition.startswith("class Policy:"):
        # Return the decorated method, not an unrelated enclosing class.
        exact = definition[definition.index("    @requires_review"):]
    assert exact in excerpt
    assert sliced
    assert count_tokens(excerpt) <= 160


@pytest.mark.parametrize(
    "source",
    [
        _source(DEFINITION + "\n" + DEFINITION),
        _source("def is_allowed(:\n    pass\n"),
        _source("def is_allowed(user):\n" + "    value = 123\n" * 200),
        _source(DEFINITION) + "\x00",
    ],
)
def test_ambiguous_invalid_and_oversized_definitions_use_existing_fallback(source):
    assert _query_local_python_definition(source, "is_allowed", 96) is None
    excerpt, _ = _bounded_exact_excerpt(source, "is_allowed", 96)
    assert count_tokens(excerpt) <= 96


def test_parser_input_is_bounded_before_parsing(monkeypatch):
    def unexpected_parse(*args, **kwargs):
        raise AssertionError("oversized input must not reach AST parsing")

    monkeypatch.setattr("ast.parse", unexpected_parse)
    source = "#" * (_MAX_PYTHON_EXCERPT_SOURCE_CHARS + 1) + "\n" + DEFINITION
    assert _query_local_python_definition(source, "is_allowed", 96) is None


def test_recovered_source_is_parsed_never_executed():
    source = "raise RuntimeError('source must not execute')\n" + _source()
    assert DEFINITION in _query_local_python_definition(source, "is_allowed", 96)


def test_already_fitting_source_is_returned_unchanged():
    excerpt, sliced = _bounded_exact_excerpt(DEFINITION, "is_allowed", 96)
    assert excerpt == DEFINITION
    assert not sliced


def test_mcp_search_returns_the_definition_body(tmp_path, monkeypatch):
    from entroly.compression_mcp import create_compression_mcp_server

    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setenv("ENTROLY_OPTIMIZATION_LEDGER", str(tmp_path / "ledger.sqlite3"))
    path = tmp_path / "recovery.json"
    _store(CompressionRetrievalStore(path), _source())
    server = create_compression_mcp_server(str(path))

    blocks = asyncio.run(
        server.call_tool(
            "search_compressed_spans",
            {"query": "is_allowed", "max_tokens_per_span": 96},
        )
    )
    # MCP SDK versions may return (content blocks, structured content).
    if isinstance(blocks, tuple):
        blocks = blocks[0]
    payload = json.loads(blocks[0].text)

    assert payload["status"] == "ok"
    assert DEFINITION in payload["spans"][0]["content"]
    assert count_tokens(payload["spans"][0]["content"]) <= 96


def test_session_rescue_recovery_keeps_frozen_prefix_across_topic_change(tmp_path):
    from entroly.session_rescue import SessionRescueController, SessionRescuePolicy

    path = tmp_path / "session.json"
    store = CompressionRetrievalStore(path, scope_id="alpha")
    controller = SessionRescueController(
        recovery_store=store,
        policy=SessionRescuePolicy(
            soft_watermark=0.20, hard_watermark=0.30, target_watermark=0.10,
            failure_watermark=0.98, loop_min_watermark=0.05,
            tail_messages=2, max_block_budget=96,
        ),
    )
    messages = [
        {"role": "system", "content": "Read the source before answering."},
        {"role": "tool", "name": "read_file", "content": _source()},
        {"role": "assistant", "content": "Reviewing the repository."},
        {"role": "user", "content": "Continue inspecting other files."},
    ]
    first = controller.rescue("coding", messages, context_window=5_000)
    assert first.recovery_receipts
    grown = messages + [
        {"role": "assistant", "content": "Other files have been inspected."},
        {"role": "user", "content": "Return to is_allowed and its suspension check."},
    ]
    second = controller.rescue("coding", grown, context_window=5_000)
    assert first.messages[1] == second.messages[1]

    recovered = CompressionRetrievalStore(path, scope_id="alpha").search_exact_excerpts(
        "is_allowed", max_tokens_per_span=96, record_retrieval=True
    )

    assert DEFINITION in recovered[0].content
    assert first.messages[1] == second.messages[1]
    assert messages[1]["content"] == _source()
