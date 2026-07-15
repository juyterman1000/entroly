from __future__ import annotations

import asyncio
import builtins
import json
import subprocess
import sys
import time

import pytest

from entroly.compression_mcp import create_compression_mcp_server
from entroly.compression_proxy import compress_proxy_payload, compress_proxy_payload_from_env
from entroly.compression_retrieval_store import (
    CompressionRetrievalStore,
    _count_o200k_tokens,
)


_CONCURRENT_WRITER = r"""
import json
import sys
import time
from pathlib import Path

from entroly.compression_retrieval_store import CompressionRetrievalStore

store_path = Path(sys.argv[1])
coordination_dir = Path(sys.argv[2])
worker_id = int(sys.argv[3])
entries = int(sys.argv[4])
store = CompressionRetrievalStore(store_path)
(coordination_dir / f"ready-{worker_id}").write_text("ready", encoding="utf-8")
start = coordination_dir / "start"
deadline = time.monotonic() + 30
while not start.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("test writer timed out waiting for start")
    time.sleep(0.005)

references = []
for entry_index in range(entries):
    payload = (
        f"worker={worker_id} entry={entry_index}\n"
        + (f"exact concurrent payload {worker_id}:{entry_index} " * 20).strip()
    )
    stored = store.put(
        original_text=payload,
        compressed_text="[omitted]",
        receipt={
            "original_tokens": len(payload) // 4,
            "compressed_tokens": 3,
            "omitted_spans": [{"start_line": 1, "end_line": 2}],
        },
    )
    references.append(
        {
            "receipt_id": stored.receipt_id,
            "span_id": stored.spans[0].span_id,
            "payload": payload,
        }
    )
(coordination_dir / f"result-{worker_id}.json").write_text(
    json.dumps(references), encoding="utf-8"
)
"""


_CONCURRENT_RETRIEVER = r"""
import sys
import time
from pathlib import Path

from entroly.compression_retrieval_store import CompressionRetrievalStore

store_path = Path(sys.argv[1])
coordination_dir = Path(sys.argv[2])
worker_id = int(sys.argv[3])
receipt_id = sys.argv[4]
span_id = sys.argv[5]
store = CompressionRetrievalStore(store_path)
(coordination_dir / f"retrieval-ready-{worker_id}").write_text(
    "ready", encoding="utf-8"
)
start = coordination_dir / "retrieval-start"
deadline = time.monotonic() + 30
while not start.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("test retriever timed out waiting for start")
    time.sleep(0.005)
span = store.retrieve_span(
    receipt_id,
    span_id,
    retrieval_id=f"concurrent-retrieval-{worker_id}",
)
if span is None:
    raise RuntimeError("stored span disappeared")
"""


def test_retrieval_store_saves_and_fetches_omitted_spans(tmp_path) -> None:
    store_path = tmp_path / "compression-store.json"
    store = CompressionRetrievalStore(store_path)
    heavy = "\n".join([f"background line {i}" for i in range(500)] + ["ERROR final failure line"])
    body = {
        "messages": [
            {"role": "user", "content": "why did it fail?"},
            {"role": "tool", "content": heavy},
        ]
    }

    result = compress_proxy_payload(
        body,
        query="final failure",
        budget_tokens=120,
        retrieval_store=store,
    )

    retrieval = result.receipt.receipts[0]["retrieval"]
    receipt_id = retrieval["receipt_id"]
    span_id = retrieval["span_ids"][0]
    span = store.get_span(receipt_id, span_id)

    assert result.changed
    assert retrieval["span_count"] >= 1
    assert span is not None
    assert "background line" in span.content

    restored = CompressionRetrievalStore(store_path)
    restored_span = restored.get_span(receipt_id, span_id)
    assert restored_span is not None
    assert restored_span.content == span.content


def test_long_lived_reader_observes_another_store_commit(tmp_path) -> None:
    path = tmp_path / "shared.json"
    reader = CompressionRetrievalStore(path)
    writer = CompressionRetrievalStore(path)
    original = "line one\nline two"

    stored = writer.put(
        original_text=original,
        compressed_text="[omitted]",
        receipt={
            "original_tokens": 4,
            "compressed_tokens": 2,
            "omitted_spans": [{"start_line": 1, "end_line": 2}],
        },
    )

    observed = reader.get_span(stored.receipt_id, stored.spans[0].span_id)
    assert observed is not None
    assert observed.content == original


def test_independent_concurrent_writers_preserve_every_payload(tmp_path) -> None:
    store_path = tmp_path / "shared.json"
    workers = 4
    entries = 5
    processes: list[subprocess.Popen[str]] = []
    for worker_id in range(workers):
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _CONCURRENT_WRITER,
                    str(store_path),
                    str(tmp_path),
                    str(worker_id),
                    str(entries),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    for _attempt in range(600):
        if len(list(tmp_path.glob("ready-*"))) == workers:
            break
        if any(process.poll() is not None for process in processes):
            break
        time.sleep(0.01)
    assert len(list(tmp_path.glob("ready-*"))) == workers
    (tmp_path / "start").write_text("start", encoding="utf-8")

    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"

    restored = CompressionRetrievalStore(store_path)
    references = []
    for worker_id in range(workers):
        references.extend(
            json.loads(
                (tmp_path / f"result-{worker_id}.json").read_text(encoding="utf-8")
            )
        )
    assert len(restored.list_receipts()) == workers * entries
    for reference in references:
        span = restored.get_span(reference["receipt_id"], reference["span_id"])
        assert span is not None
        assert span.content == reference["payload"]


def test_independent_retrievers_preserve_every_accounting_debit(tmp_path) -> None:
    store_path = tmp_path / "shared.json"
    original = "first line\nsecond line"
    stored = CompressionRetrievalStore(store_path).put(
        original_text=original,
        compressed_text="[omitted]",
        receipt={
            "original_tokens": 6,
            "compressed_tokens": 2,
            "omitted_spans": [{"start_line": 1, "end_line": 2}],
        },
    )
    workers = 4
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CONCURRENT_RETRIEVER,
                str(store_path),
                str(tmp_path),
                str(worker_id),
                stored.receipt_id,
                stored.spans[0].span_id,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker_id in range(workers)
    ]
    for _attempt in range(600):
        if len(list(tmp_path.glob("retrieval-ready-*"))) == workers:
            break
        if any(process.poll() is not None for process in processes):
            break
        time.sleep(0.01)
    assert len(list(tmp_path.glob("retrieval-ready-*"))) == workers
    (tmp_path / "retrieval-start").write_text("start", encoding="utf-8")

    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"

    restored = CompressionRetrievalStore(store_path)
    span = restored.get_span(stored.receipt_id, stored.spans[0].span_id)
    receipt = restored.get_receipt(stored.receipt_id)
    assert span is not None
    assert receipt is not None
    assert span.retrieval_count == workers
    assert len(span.retrieval_ids) == workers
    assert receipt.retrieval_count == workers


def test_retrieval_store_searches_omitted_spans() -> None:
    store = CompressionRetrievalStore()
    heavy = "\n".join(["payment worker heartbeat" for _ in range(400)] + ["ERROR auth outage INC-777"])
    body = {"messages": [{"role": "tool", "content": heavy}]}

    result = compress_proxy_payload(
        body,
        query="auth outage",
        budget_tokens=120,
        retrieval_store=store,
    )

    assert result.changed
    matches = store.search("payment worker")
    assert matches
    assert "payment worker" in matches[0].content


def test_retrieval_store_search_preserves_hyphenated_id_before_punctuation() -> None:
    store = CompressionRetrievalStore()
    for case_id in ("CASE-OTHER-001", "CASE-TARGET-777"):
        content = f"audit recovery record for {case_id}\nrecovery code RCV-{case_id}"
        store.put(
            original_text=content,
            compressed_text="[omitted]",
            receipt={
                "original_tokens": len(content) // 4,
                "compressed_tokens": 3,
                "omitted_spans": [{"start_line": 1, "end_line": 2}],
            },
        )

    matches = store.search("What recovery code belongs to CASE-TARGET-777?")

    assert matches
    assert "CASE-TARGET-777" in matches[0].content


def test_exact_excerpt_search_bounds_tokens_and_preserves_query_window() -> None:
    store = CompressionRetrievalStore()
    heavy = "\n".join(
        ["payment worker heartbeat" for _ in range(300)]
        + ["audit case CASE-4242 recovery code RCV-998877"]
        + ["trailing worker heartbeat" for _ in range(300)]
    )
    stored = store.put(
        original_text=heavy,
        compressed_text="[omitted]",
        receipt={
            "original_tokens": len(heavy) // 4,
            "compressed_tokens": 3,
            "omitted_spans": [{"start_line": 1, "end_line": 601}],
        },
    )

    matches = store.search_exact_excerpts(
        "CASE-4242 recovery code",
        limit=3,
        max_tokens_per_span=128,
        record_retrieval=True,
        retrieval_id="bounded-search",
    )

    assert len(matches) == 1
    assert "RCV-998877" in matches[0].content
    assert "exact excerpt gap" in matches[0].content
    assert matches[0].retrieved_tokens <= 128
    full = store.get_span(stored.receipt_id, stored.spans[0].span_id)
    assert full is not None
    assert full.content == heavy
    assert full.retrieved_tokens == matches[0].retrieved_tokens


def test_exact_excerpt_uses_conservative_byte_cap_without_tiktoken(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def import_without_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("simulated base install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tiktoken)

    assert _count_o200k_tokens("source-exact ✓") == len(
        "source-exact ✓".encode("utf-8")
    )


def test_exact_excerpt_retrieval_is_idempotent() -> None:
    store = CompressionRetrievalStore()
    content = "lead\n" + "background\n" * 500 + "CASE-99 RCV-12345\n"
    stored = store.put(
        original_text=content,
        compressed_text="lead",
        receipt={
            "original_tokens": len(content) // 4,
            "compressed_tokens": 1,
            "omitted_spans": [{"start_line": 1, "end_line": 502}],
        },
    )
    first = store.retrieve_span_excerpt(
        stored.receipt_id,
        stored.spans[0].span_id,
        query="CASE-99",
        max_tokens=96,
        retrieval_id="same-retry",
    )
    second = store.retrieve_span_excerpt(
        stored.receipt_id,
        stored.spans[0].span_id,
        query="CASE-99",
        max_tokens=96,
        retrieval_id="same-retry",
    )

    assert first is not None and second is not None
    assert first.content == second.content
    persisted = store.get_span(stored.receipt_id, stored.spans[0].span_id)
    assert persisted is not None
    assert persisted.retrieval_count == 1
    assert persisted.retrieved_tokens == first.retrieved_tokens


def test_exact_excerpt_retrieval_rolls_back_when_persistence_fails(
    tmp_path, monkeypatch
) -> None:
    store = CompressionRetrievalStore(tmp_path / "store.json")
    content = "lead\n" + "background\n" * 500 + "CASE-99 RCV-12345\n"
    stored = store.put(
        original_text=content,
        compressed_text="lead",
        receipt={
            "original_tokens": len(content) // 4,
            "compressed_tokens": 1,
            "omitted_spans": [{"start_line": 1, "end_line": 502}],
        },
    )
    monkeypatch.setattr(
        store,
        "_persist",
        lambda: (_ for _ in ()).throw(OSError("simulated excerpt persistence failure")),
    )

    with pytest.raises(OSError, match="simulated excerpt persistence failure"):
        store.retrieve_span_excerpt(
            stored.receipt_id,
            stored.spans[0].span_id,
            query="CASE-99",
            max_tokens=96,
            retrieval_id="failed-excerpt",
        )

    span = store.get_span(stored.receipt_id, stored.spans[0].span_id)
    receipt = store.get_receipt(stored.receipt_id)
    assert span is not None and receipt is not None
    assert span.retrieval_count == 0
    assert span.retrieved_tokens == 0
    assert span.retrieval_ids == []
    assert receipt.retrieval_count == 0


def test_exact_excerpt_search_returns_complete_query_matching_json_object() -> None:
    rows = [
        {
            "audit_case": f"CASE-ORD-{index:04d}",
            "payload": "routine",
            "status": "ok",
        }
        for index in range(100)
    ]
    rows[73] = {
        "audit_case": "CASE-0718-000",
        "payload": "q" * 300 + " RCV-506714411 " + "q" * 300,
        "recovery_code": "RCV-506714411",
        "status": "ok",
    }
    content = json.dumps(rows, indent=2, sort_keys=True)
    store = CompressionRetrievalStore()
    stored = store.put(
        original_text=content,
        compressed_text="[omitted]",
        receipt={
            "original_tokens": len(content) // 4,
            "compressed_tokens": 3,
            "omitted_spans": [{"start_line": 1, "end_line": len(content.splitlines())}],
        },
    )

    matches = store.search_exact_excerpts(
        "recovery code for audit case CASE-0718-000",
        max_tokens_per_span=600,
        record_retrieval=True,
        retrieval_id="json-object",
    )

    assert len(matches) == 1
    parsed = json.loads(matches[0].content)
    assert parsed == rows[73]
    assert matches[0].content in content
    assert matches[0].retrieved_tokens <= 600
    full = store.get_span(stored.receipt_id, stored.spans[0].span_id)
    assert full is not None and full.content == content


def test_exact_excerpt_json_field_projection_falls_back_for_minified_json() -> None:
    rows = [
        {
            "audit_case": "CASE-4242",
            "payload": "q" * 200,
            "recovery_code": "RCV-998877",
        }
    ]
    content = json.dumps(rows, separators=(",", ":"))
    store = CompressionRetrievalStore()
    store.put(
        original_text=content,
        compressed_text="[omitted]",
        receipt={
            "original_tokens": len(content) // 4,
            "compressed_tokens": 3,
            "omitted_spans": [{"start_line": 1, "end_line": 1}],
        },
    )

    matches = store.search_exact_excerpts(
        "recovery code for audit case CASE-4242",
        max_tokens_per_span=600,
    )

    assert len(matches) == 1
    assert json.loads(matches[0].content) == rows[0]
    assert matches[0].content in content


def test_mcp_search_schema_adds_bounded_excerpt_without_new_required_args() -> None:
    server = create_compression_mcp_server()
    tools = asyncio.run(server.list_tools())
    search = next(tool for tool in tools if tool.name == "search_compressed_spans")

    assert search.inputSchema["required"] == ["query"]
    properties = search.inputSchema["properties"]
    assert properties["limit"]["default"] == 5
    assert properties["store_path_override"]["default"] == ""
    assert properties["retrieval_id"]["default"] == ""
    assert properties["max_tokens_per_span"]["default"] == 600


def test_retrieval_accounting_rolls_back_when_persistence_fails(
    tmp_path, monkeypatch
) -> None:
    store = CompressionRetrievalStore(tmp_path / "store.json")
    heavy = "\n".join(
        ["payment worker heartbeat" for _ in range(400)] + ["ERROR auth outage INC-777"]
    )
    result = compress_proxy_payload(
        {"messages": [{"role": "tool", "content": heavy}]},
        query="auth outage",
        budget_tokens=120,
        retrieval_store=store,
    )
    retrieval = result.receipt.receipts[0]["retrieval"]
    receipt_id = retrieval["receipt_id"]
    span_id = retrieval["span_ids"][0]
    monkeypatch.setattr(
        store,
        "_persist",
        lambda: (_ for _ in ()).throw(OSError("simulated disk failure")),
    )

    with pytest.raises(OSError, match="simulated disk failure"):
        store.retrieve_span(receipt_id, span_id, retrieval_id="failed-retrieval")

    span = store.get_span(receipt_id, span_id)
    receipt = store.get_receipt(receipt_id)
    assert span is not None
    assert receipt is not None
    assert span.retrieval_count == 0
    assert span.retrieved_tokens == 0
    assert span.retrieval_ids == []
    assert receipt.retrieval_count == 0


def test_env_proxy_mode_uses_store_when_enabled(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "store.json"
    monkeypatch.setenv("ENTROLY_COMPRESSION_PROXY_MODE", "elc")
    monkeypatch.setenv("ENTROLY_ELC_BUDGET_TOKENS", "120")
    monkeypatch.setenv("ENTROLY_COMPRESSION_STORE", str(store_path))
    heavy = "\n".join(["compile ok" for _ in range(400)] + ["ERROR link failure"])
    body = {"messages": [{"role": "tool", "content": heavy}]}

    result = compress_proxy_payload_from_env(body, query="link failure")

    assert result.changed
    assert store_path.exists()
    assert result.receipt.receipts[0]["retrieval"]["span_count"] >= 1


def test_env_proxy_mode_defaults_to_off(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_COMPRESSION_PROXY_MODE", raising=False)
    heavy = "\n".join(["compile ok" for _ in range(400)] + ["ERROR link failure"])
    body = {"messages": [{"role": "tool", "content": heavy}]}

    result = compress_proxy_payload_from_env(body, query="link failure")

    assert result.changed is False
    assert result.body == body


def test_env_proxy_mode_ignores_invalid_budget_when_off(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_COMPRESSION_PROXY_MODE", raising=False)
    monkeypatch.setenv("ENTROLY_ELC_BUDGET_TOKENS", "not-an-int")
    body = {"messages": [{"role": "tool", "content": "x" * 5000}]}

    result = compress_proxy_payload_from_env(body, query="anything")

    assert result.changed is False
    assert result.body == body
