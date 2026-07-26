"""Build the verified PR181 replacement branch from the latest-main merge ref.

Temporary stabilization helper. The workflow removes this file before creating the
final branch, so it cannot ship as product surface.
"""
from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement target, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "entroly/compression_retrieval_store.py",
    "        lines = original_text.splitlines()\n",
    "        # Preserve the original line terminators. Recovery is an integrity\n"
    "        # boundary: CRLF/LF and a trailing newline are source bytes, not\n"
    "        # presentation details.\n"
    "        lines = original_text.splitlines(keepends=True)\n",
)
replace_once(
    "entroly/compression_retrieval_store.py",
    '            content = "\\n".join(lines[start - 1 : end])\n',
    '            content = "".join(lines[start - 1 : end])\n',
)
replace_once(
    "entroly/session_rescue.py",
    '''            compacted, message_receipts = self._compact_message(
                message,
                query=query,
            )
''',
    '''            # Frozen historical bytes must not depend on the newest user
            # query. Otherwise a daemon restart or state eviction can
            # recompress the same old tool output differently and churn the
            # provider-cache prefix. Session rescue is structural, so use a
            # stable query-independent compression contract.
            compacted, message_receipts = self._compact_message(
                message,
                query="",
            )
''',
)
replace_once(
    "entroly/session_rescue.py",
    '''        span_ids = ",".join(span.span_id for span in stored.spans)
        marker = (
            "[entroly-recovery: "
            f"receipt={stored.receipt_id}; spans={span_ids or 'none'}; "
            f"original_sha256={stored.original_hash}]"
        )
''',
    '''        if len(stored.spans) != 1:
            raise RuntimeError(
                "session rescue requires one exact full-original recovery span"
            )
        span_id = stored.spans[0].span_id
        marker = f"[entroly-recovery:{stored.receipt_id}:{span_id}]"
''',
)

session_test_path = Path("tests/test_session_rescue.py")
session_tests = session_test_path.read_text(encoding="utf-8")
marker_anchor = '    assert stored.spans[0].content == tool["content"]\n'
marker_test = '''    assert (
        f"[entroly-recovery:{stored.receipt_id}:{stored.spans[0].span_id}]"
        in result.messages[1]["content"]
    )
'''
if session_tests.count(marker_anchor) != 1:
    raise SystemExit("session-rescue marker-test anchor changed")
session_tests = session_tests.replace(marker_anchor, marker_anchor + marker_test, 1)

restart_test = '''

def test_restart_with_different_query_keeps_rescued_prefix_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import entroly.session_rescue as rescue_module

    class _Receipt:
        def as_dict(self) -> dict[str, object]:
            return {
                "original_tokens": 500,
                "compressed_tokens": 5,
                "omitted_spans": [],
            }

    class _Result:
        changed = True
        receipt = _Receipt()

        def __init__(self, query: str) -> None:
            self.compressed = f"structural-summary query={query}"

        def with_receipt_header(self) -> str:
            return self.compressed

    def query_sensitive_compressor(
        _text: str, *, query: str, budget_tokens: int, min_savings: float
    ) -> _Result:
        assert budget_tokens > 0
        assert min_savings > 0
        return _Result(query)

    monkeypatch.setattr(
        rescue_module, "compress_evidence_locked", query_sensitive_compressor
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    policy = {
        "soft_watermark": 0.20,
        "hard_watermark": 0.30,
        "target_watermark": 0.10,
        "failure_watermark": 0.95,
        "loop_min_watermark": 0.05,
        "tail_messages": 2,
    }
    first, _ = _controller(first_dir, **policy)
    restarted, _ = _controller(second_dir, **policy)
    messages = [
        {"role": "system", "content": "stable policy"},
        {"role": "tool", "content": "old tool output " * 500},
        {"role": "assistant", "content": "checking"},
        {"role": "user", "content": "continue"},
    ]

    before_restart = first.rescue(
        "conv", messages, context_window=4_000, query="payment failure"
    )
    after_restart = restarted.rescue(
        "conv", messages, context_window=4_000, query="unrelated auth question"
    )

    assert before_restart.messages[1] == after_restart.messages[1]
    assert before_restart.recovery_receipts == after_restart.recovery_receipts
    assert "query=" in before_restart.messages[1]["content"]
    assert "payment failure" not in before_restart.messages[1]["content"]
    assert "unrelated auth question" not in after_restart.messages[1]["content"]
'''
restart_anchor = "\n\ndef test_loop_signal_triggers_before_hard_watermark"
if session_tests.count(restart_anchor) != 1:
    raise SystemExit("session-rescue restart-test anchor changed")
session_tests = session_tests.replace(restart_anchor, restart_test + restart_anchor, 1)
session_test_path.write_text(session_tests, encoding="utf-8")

store_test_path = Path("tests/test_compression_retrieval_store.py")
store_tests = store_test_path.read_text(encoding="utf-8")
exact_test = '''

@pytest.mark.parametrize(
    "original",
    [
        "line one\\nline two\\n",
        "line one\\r\\nline two\\r\\n",
    ],
)
def test_full_span_recovery_preserves_exact_line_endings(
    tmp_path, original: str
) -> None:
    path = tmp_path / "exact-line-endings.json"
    stored = CompressionRetrievalStore(path).put(
        original_text=original,
        compressed_text="[omitted]",
        receipt={
            "original_tokens": 10,
            "compressed_tokens": 2,
            "omitted_spans": [{"start_line": 1, "end_line": 2}],
        },
    )

    assert stored.spans[0].content == original
    restored = CompressionRetrievalStore(path).get_span(
        stored.receipt_id, stored.spans[0].span_id
    )
    assert restored is not None
    assert restored.content == original
'''
exact_anchor = "\n\ndef test_recovery_store_byte_limit_fails_before_commit"
if store_tests.count(exact_anchor) != 1:
    raise SystemExit("retrieval-store exact-test anchor changed")
store_tests = store_tests.replace(exact_anchor, exact_test + exact_anchor, 1)
store_test_path.write_text(store_tests, encoding="utf-8")
