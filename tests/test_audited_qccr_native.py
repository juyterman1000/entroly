from __future__ import annotations

import pytest

from entroly.native_status import QCCR_SYMBOLS, native_status

_NATIVE = native_status(QCCR_SYMBOLS)
pytestmark = pytest.mark.skipif(
    not _NATIVE.ok,
    reason="installed entroly_core does not include the current QCCR engine",
)

from entroly.audited_qccr import select_with_audit  # noqa: E402


def test_native_audit_emits_exact_atomic_candidate_receipt() -> None:
    first = "Résumé background. The Dutch name is:\nRhijn."
    second = "Authentication tokens rotate. Authentication failures are logged."
    fragments = [
        {
            "fragment_id": "unicode-1",
            "source": "doc.txt",
            "content": first,
            "start_byte": 0,
            "end_byte": len(first.encode("utf-8")),
            "token_count": 20,
        },
        {
            "fragment_id": "other-1",
            "source": "other.txt",
            "content": second,
            "start_byte": 0,
            "end_byte": len(second.encode("utf-8")),
            "token_count": 20,
        },
    ]
    result = select_with_audit(fragments, 12, "What is the Dutch name?")

    assert result["selection_mode"] == "atomic_audited"
    assert result["emitted_tokens"] <= 12
    assert result["metrics"]["scope"] == "candidate_units"
    assert result["metrics"]["source_span_integrity"] is True
    assert result["candidates"]
    assert any(candidate["selected"] for candidate in result["candidates"])
    assert any(not candidate["selected"] for candidate in result["candidates"])
    assert all(candidate["trimmed"] is False for candidate in result["candidates"])

    selected_text = "\n".join(fragment["content"] for fragment in result["selected"])
    assert "Rhijn" in selected_text

    by_id = {fragment["fragment_id"]: fragment for fragment in fragments}
    for candidate in result["candidates"]:
        source = by_id[candidate["fragment_id"]]
        encoded = source["content"].encode("utf-8")
        start = candidate["start_byte"] - source["start_byte"]
        end = candidate["end_byte"] - source["start_byte"]
        assert encoded[start:end].decode("utf-8")
