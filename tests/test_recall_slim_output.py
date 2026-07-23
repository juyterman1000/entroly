"""recall_relevant returns a slim ranked pointer list, not 90KB of bodies.

Dogfooding: a top_k=8 recall returned ~90KB of full fragment bodies, overflowed
the MCP result cap, and spilled to a file — useless as a live context layer.
The slimmer keeps what an agent needs to locate code (source + score + snippet)
and drops the bodies unless full=True.
"""

from __future__ import annotations

import json

from entroly.server import _slim_recall_results


def _fake_results(n: int = 8, body_chars: int = 8000) -> list[dict]:
    return [
        {
            "source": f"file:src/mod_{i}.py",
            "content": "x " * (body_chars // 2),
            "fragment_id": f"f{i}",
            "relevance": 100.0 - i,
            "token_count": 500,
        }
        for i in range(n)
    ]


def test_slim_drops_bodies_and_keeps_pointers() -> None:
    slim = _slim_recall_results(_fake_results())
    assert len(slim) == 8
    first = slim[0]
    assert first["rank"] == 1
    assert first["source"] == "file:src/mod_0.py"
    assert first["fragment_id"] == "f0"
    assert first["score"] == 100.0
    assert len(first["snippet"]) <= 200
    assert "content" not in first  # the body is gone


def test_slim_output_is_orders_of_magnitude_smaller() -> None:
    results = _fake_results(n=8, body_chars=12000)
    full = json.dumps({"results": results})
    slim = json.dumps({"results": _slim_recall_results(results)})
    assert len(slim) * 10 < len(full)  # >= 10x smaller
    assert len(slim) < 8000  # comfortably under the tool result cap


def test_slim_handles_field_variants_and_non_dicts() -> None:
    mixed = [
        {"source_path": "a.py", "text": "alpha", "id": "x1", "score": 5},
        "not-a-dict",
        {"path": "b.py", "content": "beta beta", "relevance_score": 3.2},
    ]
    slim = _slim_recall_results(mixed)  # type: ignore[arg-type]
    assert [e["source"] for e in slim] == ["a.py", "b.py"]
    assert slim[0]["fragment_id"] == "x1"
    assert slim[0]["score"] == 5.0
    assert slim[1]["rank"] == 2  # rank counts kept, non-dict skipped
