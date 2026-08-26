"""No MCP tool may return a response too large for an agent to receive.

Dogfooding this repository found three tools overflowing the MCP result cap in
a single sitting. In every case the engine was right and the wire boundary was
wrong:

    optimize_context          378,545 chars at an 8,000-token budget
                              (selected/selected_fragments byte-identical
                              duplicates, plus uncompacted provenance)
                              -- the selection itself used 7,004 tokens
    analyze_codebase_health    71,805 chars
                              (four unbounded finding lists, no wire boundary)
    smart_read                 the inverse: 23 tokens of a requested 800

Slimming had been applied to recall_relevant and partly to optimize_context,
but never audited across every tool, so each new overflow was found by an agent
hitting it rather than by CI. These tests are that audit.
"""

from __future__ import annotations

import json

import pytest

NEWLINE = chr(10)
TAB_CHAR = chr(9)

# The MCP result cap is ~25k tokens; at ~4 chars/token that is ~100k characters,
# beyond which the response is rejected before the agent sees any of it. The
# observed failure was a 520,569-char optimize_context result.
#
# This is the real ceiling rather than a comfortable-looking round number: a
# threshold tighter than the platform's would fail builds for responses that
# actually work, and one looser would let a genuinely unusable response through.
MAX_WIRE_CHARS = 100_000


def _fragment(i: int) -> dict:
    return {
        "id": f"f{i}",
        "source": f"file:module_{i}.py",
        # Sized so the whole selection lands near a realistic 8,000-token
        # budget (~4 chars/token). 395 fragments of 400 chars would be ~40k
        # tokens of content, which no 8k-budget selection could ever contain,
        # so asserting the wire size against that would test a shape the tool
        # never produces.
        "content": "x" * 72,
        "token_count": 18,
        "relevance": 0.8,
        "content_sha256": "deadbeef",
        "retrieval_handle": f"h{i}",
        "entropy_score": 0.5,
        "variant": "full",
    }


def test_optimize_result_fits_the_wire_after_compaction() -> None:
    from entroly.provenance import compact_optimize_result_for_wire

    selection = [_fragment(i) for i in range(395)]
    result = {
        "selected_fragments": selection,
        "selected": selection,
        "provenance": {"fragments": [_fragment(i) for i in range(395)], "query": "q"},
    }
    compact_optimize_result_for_wire(result)
    size = len(json.dumps(result))
    assert size <= MAX_WIRE_CHARS, (
        f"optimize_context payload is {size:,} chars, over the {MAX_WIRE_CHARS:,} "
        "wire budget; an agent receives an error instead of context"
    )


def test_health_report_fits_the_wire_after_capping() -> None:
    from entroly.server import _compact_health_report_for_wire

    raw = json.dumps({
        "health_grade": "C",
        "code_health_score": 79.2,
        "summary": "s",
        "top_recommendation": "r",
        "clone_pairs": [{"source_a": f"a{i}.py", "source_b": f"b{i}.py",
                         "similarity": 0.9, "clone_type": "Type-1/2"} for i in range(83)],
        "dead_symbols": [{"symbol": f"sym_{i}", "source": f"f{i}.py"} for i in range(50)],
        "god_files": [{"source": f"g{i}.py", "reverse_deps": 100} for i in range(40)],
        "arch_violations": [{"from": f"a{i}", "to": f"b{i}"} for i in range(23)],
    })
    compacted = _compact_health_report_for_wire(raw)
    size = len(compacted)
    assert size <= MAX_WIRE_CHARS, (
        f"health report is {size:,} chars, over the {MAX_WIRE_CHARS:,} wire budget"
    )
    report = json.loads(compacted)
    assert report["health_grade"] == "C", "the diagnosis must survive the cap"
    assert report["truncated"]["total_counts"]["clone_pairs"] == 83, (
        "real totals must stay visible; a capped list must not hide how much was found"
    )


# ── The opposite failure: a tool that under-uses its budget ─────────────────


@pytest.mark.parametrize(
    ("path", "query", "budget"),
    [
        ("entroly/provenance.py", "how does wire compaction strip fragments", 800),
        ("entroly/auto_index.py", "chunk oversized source files into parts", 1500),
    ],
)
def test_smart_read_uses_a_meaningful_share_of_its_budget(path, query, budget) -> None:
    """A read that skips everything is as useless as one that overflows.

    `entroly/provenance.py` at an 800-token budget returned 23 tokens -- 3% --
    with 7 of its 8 blocks skipped, because SKIP had no entry in the
    under-budget upgrade ladder and so was unreachable however much budget was
    free.
    """
    from pathlib import Path

    from entroly.semantic_resolution import resolve

    source = Path(path).read_text(encoding="utf-8")
    result = resolve(source, query=query, budget=budget, file_path=path)
    used = result.total_tokens
    assert used <= budget, f"{path} returned {used} tokens, over its {budget} budget"
    assert used >= budget * 0.5, (
        f"{path} used only {used} of {budget} tokens "
        f"({used * 100 // budget}%); resolutions were {result.resolution_counts}"
    )


def test_smart_read_gives_full_detail_to_the_block_the_query_names() -> None:
    """The documented contract: "Blocks matching the query -> FULL".

    Before the ladder fix, `chunk_oversized_source` came back as a bare
    signature for a query naming it almost verbatim, while unrelated functions
    were rendered in full.
    """
    from pathlib import Path

    from entroly.semantic_resolution import resolve

    path = "entroly/auto_index.py"
    source = Path(path).read_text(encoding="utf-8")
    result = resolve(
        source,
        query="chunk oversized source files into parts",
        budget=1500,
        file_path=path,
    )
    marker = "def chunk_oversized_source"
    assert marker in result.output, "the query-named block must appear at all"
    segment = result.output[result.output.index(marker):][:400]
    header, _, remainder = segment.partition(NEWLINE)
    # A signature-only render stops at the return annotation; a full render
    # continues into an indented body.
    assert remainder.startswith((' ', TAB_CHAR)), (
        'the block the query names must render as source, not a bare '
        f'signature; header was {header!r}'
    )
