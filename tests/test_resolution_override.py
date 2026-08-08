"""A pinned resolution must be honoured exactly, or reported — never quietly changed.

Automatic resolution selection is the right default, but it cannot be right for
every question: measured on this repository, a signature-level view answered
12/12 questions whose evidence lives in a signature and 0/20 whose evidence
lives in a function body. A caller that knows which kind of question it is
asking needs a way to say so.

The dangerous failure is not "too many tokens" — it is returning a lower
resolution than the caller asked for while reporting success, because the caller
then reasons about source it never received.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from entroly.semantic_resolution import Resolution, resolve

SOURCE = '''\
"""Module docstring."""


def alpha(a, b):
    """Add two numbers."""
    total = a + b
    for _ in range(10):
        total += 1
    return total


def beta(x):
    """Multiply by two."""
    scratch = x * 2
    scratch += 1
    return scratch


class Gamma:
    """A class."""

    def method(self, value):
        """Do a thing."""
        return value + 1
'''


def test_default_is_still_automatic() -> None:
    result = resolve(SOURCE, query="alpha", budget=1000)
    assert result.forced_resolution is None


def test_full_returns_the_exact_original_text() -> None:
    """FULL must preserve text outside extracted blocks and final newlines."""
    source = (
        '"""Module docstring."""\r\n'
        "import os\r\n"
        "SETTING = 7\r\n"
        "# comment before the function\r\n"
        "\r\n"
        "def alpha():\r\n"
        "    return os.name\r\n"
    )
    result = resolve(source, query="", budget=1000, resolution=Resolution.FULL)

    assert result.forced_resolution == "full"
    assert result.resolution_counts == {"full": 1}
    assert result.output == source


def test_low_returns_only_stubs() -> None:
    result = resolve(SOURCE, query="", budget=1000, resolution=Resolution.LOW)
    assert set(result.resolution_counts) == {"low"}
    assert "total += 1" not in result.output
    assert "alpha" in result.output


def test_pinned_resolution_is_not_demoted_by_a_tiny_budget() -> None:
    """The load-bearing property.

    A caller asking for FULL under an impossible budget must receive FULL and
    be told it overran — not a silently downgraded MEDIUM that looks like a
    successful read.
    """
    result = resolve(SOURCE, query="", budget=1, resolution=Resolution.FULL)

    assert set(result.resolution_counts) == {"full"}
    assert "total += 1" in result.output
    assert result.over_budget is True
    assert result.total_tokens > result.budget


def test_automatic_mode_still_demotes_to_fit() -> None:
    """The override must not disturb existing budget behaviour."""
    result = resolve(SOURCE, query="", budget=1)
    assert result.forced_resolution is None
    assert result.resolution_counts.get("full", 0) == 0


def test_over_budget_is_false_when_it_fits() -> None:
    result = resolve(SOURCE, query="", budget=100_000, resolution=Resolution.FULL)
    assert result.over_budget is False


@pytest.mark.parametrize("level", ["full", "medium", "low"])
def test_every_documented_level_is_accepted(level: str) -> None:
    result = resolve(SOURCE, query="", budget=1000, resolution=level)
    assert result.forced_resolution == level


def test_diff_requires_a_baseline_instead_of_silently_returning_stubs() -> None:
    with pytest.raises(ValueError, match="requires previous_source"):
        resolve(SOURCE, resolution=Resolution.DIFF)


def test_diff_captures_top_level_additions_and_deletions() -> None:
    previous = "import old_name\n\ndef keep():\n    return 1\n\ndef removed():\n    return 2\n"
    current = "import new_name\n\ndef keep():\n    return 1\n\nADDED = 3\n"

    result = resolve(
        current,
        file_path="sample.py",
        resolution=Resolution.DIFF,
        previous_source=previous,
    )

    assert result.forced_resolution == "diff"
    assert "-import old_name" in result.output
    assert "+import new_name" in result.output
    assert "-def removed():" in result.output
    assert "+ADDED = 3" in result.output


def test_exact_line_range_is_inclusive_and_preserves_newlines() -> None:
    source = "one\r\ntwo\r\nthree\r\nfour"
    result = resolve(source, line_start=2, line_end=3)

    assert result.output == "two\r\nthree\r\n"
    assert result.line_range == (2, 3)
    assert result.resolution_counts == {"lines": 1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"line_start": 1}, "provided together"),
        ({"line_start": 0, "line_end": 1}, "1 <= line_start"),
        ({"line_start": 2, "line_end": 1}, "1 <= line_start"),
        ({"line_start": 1, "line_end": 99}, "exceeds file length"),
        (
            {"line_start": 1, "line_end": 1, "resolution": "full"},
            "cannot be combined",
        ),
    ],
)
def test_invalid_line_ranges_fail_visibly(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve("one\ntwo\n", **kwargs)


def test_smart_read_wires_exact_full_diff_and_line_ranges(tmp_path, monkeypatch) -> None:
    from entroly.server import create_mcp_server

    source = "TOP = 1\r\n# keep this comment\r\ndef alpha():\r\n    return TOP\r\n"
    source_path = tmp_path / "sample.py"
    source_path.write_bytes(source.encode("utf-8"))
    monkeypatch.setenv("ENTROLY_SOURCE", str(tmp_path))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))

    mcp, _ = create_mcp_server(allowed_tools={"smart_read"})
    smart_read_tool = mcp._tool_manager._tools["smart_read"]
    assert "ctx" not in smart_read_tool.parameters["properties"]
    assert {"resolution", "previous_source", "line_start", "line_end", "fresh"} <= set(
        smart_read_tool.parameters["properties"]
    )
    smart_read = smart_read_tool.fn
    ctx = SimpleNamespace(session=object(), client_id="primary")

    full = smart_read(str(source_path), ctx, resolution="full")
    assert full == source

    repeated = smart_read(str(source_path), ctx, resolution="full")
    assert repeated.startswith("~")
    assert repeated[1:].isdigit()

    line_range = smart_read(str(source_path), ctx, line_start=2, line_end=3)
    assert line_range == "# keep this comment\r\ndef alpha():\r\n"

    previous = source.replace("TOP = 1", "TOP = 0")
    diff = json.loads(
        smart_read(
            str(source_path),
            ctx,
            resolution="diff",
            previous_source=previous,
        )
    )
    assert "-TOP = 0" in diff["output"]
    assert "+TOP = 1" in diff["output"]

    missing_baseline = json.loads(
        smart_read(str(source_path), ctx, resolution="diff")
    )
    assert "requires previous_source" in missing_baseline["error"]

    separate_agent = SimpleNamespace(session=object(), client_id="subagent")
    isolated = smart_read(str(source_path), separate_agent, resolution="full")
    assert isolated == source

    # Custom MCP embeddings may omit a session object. Distinct Context
    # instances must still never share read-delivery history.
    sessionless_a = SimpleNamespace(session=None, client_id="embedded")
    sessionless_b = SimpleNamespace(session=None, client_id="embedded")
    assert smart_read(str(source_path), sessionless_a, resolution="full") == source
    assert smart_read(str(source_path), sessionless_b, resolution="full") == source

    refreshed = smart_read(str(source_path), ctx, resolution="full", fresh=True)
    assert refreshed == source


def test_skip_is_rejected_rather_than_returning_an_empty_document() -> None:
    with pytest.raises(ValueError, match="full/medium/diff/structure/low"):
        resolve(SOURCE, query="", budget=1000, resolution="skip")


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="full/medium/diff/structure/low"):
        resolve(SOURCE, query="", budget=1000, resolution="signatures")
