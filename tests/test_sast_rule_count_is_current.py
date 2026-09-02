"""The advertised rule count must be the rule count.

`scan_for_vulnerabilities` is an MCP tool, so its docstring is not a comment --
it is the interface description an agent reads to decide whether the tool is
worth calling. It advertised a "55-rule engine" while `sast.rs` had grown to
151 rules, understating the engine roughly threefold. Every other statement of
the number in the repository was already correct, so this was one stale copy
rather than a disagreement about the truth.

A hardcoded count is exactly the thing that goes stale, which is why this pins
the docstring to the table rather than to another literal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SAST = Path("entroly-engine/src/sast.rs")
_SERVER = Path("entroly/server.py")
_RULES_MARKER = "static RULES: &[SastRule] = &["


def _rule_ids() -> list[str]:
    """Ids declared inside the RULES table, not merely anywhere in the file."""
    source = _SAST.read_text(encoding="utf-8")
    start = source.index(_RULES_MARKER) + len(_RULES_MARKER) - 1
    depth = 0
    end = start
    for index in range(start, len(source)):
        char = source[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index
                break
    else:  # pragma: no cover - unbalanced table would be a syntax error
        pytest.fail("RULES table is unbalanced")
    return re.findall(r'\bid:\s*"([A-Za-z0-9_.-]+)"', source[start:end])


@pytest.mark.skipif(not _SAST.exists(), reason="engine source not in this tree")
def test_rule_ids_are_unique() -> None:
    ids = _rule_ids()
    assert ids, "no rules parsed; the table shape changed and this guard is blind"
    duplicates = {rule for rule in ids if ids.count(rule) > 1}
    assert not duplicates, f"duplicate rule ids: {sorted(duplicates)}"


@pytest.mark.skipif(not _SAST.exists(), reason="engine source not in this tree")
def test_the_mcp_tool_advertises_the_real_rule_count() -> None:
    count = len(_rule_ids())
    docstring = _SERVER.read_text(encoding="utf-8")

    advertised = re.findall(r"Uses a (\d+)-rule engine", docstring)
    assert advertised, "scan_for_vulnerabilities no longer states a rule count"
    assert len(advertised) == 1, f"several counts advertised: {advertised}"
    assert int(advertised[0]) == count, (
        f"scan_for_vulnerabilities advertises {advertised[0]} rules; sast.rs "
        f"defines {count}. An agent reads that docstring to decide whether the "
        "tool is worth calling."
    )
