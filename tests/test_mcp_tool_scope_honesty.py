"""A tool's scope must read the same wherever a client learns about it.

Found by dogfooding. ``locate_evidence`` was called expecting it to search the
repository, because the server-level instructions read:

    "read_source_file for a known workspace path, locate_evidence for exact
     source spans answering a natural-language query, remember_fragment for
     caller-supplied text"

Attaching "caller-supplied text" to ``remember_fragment`` and withholding it
from ``locate_evidence`` licenses exactly one inference: that
``locate_evidence`` reads the workspace. It does not -- ``text`` is its first
required parameter and there is no index behind it.

The per-tool description in the public surface was already correct ("Locate
exact passages and sentence offsets in caller-supplied text"), which is what
makes this worth a test rather than a one-line edit. The two descriptions sat
thousands of lines apart in one file and disagreed, and the one a client
receives as ambient context was the wrong one. Nothing compared them.

The tool's design is sound and is not changed here: a bounded, caller-supplied
corpus with exact offsets and abstention on invalid ranker output is a
deliberate contract, not a missing feature. Only the published scope is
corrected, and pinned.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1] / "entroly" / "server.py"


@pytest.fixture(scope="module")
def source() -> str:
    if not SERVER.is_file():
        pytest.skip(f"server module is absent: {SERVER}")
    return SERVER.read_text(encoding="utf-8")


def test_the_server_instructions_do_not_imply_a_repository_search(source: str) -> None:
    from entroly.server import _PUBLIC_MCP_INSTRUCTIONS

    instructions = _PUBLIC_MCP_INSTRUCTIONS
    assert "locate_evidence" in instructions

    # Bind the claim to locate_evidence's own clause. Splitting on sentences
    # put all four tools in one span, so "caller-supplied" anywhere in it
    # satisfied the assertion -- including when, as before, the phrase belonged
    # to remember_fragment and was the very thing being withheld here.
    start = instructions.index("locate_evidence")
    end = instructions.find("remember_fragment", start)
    clause = instructions[start : end if end != -1 else len(instructions)]

    assert "pass" in clause or "supplied" in clause, (
        "locate_evidence's own clause must say the caller supplies the text. "
        "Saying it for remember_fragment and withholding it here is what "
        "implied a workspace search.\n"
        f"  clause: {clause!r}"
    )
    assert "never reads the workspace" in instructions, (
        "state the boundary outright -- the obvious reading of this tool's "
        "name and purpose is the wrong one"
    )


def test_the_docstring_agrees_with_the_published_description(source: str) -> None:
    """The full profile sees the docstring; the public profile sees the override."""
    start = source.find("def locate_evidence(")
    assert start != -1, "locate_evidence was not found; re-point this guard"
    docstring = source[start : start + 2000]

    assert "does not read the workspace" in docstring, (
        "a full-profile client sees only the docstring, so the boundary has to "
        "be stated there too"
    )

    # The public override, which was already correct and must stay that way.
    assert (
        "Locate exact passages and sentence offsets in caller-supplied text."
        in source
    )


def test_text_is_a_required_parameter_not_an_optional_corpus(source: str) -> None:
    """If this ever gains a default, the documented contract changes with it."""
    start = source.find("def locate_evidence(")
    signature = source[start : source.find(")", start)]

    assert "text: str," in signature, (
        "text is the corpus and must stay required; a default would turn a "
        "missing-corpus error into a silent empty result"
    )
    assert "text: str = " not in signature
