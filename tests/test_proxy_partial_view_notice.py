"""A lossy tool-result view must say so.

`proxy_transform` picks the cheapest codec rendering of a tool result. Every
rendering other than ``*.full`` drops content, and the recovery store they
reference is discarded when the call returns. A source file read through a
tool reached the agent as signatures with every body deleted and nothing
saying anything was missing; the agent then edits code it cannot see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.codec_ablation import FIXTURES
from entroly.proxy_transform import compress_tool_output

NOTICE = "[entroly: partial "
REAL_MODULE = Path(__file__).resolve().parent.parent / "entroly" / "session_rescue.py"


def _code_with_bodies() -> str:
    lines = ["def check():"]
    for i in range(60):
        lines.append(f"    if a{i} or b{i} and not c{i}:")
        lines.append(f"        return x{i} or y{i}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "content",
    [_code_with_bodies(), REAL_MODULE.read_text(encoding="utf-8")[:6000]],
    ids=["function_bodies", "real_module"],
)
def test_code_skeleton_announces_what_it_dropped(content: str) -> None:
    compressed, kind, savings = compress_tool_output(content)

    assert kind == "codec"
    first_line = compressed.splitlines()[0]
    assert first_line.startswith(NOTICE)
    assert "code.skeleton" in first_line
    assert "Re-read the file or re-run the command" in first_line
    assert savings > 0.10


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_lossy_codec_view_carries_the_notice(name: str) -> None:
    text, _required = FIXTURES[name]()
    compressed, kind, _ = compress_tool_output(text)

    if kind == "codec" and compressed != text:
        assert compressed.startswith(NOTICE), f"{name}: lossy view without a notice"


def test_notice_is_byte_stable_for_the_prefix_cache() -> None:
    content = _code_with_bodies()
    assert compress_tool_output(content) == compress_tool_output(content)


def test_uncompressed_output_carries_no_notice() -> None:
    short = "ok\n"
    compressed, kind, _ = compress_tool_output(short)
    assert (compressed, kind) == (short, "none")


def test_notice_never_makes_the_result_larger(monkeypatch) -> None:
    import entroly.proxy_transform as proxy_transform

    # A form that saves less than the notice costs must not be emitted.
    content = _code_with_bodies()
    assert compress_tool_output(content)[1] == "codec"
    monkeypatch.setattr(
        proxy_transform,
        "_partial_view_notice",
        lambda best, reps: "N" * (len(content) + 1),
    )
    compressed, kind, _ = compress_tool_output(content)
    assert kind != "codec"
    assert len(compressed) <= len(content)
