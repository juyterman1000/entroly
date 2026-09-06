"""An omitted-evidence preview must not misrepresent its source.

`_preview` was `" ".join(text.split())`, collapsing every newline, tab and
indent. The omitted-evidence explorer renders previews in a block styled
`white-space: pre-wrap`, immediately beside selected fragments that keep their
line breaks -- so a Python function arrived as one run-on line, no longer valid
in its own language, and read as corrupted next to the intact code above it.

Omitted evidence exists so a user can audit what the selector left out. A
preview that silently reflows the source is a worse answer than a shorter,
faithful one. Contexts that genuinely need one line flatten at render time.
"""
from __future__ import annotations

from entroly.context_receipts.receipts import _one_line
from entroly.context_receipts.selection import _preview

CODE = (
    "def render_svg_icon(name, size):\n"
    "    palette = load_palette(name)\n"
    "    return draw(palette, size)\n"
    "\n"
    "def load_palette(name):\n"
    "    return PALETTES.get(name, DEFAULT)\n"
)


def test_preview_keeps_line_structure():
    preview = _preview(CODE)
    assert "\n" in preview, "preview was flattened to a single line"
    assert "    palette = load_palette(name)" in preview, (
        "indentation was stripped; the excerpt no longer reflects the source"
    )
    assert preview.count("\n") == 4, f"unexpected line count: {preview!r}"


def test_preview_drops_blank_lines_and_trailing_whitespace():
    """Faithful to structure, not to incidental whitespace."""
    preview = _preview("a = 1   \n\n\n\nb = 2\t\n")
    assert preview == "a = 1\nb = 2"


def test_preview_is_still_bounded():
    """Preserving newlines must not let a preview grow without limit."""
    preview = _preview("\n".join(f"line {i} with some content" for i in range(200)))
    assert len(preview) <= 240
    assert preview.endswith("...")


def test_preview_of_single_line_text_is_unchanged():
    assert _preview("just one line") == "just one line"


def test_markdown_contexts_flatten_at_render_time():
    """A `- Preview:` bullet needs one line; that is a rendering choice."""
    rendered = _one_line(_preview(CODE))
    assert "\n" not in rendered
    assert "def render_svg_icon(name, size): palette = load_palette(name)" in rendered


def test_preview_never_fabricates_content():
    """Whatever survives truncation must be a substring of the source.

    Cheap to state, and it is the property that makes a preview quotable in an
    audit at all.
    """
    for text in (CODE, "x" * 500, "a\nb\nc", "  leading\n\ttabbed  \n"):
        preview = _preview(text).removesuffix("...")
        for line in preview.splitlines():
            assert line in text, f"preview line {line!r} is not present in the source"
