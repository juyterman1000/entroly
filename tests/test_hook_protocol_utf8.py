"""The activation hook must receive the user's prompt, not a re-encoding of it.

Root cause these cover (measured on Windows, cp1252 console):

  host writes UTF-8 JSON bytes
  -> sys.stdin.read() decodes them with the ANSI code page
  -> prompt becomes mojibake, or gains lone surrogates
  -> json.loads often still succeeds
  -> selection runs against a query that is not the user's

Two distinct failure classes, both covered below:

  A. surrogate-producing -- a UTF-8 byte that cp1252 leaves undefined
     (0x81/0x8D/0x8F/0x90/0x9D) becomes U+DCxx, and run_hook later raised
     `UnicodeEncodeError: surrogates not allowed` at its prompt_sha256 line.
  B. silent mojibake -- every byte happens to be cp1252-defined, so nothing
     raises and the corrupted query is used. This is the dangerous one, and it
     is why these tests assert exact equality rather than "did not crash".
"""

from __future__ import annotations

import io
import json

import pytest

from entroly.agent_activation import parse_hook_input, read_protocol_stdin

# Spans Latin-1-representable accents, CJK, Devanagari, arrows, box drawing,
# mathematical symbols and an astral-plane emoji.
RICH_PROMPT = (
    "ASCII\n"
    "arrow: →\n"
    "box:\n┌──────┐\n"
    "│ test │\n└──────┘\n"
    "math: Φ λ ≥ ≤\n"
    "emoji: \U0001f9e0\n"
    "café\n"
    "中文\n"
    "हिन्दी\n"
)


class _WindowsStdin:
    """A text stdin that decodes with cp1252, as Windows actually supplies.

    ``errors='surrogateescape'`` matches the measured interpreter state. The
    ``buffer`` attribute holds the true host bytes, which is what
    ``read_protocol_stdin`` is expected to use.
    """

    def __init__(self, payload: bytes) -> None:
        self.buffer = io.BytesIO(payload)
        self._text = io.TextIOWrapper(
            io.BytesIO(payload), encoding="cp1252", errors="surrogateescape"
        )

    def read(self) -> str:  # the old, broken path
        return self._text.read()


def _payload(prompt: str) -> bytes:
    """Serialize as the real host does: raw UTF-8, not \\uXXXX escapes.

    ``ensure_ascii=False`` is load-bearing. With the default True, json.dumps
    emits pure-ASCII escapes, cp1252 decodes them losslessly, and none of the
    corruption below reproduces -- the fixture would silently test nothing. The
    live plugin launcher was observed sending raw UTF-8 bytes.
    """
    return json.dumps(
        {"hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": "."},
        ensure_ascii=False,
    ).encode("utf-8")


def test_u1_exact_unicode_preserved_through_the_protocol_boundary():
    """U1 + U4 + U5: byte-for-byte round trip, including emoji and scripts."""
    stdin = _WindowsStdin(_payload(RICH_PROMPT))
    parsed = parse_hook_input(read_protocol_stdin(stdin))
    assert parsed["prompt"] == RICH_PROMPT
    # No lone surrogates may survive into the query.
    assert not [c for c in parsed["prompt"] if 0xDC80 <= ord(c) <= 0xDCFF]
    # And the prompt must still be encodable, which is what run_hook does.
    parsed["prompt"].encode("utf-8")


def test_u2_surrogate_producing_input_no_longer_corrupts_or_raises():
    """U2: the class that produced the visible UnicodeEncodeError.

    The culprit is U+2510 ``┐``, which encodes as ``E2 94 90``. 0x90 is one of
    the five bytes cp1252 leaves undefined (0x81/0x8D/0x8F/0x90/0x9D), so
    surrogateescape mapped it to U+DC90 and run_hook's ``query.encode("utf-8")``
    raised ``surrogates not allowed``.

    Worth recording precisely, because the obvious suspects are innocent:
    U+1F9E0 (F0 9F A7 A0), U+2192 (E2 86 92), U+03A6 (CE A6) and U+4E2D
    (E4 B8 AD) are all composed entirely of cp1252-*defined* bytes and cannot
    produce a surrogate. Only the box-drawing down-and-left corner in the
    original fixture did.
    """
    prompt = "box ┐ tail"
    raw = _payload(prompt)
    assert 0x90 in raw, "fixture must contain a cp1252-undefined byte"

    stdin = _WindowsStdin(raw)

    # Demonstrate the old behaviour on the same bytes, so this test fails loudly
    # if someone restores sys.stdin.read().
    legacy = stdin.read()
    assert [c for c in legacy if 0xDC80 <= ord(c) <= 0xDCFF], (
        "fixture no longer reproduces surrogate escaping"
    )
    with pytest.raises(UnicodeEncodeError):
        legacy.encode("utf-8")

    stdin = _WindowsStdin(raw)
    assert parse_hook_input(read_protocol_stdin(stdin))["prompt"] == prompt


def test_u3_silent_mojibake_input_is_fixed():
    """U3: the dangerous class -- no exception, wrong query.

    Every UTF-8 byte of these characters is cp1252-defined, so the old path
    parsed successfully and selection used a different string.
    """
    prompt = "café → 中文"
    raw = _payload(prompt)
    assert not any(b in (0x81, 0x8D, 0x8F, 0x90, 0x9D) for b in raw), (
        "fixture must avoid cp1252-undefined bytes to exercise the silent path"
    )

    stdin = _WindowsStdin(raw)
    legacy_prompt = json.loads(stdin.read())["prompt"]
    assert legacy_prompt != prompt, "fixture no longer reproduces mojibake"

    stdin = _WindowsStdin(raw)
    assert parse_hook_input(read_protocol_stdin(stdin))["prompt"] == prompt


def test_u6_invalid_utf8_fails_explicitly_rather_than_substituting():
    """U6: a malformed payload must not become replacement characters."""

    class _Binary:
        def __init__(self, payload: bytes) -> None:
            self.buffer = io.BytesIO(payload)

    bad = b'{"prompt": "caf\xff\xfe"}'
    with pytest.raises(UnicodeDecodeError):
        read_protocol_stdin(_Binary(bad))


def test_u8_ascii_behaviour_unchanged():
    """U8: the common path must be untouched."""
    prompt = "plain ascii prompt"
    stdin = _WindowsStdin(_payload(prompt))
    assert parse_hook_input(read_protocol_stdin(stdin))["prompt"] == prompt


def test_text_only_stdin_without_buffer_still_works():
    """Embedding hosts and tests may install a bufferless text stdin.

    io.StringIO has no `.buffer`; that text is already decoded, so it is taken
    as given rather than round-tripped.
    """
    payload = json.dumps({"prompt": RICH_PROMPT})
    assert parse_hook_input(read_protocol_stdin(io.StringIO(payload)))["prompt"] == (
        RICH_PROMPT
    )
