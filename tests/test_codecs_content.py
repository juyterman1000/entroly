"""Code, document, conversation and schema codecs (5.4-5.8).

Each is held to what its content is actually read for, not to a ratio:

* code -- a body without its imports and call surface cannot be reasoned about
* documents -- a topically relevant span is not automatically the answer-bearing
  one, so neighbours are protected; an unattributed claim is not evidence
* conversation -- standing instructions bind later turns, and the cache-hot
  prefix must survive byte-identical or every later request pays a cache miss
* schema -- required fields, types and enums are the contract; prose is not

Routing is tested too. ShellCodec used to claim Python source, because its
prompt pattern `^\\s*[$#>]\\s+\\S` matched every `# comment` line, and CodeCodec's
confidence scaled with declaration DENSITY so a large well-factored module lost
the race. Both were wrong and both are pinned below.
"""

from __future__ import annotations

import json

import pytest

from entroly.codec import RecoveryStore, verify_all
from entroly.codecs_builtin import default_registry
from entroly.codecs_content import (
    CodeCodec,
    ConversationCodec,
    DocumentCodec,
    SchemaCodec,
)

SOURCE = '''import os
import sys
from pathlib import Path


def load_config(path):
    """Read the config."""
    raw = Path(path).read_text()
    if not raw:
        raise ValueError("config is invalid or empty")
    parsed = parse(raw)
    for key in parsed:
        validate(key)
    return parsed


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = {}

    def run(self, task):
        result = self.execute(task)
        self.state[task] = result
        return result
'''

CONVERSATION = """system: You are a release engineer. Never force-push to main.
user: hey, can you look at the build
assistant: Sure, checking the pipeline now.
user: what did you find
assistant: The wheel job timed out on Windows.
assistant: We decided to pin the runner image to 2026.6 instead of latest.
user: sounds good
assistant: Also noting we must never skip the signing step.
user: thanks
"""


def _document() -> str:
    filler = "The service exposes a queue interface for background work. "
    answer = "Latency is bounded by the retry budget, which defaults to three attempts. "
    cited = "Throughput was measured at 4,200 requests per second [7]. "
    return (filler * 12) + answer + (filler * 8) + cited + (filler * 10)


def _schema() -> str:
    return json.dumps(
        {
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "Order": {
                        "type": "object",
                        "required": ["order_id", "amount_cents"],
                        "description": "An order. " + "Long prose. " * 40,
                        "properties": {
                            "order_id": {"type": "string", "format": "uuid"},
                            "amount_cents": {"type": "integer", "minimum": 1},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "paid", "refunded"],
                                "description": "Status. " + "More prose. " * 40,
                            },
                        },
                    }
                }
            },
        },
        indent=2,
    )


# ── Routing ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label,text,expected",
    [
        ("python source", SOURCE, "code"),
        ("conversation", CONVERSATION, "conversation"),
        ("openapi schema", _schema(), "schema"),
        ("shell run", "$ pytest tests/\ntests/a.py::t PASSED\n1 failed\nexit code 1", "shell"),
        (
            "log",
            "2026-08-02T10:00:00Z INFO up\n2026-08-02T10:00:01Z ERROR x\n"
            "2026-08-02T10:00:02Z ERROR x",
            "log",
        ),
        ("plain json", json.dumps({"a": 1, "items": [{"x": i} for i in range(5)]}), "json"),
    ],
)
def test_registry_routes_content_to_its_own_codec(label, text, expected):
    codec = default_registry().select(text)
    assert codec is not None and codec.name == expected, (
        f"{label} routed to {codec.name if codec else None}, expected {expected}"
    )


def test_source_code_is_not_claimed_by_the_shell_codec():
    """`# comment` is not a root prompt."""
    from entroly.codecs_builtin import ShellCodec

    commented = "# set up the widget\n# then run it\nvalue = compute()\n"
    assert not ShellCodec().supports(commented), (
        "comment lines were being read as shell prompts, which routed every "
        "Python and shell script to the shell codec"
    )


def test_code_confidence_does_not_fall_with_file_size():
    """A large, well-factored module must not lose the routing race."""
    small = CodeCodec().supports(SOURCE).confidence
    padded = CodeCodec().supports(SOURCE + "\n    x = 1" * 500).confidence
    assert padded >= small - 0.01, (
        f"confidence collapsed from {small} to {padded} purely because the file "
        f"got longer; that is declaration density, not evidence"
    )


# ── Code ────────────────────────────────────────────────────────────────────


def test_code_skeleton_keeps_the_call_surface():
    rep = CodeCodec().representations(SOURCE, source_id="runner.py")[-1]
    for needed in ("import os", "from pathlib import Path", "def load_config", "class Runner"):
        assert needed in rep.text, f"{needed!r} missing from skeleton:\n{rep.text}"
    assert "self.state[task] = result" not in rep.text, "bodies should be elided"


def test_code_error_strings_survive():
    rep = CodeCodec().representations(SOURCE, source_id="runner.py")[-1]
    assert "config is invalid or empty" in rep.text, (
        "error strings are what a user greps for and must survive skeletonising"
    )


def test_code_bodies_are_recoverable():
    store = RecoveryStore()
    rep = CodeCodec(store).representations(SOURCE, source_id="runner.py")[-1]
    assert rep.recovery is not None
    assert "self.state[task] = result" in store.recover(rep.recovery)


def test_code_protected_evidence_verifies():
    assert verify_all(CodeCodec().representations(SOURCE, source_id="r.py")) == {}


# ── Documents ───────────────────────────────────────────────────────────────


def test_document_without_a_query_offers_only_the_original():
    reps = DocumentCodec().representations(_document(), source_id="d.md")
    assert len(reps) == 1 and reps[0].distortion_risk == 0.0, (
        "with no query there is no basis for calling one span more "
        "answer-bearing than another"
    )


def test_document_query_selects_the_answer_span():
    reps = DocumentCodec().representations(
        _document(), source_id="d.md", query="what bounds latency"
    )
    assert len(reps) == 2
    assert "retry budget" in reps[-1].text
    assert reps[-1].token_cost < reps[0].token_cost


def test_document_keeps_citations_even_when_unmatched():
    rep = DocumentCodec().representations(
        _document(), source_id="d.md", query="what bounds latency"
    )[-1]
    assert "[7]" in rep.text, (
        "an unattributed claim is not evidence; citations survive regardless of "
        "whether they matched the query"
    )
    assert verify_all([rep]) == {}


def test_document_dropped_spans_are_recoverable():
    store = RecoveryStore()
    rep = DocumentCodec(store).representations(
        _document(), source_id="d.md", query="what bounds latency"
    )[-1]
    assert rep.recovery is not None and rep.recovery.item_count > 0
    assert "queue interface" in store.recover(rep.recovery)


# ── Conversation ────────────────────────────────────────────────────────────


def test_conversation_keeps_standing_instructions_and_decisions():
    rep = ConversationCodec().representations(CONVERSATION, source_id="chat")[-1]
    assert "Never force-push to main" in rep.text, "a standing instruction still binds"
    assert "pin the runner image" in rep.text, "a recorded decision still binds"
    assert "must never skip the signing step" in rep.text
    assert "sounds good" not in rep.text, "narrative chatter is what should go"


def test_conversation_prefix_is_byte_identical():
    """Rewriting the cached prefix costs a cache miss on every later request."""
    rep = ConversationCodec().representations(CONVERSATION, source_id="chat")[-1]
    prefix = CONVERSATION.split("\n")[0]
    assert rep.text.startswith(prefix), (
        f"the leading system block must be emitted unchanged and first;\n"
        f"  expected start: {prefix!r}\n  got: {rep.text[:len(prefix)]!r}"
    )


def test_conversation_pruned_turns_are_recoverable():
    store = RecoveryStore()
    rep = ConversationCodec(store).representations(CONVERSATION, source_id="chat")[-1]
    assert rep.recovery is not None
    assert "sounds good" in store.recover(rep.recovery)


# ── Schema ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label,needle",
    [
        ("required list", '"required"'),
        ("required field name", "order_id"),
        ("type", '"integer"'),
        ("enum values", "refunded"),
        ("constraint", '"minimum"'),
        ("format", '"uuid"'),
    ],
)
def test_schema_never_drops_the_contract(label, needle):
    rep = SchemaCodec().representations(_schema(), source_id="api.json")[-1]
    assert needle in rep.text, (
        f"{label} is part of the contract a caller must satisfy; dropping it "
        f"changes the API.\n---\n{rep.text[:400]}"
    )


def test_schema_drops_prose_and_shrinks():
    reps = SchemaCodec().representations(_schema(), source_id="api.json")
    lean = reps[-1]
    assert "Long prose." not in lean.text and "More prose." not in lean.text
    assert lean.token_cost < reps[0].token_cost


def test_schema_prose_is_recoverable():
    store = RecoveryStore()
    rep = SchemaCodec(store).representations(_schema(), source_id="api.json")[-1]
    assert rep.recovery is not None
    assert "Long prose." in store.recover(rep.recovery)


def test_plain_json_is_not_claimed_as_a_schema():
    assert not SchemaCodec().supports(json.dumps({"a": 1, "b": [1, 2]}))
