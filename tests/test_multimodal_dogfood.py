"""Adversarial contracts for the advertised multimodal intake layer."""

from __future__ import annotations

import re
import struct

import pytest

from entroly.image_optimizer import (
    estimate_image_tokens,
    estimate_image_tokens_from_dimensions,
    image_dimensions,
    optimize_image_bytes,
    plan_image_optimization,
)
from entroly.multimodal import (
    ModalContent,
    ingest_diagram,
    ingest_diff,
    ingest_image,
    ingest_voice,
)


def _minimal_png(width: int, height: int) -> bytes:
    # Dimension probing only requires the PNG signature and IHDR width/height
    # offsets; no pixel decode or optional Pillow dependency is involved.
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", width, height)


def test_modal_content_clamps_confidence_and_recomputes_tokens() -> None:
    high = ModalContent("hello world", "image", "x", 9.0, -500)
    low = ModalContent("", "voice", "y", -4.0, 999)

    assert high.confidence == 1.0
    assert high.token_estimate == max(1, len(high.text) // 4)
    assert low.confidence == 0.0
    assert low.token_estimate == 1


def test_image_description_fallback_is_unicode_safe_and_structured() -> None:
    description = "تسجيل الدخول failed 🧪 — click Submit, then an error modal appears."
    first = ingest_image(description, "ui/login.png")
    second = ingest_image(description, "ui/login.png")

    assert first == second
    assert first.source_type == "image"
    assert first.metadata["method"] == "pre-described"
    assert "authentication-form" in first.metadata["regions"]
    assert "interactive-controls" in first.metadata["regions"]
    assert "alert-dialog" in first.metadata["regions"]
    assert description in first.text
    first.text.encode("utf-8")


def test_empty_and_malformed_image_inputs_degrade_without_crashing() -> None:
    empty = ingest_image("", "empty-image")
    malformed = ingest_image("A" * 63 + "!", "bad-base64")

    assert "(no text content extracted)" in empty.text
    assert empty.metadata["regions"] == ["general-ui"]
    assert malformed.source_type == "image"
    assert malformed.text


@pytest.mark.parametrize(
    ("source", "expected_type", "expected_edges"),
    [
        ("flowchart LR\nA[Client] --> B[API]", "mermaid", 1),
        ("@startuml\nClient -> API: request\n@enduml", "plantuml", 1),
        ('digraph G {\nA -> B [label="request"]\n}', "dot", 1),
        ("Client -> API: request", "text", 1),
    ],
)
def test_diagram_formats_produce_deterministic_graph_metadata(
    source: str, expected_type: str, expected_edges: int
) -> None:
    first = ingest_diagram(source, f"diagram.{expected_type}")
    second = ingest_diagram(source, f"diagram.{expected_type}")

    assert first == second
    assert first.metadata["diagram_type"] == expected_type
    assert first.metadata["edge_count"] == expected_edges
    assert first.metadata["node_count"] >= 2
    assert "Relationships / Data Flow" in first.text


def test_multifile_diff_preserves_exact_paths_without_duplicate_hunks() -> None:
    diff = """\
diff --git a/backend.py b/backend.py
--- a/backend.py
+++ b/backend.py
@@ -1,2 +1,2 @@
-def old_login(value):
+def secure_login(value):
     return value
diff --git a/billing.py b/billing.py
--- a/billing.py
+++ b/billing.py
@@ -1 +1 @@
-def charge_once():
+def charge_idempotently():
"""
    result = ingest_diff(
        diff,
        "security-fix.diff",
        commit_message="fix security injection vulnerability",
    )

    assert result.metadata["files_changed"] == 2, result.metadata
    assert result.metadata["intent"] == "security"
    assert "### backend.py" in result.text
    assert "### billing.py" in result.text
    assert "### ackend.py" not in result.text
    assert "### illing.py" not in result.text
    assert result.text.count("### backend.py") == 1
    assert result.text.count("### billing.py") == 1
    assert {"secure_login", "charge_idempotently"} <= set(
        result.metadata["symbols_changed"]
    )


def test_voice_vocabulary_preserves_complete_file_references() -> None:
    transcript = (
        "SPEAKER_00: We should move the refresh logic into auth/session.py. "
        "SPEAKER_01: Action item: update retry_policy.yaml and the /api/login endpoint. "
        "How should we test the timeout?"
    )
    result = ingest_voice(
        transcript,
        "design-meeting.txt",
        speaker_labels={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
    )

    vocabulary_match = re.search(
        r"## Technical Vocabulary Referenced\n(.*?)\n\n",
        result.text,
        flags=re.DOTALL,
    )
    assert vocabulary_match, result.text
    vocabulary = vocabulary_match.group(1).split()
    assert "session.py" in vocabulary
    assert "retry_policy.yaml" in vocabulary
    assert "/api/login" in vocabulary
    assert "p.y" not in vocabulary
    assert "y.aml" not in vocabulary
    assert "Alice" in result.text and "Bob" in result.text


def test_image_token_planning_is_positive_and_opt_in_preserves_exact_bytes() -> None:
    image = _minimal_png(4096, 2048)
    assert image_dimensions(image) == (4096, 2048)

    for provider in ("openai", "anthropic", "gemini", "unknown"):
        estimate = estimate_image_tokens(image, provider=provider)
        assert estimate.width == 4096
        assert estimate.height == 2048
        assert estimate.estimated_tokens > 0

    decision = plan_image_optimization(image, provider="anthropic")
    unchanged, disabled = optimize_image_bytes(
        image,
        provider="anthropic",
        enabled=False,
    )
    assert unchanged == image
    assert disabled.action == "preserve"
    assert disabled.reason == "disabled_explicit_opt_in_required"
    assert disabled.before == decision.before


@pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_image_estimator_rejects_impossible_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="width|height|positive"):
        estimate_image_tokens_from_dimensions(width, height)
