"""Focused regressions for the public content-aware SDK compressor."""

from __future__ import annotations

import json

import pytest

from entroly import sdk


@pytest.mark.parametrize("content_type", ["prose", "code", "json", "log"])
def test_explicit_budget_is_an_estimated_upper_bound(content_type):
    if content_type == "json":
        content = json.dumps(
            {"items": [{"id": i, "detail": f"value-{i}" * 8} for i in range(120)]},
            indent=2,
        )
    elif content_type == "log":
        content = "\n".join(
            f"2026-05-31T19:00:{i % 60:02d} INFO worker-{i % 8} flush ok t={i}ms"
            for i in range(400)
        )
    elif content_type == "code":
        content = "\n\n".join(
            f"def handler_{i}(payload):\n    return payload.get('field_{i}')"
            for i in range(180)
        )
    else:
        content = " ".join(
            f"Paragraph {i} describes a distinct operational constraint."
            for i in range(300)
        )

    out = sdk.compress(content, budget=120, content_type=content_type)

    assert out.strip()
    assert len(out) <= 120 * 4


def test_optimize_preserves_query_for_full_engine(monkeypatch):
    seen: list[tuple[int, str]] = []

    class FakeEngine:
        def __init__(self):
            self.fragments: list[dict[str, object]] = []

        def ingest_fragment(self, content: str, source: str, tokens: int):
            self.fragments.append(
                {"content": content, "source": source, "token_count": tokens}
            )

        def optimize_context(self, token_budget: int, query: str):
            seen.append((token_budget, query))
            needle = "auth" if "auth" in query else "billing"
            return {
                "selected": [
                    fragment
                    for fragment in self.fragments
                    if needle in str(fragment["source"])
                ]
            }

    monkeypatch.setattr("entroly.server.EntrolyEngine", FakeEngine)
    fragments = [
        {"source": "auth.txt", "content": "authentication context"},
        {"source": "billing.txt", "content": "billing context"},
    ]

    auth = sdk.optimize(fragments, budget=40, query="fix auth")
    billing = sdk.optimize(fragments, budget=40, query="fix billing")

    assert seen == [(40, "fix auth"), (40, "fix billing")]
    assert auth["context_text"] == "authentication context"
    assert billing["context_text"] == "billing context"


def test_compress_messages_caps_budget_for_named_model(monkeypatch):
    monkeypatch.setattr("entroly.proxy_config.context_window_for_model", lambda model: 1_000)
    messages = [
        {"role": "system", "content": "constraint alpha beta gamma. " * 300},
        {"role": "assistant", "content": "implementation detail delta epsilon. " * 300},
        {"role": "user", "content": "Keep this recent request verbatim."},
    ]

    out = sdk.compress_messages(
        messages,
        budget=10_000,
        preserve_last_n=1,
        model="test-model",
        distill=False,
    )

    assert out[-1] == messages[-1]
    assert len(out[0]["content"]) < len(messages[0]["content"])
    assert sum(len(message["content"]) // 4 for message in out) <= 800
