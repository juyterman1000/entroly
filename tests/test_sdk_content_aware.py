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


def test_code_compression_preserves_symbol_skeleton_under_budget():
    filler = "\n".join(
        f"const unrelatedWorkerHelper{i} = 'tenant project retry queue';"
        for i in range(300)
    )
    content = (
        "export class IngestionService {\n"
        "  private async processTraceEventList(params): Promise<void> {\n"
        "    const traceRecords = this.mapTraceEventsToRecords(params);\n"
        "  }\n"
        "  private mapTraceEventsToRecords(params): TraceRecordInsertType[] {\n"
        "    return params.traceEventList.map((trace) => ({ id: trace.id }));\n"
        "  }\n"
        "}\n"
        "export type TraceRecordInsertType = { id: string };\n"
        f"{filler}\n"
    )

    out = sdk.compress(content, budget=160, content_type="code")

    assert len(out) <= 160 * 4
    assert "IngestionService" in out
    assert "processTraceEventList" in out
    assert "mapTraceEventsToRecords" in out
    assert "TraceRecordInsertType" in out


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


def test_compress_messages_compresses_huge_recent_context_to_budget():
    messages = [
        {"role": "system", "content": "system constraints"},
        {"role": "user", "content": "export function traceWorker() { return 'ok'; }\n" * 600},
        {"role": "assistant", "content": "I can inspect that context."},
        {"role": "user", "content": "What does traceWorker do?"},
    ]

    out = sdk.compress_messages(messages, budget=500, preserve_last_n=4, distill=False)

    assert out[-1] == messages[-1]
    assert sum(len(message["content"]) // 4 for message in out) <= 500
    assert len(out[1]["content"]) < len(messages[1]["content"])


def test_compress_messages_has_gentle_relative_operating_point():
    distractors = "\n\n".join(
        f"Archive {i}: The Cedar service uses routine key-{i} for batch cleanup."
        for i in range(80)
    )
    evidence = (
        "Orchid protocol deployment notes: the Orchid protocol uses the "
        "sapphire key for production recovery."
    )
    messages = [
        {"role": "user", "content": f"{distractors}\n\n{evidence}"},
        {"role": "user", "content": "Which key does the Orchid protocol use?"},
    ]
    before = sum(len(message["content"]) // 4 for message in messages)

    out = sdk.compress_messages(
        messages,
        target_ratio=0.90,
        preserve_last_n=2,
        distill=False,
    )

    after = sum(len(message["content"]) // 4 for message in out)
    reduction = 1.0 - after / before
    assert 0.08 <= reduction <= 0.20
    assert "sapphire key" in out[0]["content"]
    assert out[-1] == messages[-1]


def test_compress_messages_balanced_profile_keeps_query_evidence_under_budget():
    context = "\n\n".join(
        [
            *(
                f"Unrelated service {i} is deployed in region-{i % 7}."
                for i in range(160)
            ),
            (
                "Orchid protocol incident record: production recovery requires "
                "the sapphire key and approval from the incident commander."
            ),
            *(
                f"Historical service {i} completed a routine migration."
                for i in range(160, 260)
            ),
        ]
    )
    messages = [
        {"role": "user", "content": context},
        {"role": "user", "content": "Which key is required by the Orchid protocol?"},
    ]

    out = sdk.compress_messages(
        messages,
        budget=220,
        preserve_last_n=2,
        distill=False,
        profile="balanced",
    )

    assert "sapphire key" in out[0]["content"]
    assert out[-1] == messages[-1]
    assert sum(len(message["content"]) // 4 for message in out) <= 220


@pytest.mark.parametrize("target_ratio", [0, -0.1, 1.01])
def test_compress_messages_rejects_invalid_target_ratio(target_ratio):
    with pytest.raises(ValueError, match="target_ratio"):
        sdk.compress_messages(
            [{"role": "user", "content": "hello"}],
            target_ratio=target_ratio,
        )


def test_compress_messages_target_ratio_one_is_exact_passthrough():
    messages = [
        {"role": "tool", "content": "old tool evidence " * 200},
        {"role": "assistant", "content": "intermediate analysis"},
        {"role": "user", "content": "What happened?"},
    ]

    out = sdk.compress_messages(
        messages,
        target_ratio=1,
        preserve_last_n=1,
    )

    assert out is messages


def test_compress_messages_rejects_unknown_profile_even_when_in_budget():
    with pytest.raises(ValueError, match="Unknown profile"):
        sdk.compress_messages(
            [{"role": "user", "content": "hello"}],
            profile="reckless",
        )


def test_compress_messages_safe_profile_skips_assistant_distillation(monkeypatch):
    def unexpected_distillation(*args, **kwargs):
        raise AssertionError("safe profile must not distill assistant evidence")

    monkeypatch.setattr(
        "entroly.proxy_transform.distill_response",
        unexpected_distillation,
    )
    messages = [
        {
            "role": "assistant",
            "content": (
                "Orchid protocol evidence says the sapphire key is required.\n\n"
                * 120
            ),
        },
        {"role": "user", "content": "Which key does Orchid require?"},
    ]

    out = sdk.compress_messages(
        messages,
        budget=180,
        preserve_last_n=1,
        profile="safe",
    )

    assert out[-1] == messages[-1]
    assert "sapphire key" in out[0]["content"]


def test_universal_compress_code_uses_code_path():
    from entroly.universal_compress import universal_compress

    content = (
        "export type TraceRecord = { id: string; name: string };\n"
        "export function normalizeTrace(trace: TraceRecord): TraceRecord { return trace; }\n"
    ) * 200

    out, ctype, savings = universal_compress(content, 0.3, "code")

    assert ctype == "code"
    assert out.strip()
    assert len(out) < len(content) * 0.6
    assert savings > 0.4


def test_punctuation_light_prose_does_not_noop():
    content = " ".join(
        ["important architecture compression cli mcp first user bug"] * 800
    )

    out = sdk.compress(content, content_type="text", target_ratio=0.1)

    assert out.strip()
    assert len(out) < len(content) * 0.5
    assert out != content
