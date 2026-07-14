"""Deterministic, no-model benchmark for OpenClaw evidence pinning."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from entroly.openclaw_bridge import assemble, handle_request


NEEDLE = "AUTH_EVIDENCE: refresh token rotation requires revoking the prior token."


def _assistant_message(content: str, index: int) -> dict[str, object]:
    providers = (
        ("openai", "gpt-5.6-sol"),
        ("anthropic", "claude-opus-4-6"),
        ("google", "gemini-2.5-pro"),
        ("nvidia", "nemotron-3-ultra-550b-a55b"),
        ("ollama", "qwen3:8b"),
    )
    provider, model = providers[index % len(providers)]
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "api": "openai-compatible" if provider not in {"anthropic", "google"} else provider,
        "provider": provider,
        "model": model,
        "usage": {
            "input": 1000 + index,
            "output": 100,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 1100 + index,
        },
        "stopReason": "stop",
        "timestamp": 1_700_000_000_000 + index,
    }


def _messages() -> list[dict[str, object]]:
    evidence = (
        "Architecture notes. "
        + "background detail " * 80
        + NEEDLE
        + " additional implementation detail " * 60
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are an engineering agent."}
    ]
    messages.extend(
        _assistant_message(f"distractor-{index} " * 850, index)
        for index in range(8)
    )
    messages.insert(4, _assistant_message(evidence, 99))
    messages.append(
        {
            "role": "user",
            "content": "How must authentication refresh token rotation work?",
        }
    )
    return messages


def _visible_text(message: dict[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block["text"])
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def run() -> dict[str, object]:
    messages = _messages()
    receipt_commit_token = "c" * 64
    receipt_commit_challenge = hashlib.sha256(
        receipt_commit_token.encode("utf-8")
    ).hexdigest()
    with tempfile.TemporaryDirectory(prefix="entroly-openclaw-benchmark-") as temp:
        pinned = assemble(
            {
                "operation": "assemble",
                "session_id": "benchmark",
                "messages": messages,
                "prompt": messages[-1]["content"],
                "token_budget": 1800,
                "preserve_last_n": 1,
                "workspace_dir": str(Path.cwd()),
                "receipt_dir": temp,
                "receipt_commit_challenge_sha256": receipt_commit_challenge,
                "distill": False,
            }
        )
        control = assemble(
            {
                "operation": "assemble",
                "session_id": "benchmark-control",
                "messages": messages,
                "prompt": messages[-1]["content"],
                "token_budget": 1800,
                "preserve_last_n": 1,
                "workspace_dir": str(Path.cwd()),
                "receipt_dir": temp,
                "distill": False,
                "evidence_pinning": False,
                "write_receipt": False,
            }
        )
        committed = handle_request(
            {
                "operation": "commit_receipt",
                "receipt_id": pinned["receipt_id"],
                "proposal_id": pinned["proposal_id"],
                "proposal_sha256": pinned["proposal_sha256"],
                "receipt_path": pinned["receipt_path"],
                "receipt_commit_token": receipt_commit_token,
                "workspace_dir": str(Path.cwd()),
            }
        )
        if committed.get("committed") is not True:
            raise RuntimeError("benchmark receipt was not acknowledged")
        receipt = json.loads(Path(pinned["receipt_path"]).read_text(encoding="utf-8"))
        if receipt.get("acceptance_status") != "accepted":
            raise RuntimeError("benchmark receipt did not persist acceptance")
    pinned_text = "\n".join(_visible_text(message) for message in pinned["messages"])
    control_text = "\n".join(_visible_text(message) for message in control["messages"])
    return {
        "workload": "synthetic normalized multi-provider OpenClaw session",
        "model_calls": 0,
        "message_count": len(messages),
        "token_budget_estimated": 1800,
        "source_tokens_estimated": pinned["source_tokens"],
        "evidence_pinning": {
            "assembled_tokens_estimated": pinned["estimated_tokens"],
            "tokens_saved_estimated": pinned["tokens_saved"],
            "evidence_pinned": pinned["evidence_pinned"],
            "exact_needle_retained": NEEDLE in pinned_text,
        },
        "uniform_control": {
            "assembled_tokens_estimated": control["estimated_tokens"],
            "tokens_saved_estimated": control["tokens_saved"],
            "evidence_pinned": control["evidence_pinned"],
            "exact_needle_retained": NEEDLE in control_text,
        },
        "assembled_reproducibility_hash": hashlib.sha256(
            json.dumps(
                pinned["messages"],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "caveat": "Synthetic deterministic benchmark; token counts are estimates, not billed usage.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
