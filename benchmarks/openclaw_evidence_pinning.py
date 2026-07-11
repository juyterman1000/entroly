"""Deterministic, no-model benchmark for OpenClaw evidence pinning."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from entroly.openclaw_bridge import assemble


NEEDLE = "AUTH_EVIDENCE: refresh token rotation requires revoking the prior token."


def _messages() -> list[dict[str, str]]:
    evidence = (
        "Architecture notes. "
        + "background detail " * 80
        + NEEDLE
        + " additional implementation detail " * 60
    )
    messages = [{"role": "system", "content": "You are an engineering agent."}]
    messages.extend(
        {"role": "assistant", "content": f"distractor-{index} " * 850}
        for index in range(8)
    )
    messages.insert(4, {"role": "assistant", "content": evidence})
    messages.append(
        {
            "role": "user",
            "content": "How must authentication refresh token rotation work?",
        }
    )
    return messages


def run() -> dict[str, object]:
    messages = _messages()
    with tempfile.TemporaryDirectory(prefix="entroly-openclaw-benchmark-") as temp:
        pinned = assemble(
            {
                "operation": "assemble",
                "session_id": "benchmark",
                "messages": messages,
                "prompt": messages[-1]["content"],
                "token_budget": 1800,
                "preserve_last_n": 1,
                "receipt_dir": temp,
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
                "receipt_dir": temp,
                "distill": False,
                "evidence_pinning": False,
            }
        )
        receipt = json.loads(Path(pinned["receipt_path"]).read_text(encoding="utf-8"))
    pinned_text = "\n".join(
        message.get("content", "")
        for message in pinned["messages"]
        if isinstance(message.get("content"), str)
    )
    control_text = "\n".join(
        message.get("content", "")
        for message in control["messages"]
        if isinstance(message.get("content"), str)
    )
    return {
        "workload": "synthetic long-running OpenClaw session",
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
        "receipt_reproducibility_hash": receipt["assembled_sha256"],
        "caveat": "Synthetic deterministic benchmark; token counts are estimates, not billed usage.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
