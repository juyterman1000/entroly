"""Black-box contracts for the documented base-install Python surface.

These tests import ``entroly`` exactly as downstream applications do. They do
not call private engine helpers and deliberately fail when an advertised export
silently disappears behind a broad optional-import fallback.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import entroly


DOCUMENTED_BASE_EXPORTS = (
    # Common SDK
    "compress",
    "compress_messages",
    "optimize",
    "verify",
    "create_context_receipt",
    "render_context_receipt",
    "explain_receipt_omission",
    "context_receipt_from_path",
    # Advanced local control
    "localize_files",
    "localize_fragments",
    "CacheAligner",
    "ContextLedger",
    "ProviderPrice",
    "clamp_injected_budget",
    "MemoryOS",
    # Verification and security
    "WitnessAnalyzer",
    "EICVAnalyzer",
    "EICVSuppressor",
    "stave_verify",
    "stave_risk",
    "acf_scan",
    "acf_sanitize",
    # Compression and recovery
    "compress_evidence_locked",
    "compress_proxy_payload",
    "CompressionRetrievalStore",
    "answer_with_retrieval_verification",
)


def _estimated_tokens(messages: list[dict]) -> int:
    return sum(
        len(message.get("content", "")) // 4
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )


def test_documented_base_install_exports_cannot_silently_disappear() -> None:
    missing = [name for name in DOCUMENTED_BASE_EXPORTS if not hasattr(entroly, name)]
    assert not missing, (
        "Documented base-install exports are missing. Top-level optional import "
        f"fallbacks must not hide packaging regressions: {missing}"
    )
    non_callable = [name for name in DOCUMENTED_BASE_EXPORTS if not callable(getattr(entroly, name))]
    assert not non_callable, f"Documented exports are not callable: {non_callable}"


def test_import_is_network_free_and_does_not_create_user_state(tmp_path: Path) -> None:
    """A library import must not phone home or create ~/.entroly state."""
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()

    script = r'''
import json
import os
import pathlib
import socket
import urllib.request

home = pathlib.Path(os.environ["HOME"])

def blocked(*_args, **_kwargs):
    raise AssertionError("network attempted during import")

socket.socket.connect = blocked
urllib.request.urlopen = blocked
before = sorted(str(path.relative_to(home)) for path in home.rglob("*") if path.is_file())
import entroly  # noqa: E402
assert entroly.__version__
after = sorted(str(path.relative_to(home)) for path in home.rglob("*") if path.is_file())
print(json.dumps({"before": before, "after": after}))
'''
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=work,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["after"] == payload["before"], (
        "Importing entroly created persistent user files before any feature was "
        f"called: {payload}"
    )


def test_compress_never_annihilates_inflates_or_breaks_unicode() -> None:
    samples = [
        "single",
        "مرحبا שלום नमस्ते 🧪 e\u0301\n" * 80,
        json.dumps({"rows": [{"id": i, "status": "ok"} for i in range(200)]}),
        "\n".join(f"INFO request={i} status=200 elapsed={i % 17}ms" for i in range(500)),
        "\n".join(
            [
                "class SessionManager:",
                "    def refresh_token(self, token):",
                "        return validate_token(token)",
            ]
            * 180
        ),
    ]

    for original in samples:
        budget = max(1, len(original) // 16)
        output = entroly.compress(original, budget=budget)
        assert isinstance(output, str)
        assert output.strip(), f"non-empty input was annihilated: {original[:80]!r}"
        assert len(output) <= len(original), "compress() inflated its input"
        assert len(output) // 4 <= budget, "explicit estimated-token budget was exceeded"
        output.encode("utf-8")

    assert entroly.compress("") == ""
    assert entroly.compress("already tiny", budget=100) == "already tiny"


def test_compress_messages_preserves_input_roles_and_last_user_turn() -> None:
    messages = [
        {"role": "system", "content": "system policy\n" * 180},
        {"role": "tool", "content": "large tool output needle-auth-timeout\n" * 260},
        {"role": "assistant", "content": "analysis of unrelated files\n" * 220},
        {"role": "assistant", "content": "I will inspect the authentication timeout."},
        {"role": "user", "content": "Where is needle-auth-timeout handled?"},
    ]
    original = copy.deepcopy(messages)

    compressed = entroly.compress_messages(
        messages,
        budget=180,
        preserve_last_n=2,
        profile="balanced",
        distill=True,
    )

    assert messages == original, "compress_messages mutated caller-owned input"
    assert len(compressed) == len(messages)
    assert [message.get("role") for message in compressed] == [
        message.get("role") for message in messages
    ]
    assert compressed[-2:] == messages[-2:], "recent conversation was not preserved verbatim"
    assert compressed[-1]["content"] == "Where is needle-auth-timeout handled?"
    assert _estimated_tokens(compressed) <= 180

    multimodal = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": "ok"},
    ]
    assert entroly.compress_messages(multimodal, budget=100) == multimodal


def test_public_context_receipt_is_deterministic_renderable_and_explainable() -> None:
    documents = [
        (
            "auth.md",
            "# Authentication\n\nThe refresh slack is 45 seconds.\n\n"
            "The timeout handler lives in auth/session.py.\n",
        ),
        (
            "operations.md",
            "# Operations\n\nDeployments require two approvals.\n",
        ),
    ]
    kwargs = {
        "query": "Where is the timeout handler and what is the refresh slack?",
        "budget": 50,
        "chunk_tokens": 40,
        "prefer_rust": False,
    }

    first = entroly.create_context_receipt(documents, **kwargs)
    second = entroly.create_context_receipt(documents, **kwargs)

    assert first["receipt_id"] == second["receipt_id"]
    assert first["reproducibility_hash"] == second["reproducibility_hash"]
    assert first["selected_context"]
    report = entroly.render_context_receipt(first, prefer_rust=False)
    assert "# Context Receipt" in report

    candidates = first.get("omitted_context") or first.get("selected_context") or []
    assert candidates
    chunk_id = candidates[0]["chunk_id"]
    explanation = entroly.explain_receipt_omission(first, chunk_id, prefer_rust=False)
    assert chunk_id in explanation
