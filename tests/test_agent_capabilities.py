"""
Tests for agent capability modules:
- Cross-agent shared memory (entroly.shared_memory)
- Output token reduction & steering (entroly.output_steering)
- Shell hook output compression (entroly.shell_hook)
- Multimodal image compression (entroly.image_compress)
- Failure mining & self-correction (entroly.learn)
- SDK surface bindings (entroly.sdk)
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from entroly.shared_memory import (
    SharedMemoryStore,
    SharedEntry,
    _compute_simhash,
    _hamming_distance,
)
from entroly.output_steering import (
    Effort,
    EFFORT_MAX_TOKENS,
    classify_effort,
    VerbositySteerer,
)
from entroly.shell_hook import (
    detect_command,
    compress_shell_output,
    recover_shell_output,
    PATTERNS,
)
from entroly.image_compress import compress_image, CompressionResult
from entroly.learn import FailurePattern, Correction, FailureMiner
import entroly.sdk as sdk


# ---------------------------------------------------------------------------
# 1. Shared Memory Tests
# ---------------------------------------------------------------------------

def test_shared_memory_write_and_read(tmp_path: Path):
    store = SharedMemoryStore(root=tmp_path)
    entry = store.write(
        content="PostgreSQL connection pool max_size is set to 25 in production",
        agent_id="claude-code",
        session_id="sess-123",
        tags=["database", "config"],
    )
    assert entry is not None
    assert entry.agent_id == "claude-code"
    assert "database" in entry.tags

    # Read back
    entries = store.list_entries()
    assert len(entries) == 1
    assert entries[0].entry_id == entry.entry_id
    assert "max_size is set to 25" in entries[0].content


def test_shared_memory_near_duplicate_dedup(tmp_path: Path):
    store = SharedMemoryStore(root=tmp_path)
    text1 = "PostgreSQL connection pool max_size is set to 25 in production config"
    text2 = "PostgreSQL connection pool max_size is set to 25 in production config."

    e1 = store.write(text1, agent_id="agent-1")
    assert e1 is not None

    # Near-duplicate from agent-2 should be deduplicated away
    e2 = store.write(text2, agent_id="agent-2", dedup=True)
    assert e2 is None

    # But forced write should allow it
    e3 = store.write(text2, agent_id="agent-2", dedup=False)
    assert e3 is not None


def test_shared_memory_search(tmp_path: Path):
    store = SharedMemoryStore(root=tmp_path)
    store.write("Rust memory leak in worker thread channel receiver", agent_id="codex", tags=["rust"])
    store.write("Python FastAPI endpoint authentication header validation", agent_id="cursor", tags=["python"])

    results = store.search("FastAPI authentication", top_k=5)
    assert len(results) >= 1
    assert "FastAPI" in results[0].content


def test_simhash_deterministic():
    text = "The quick brown fox jumps over the lazy dog"
    h1 = _compute_simhash(text)
    h2 = _compute_simhash(text)
    assert h1 == h2
    assert isinstance(h1, int)


# ---------------------------------------------------------------------------
# 2. Output Steering Tests
# ---------------------------------------------------------------------------

def test_classify_effort_edge_cases():
    # Minimal
    assert classify_effort("what is the path to main.rs").effort == Effort.MINIMAL
    assert classify_effort("yes").effort == Effort.MINIMAL

    # Concise
    assert classify_effort("fix the login redirect bug").effort == Effort.CONCISE

    # Exhaustive / Detailed
    assert classify_effort("architecture review of the multi-region distributed cache subsystem with edge cases").effort in (Effort.DETAILED, Effort.EXHAUSTIVE)


def test_verbosity_steerer_injection():
    steerer = VerbositySteerer()

    messages = [{"role": "system", "content": "You are a helpful coding assistant."}]
    steered_msgs, eff, max_tok = steerer.steer(messages, Effort.MINIMAL)
    assert "shortest complete form" in steered_msgs[0]["content"]
    assert eff == Effort.MINIMAL
    assert max_tok <= 250

    # Standard effort shouldn't mutate prompt
    base_msgs = [{"role": "system", "content": "You are a helpful coding assistant."}]
    unchanged, eff_std, tok_std = steerer.steer(base_msgs, Effort.STANDARD)
    assert unchanged[0]["content"] == "You are a helpful coding assistant."
    assert eff_std == Effort.STANDARD


def test_effort_max_tokens_scaling():
    assert EFFORT_MAX_TOKENS[Effort.MINIMAL] <= 200
    assert EFFORT_MAX_TOKENS[Effort.CONCISE] <= 600
    assert EFFORT_MAX_TOKENS[Effort.EXHAUSTIVE] >= 8192


# ---------------------------------------------------------------------------
# 3. Shell Hook Tests
# ---------------------------------------------------------------------------

def test_detect_command():
    assert detect_command("npm WARN deprecated inflight@1.0.6") == "npm"
    assert detect_command("Compiling entroly-core v1.0.62\n   Finished dev") == "cargo"
    assert detect_command("git commit -m 'feat: test'\nremote: Compressing objects") == "git"
    assert detect_command("test session starts\npytest tests/ -v\npassed in 0.2s") == "pytest"


def test_shell_hook_npm_compression():
    raw_npm = (
        "npm WARN deprecated inflight@1.0.6: This module is not supported\n"
        "npm WARN deprecated glob@7.2.3: Glob versions prior to v9 are no longer supported\n"
        "added 145 packages from 89 contributors and audited 146 packages in 4.21s\n"
        "found 0 vulnerabilities\n"
    )
    compressed, orig_count, comp_count, handle = compress_shell_output(raw_npm, command="npm")
    assert comp_count <= orig_count
    assert "found 0 vulnerabilities" in compressed


def test_shell_hook_recovery():
    long_output = "\n".join([f"line {i}: normal operation log event" for i in range(100)])
    compressed, orig, comp, handle = compress_shell_output(long_output, command="unknown", max_lines=20)
    if handle:
        recovered = recover_shell_output(handle)
        assert recovered == long_output


# ---------------------------------------------------------------------------
# 4. Multimodal & Image Compression Tests
# ---------------------------------------------------------------------------

def test_image_compress_handles_invalid_bytes():
    result = compress_image(b"not a real image stream", max_dimension=800)
    assert isinstance(result, CompressionResult)
    assert result.original_size == len(b"not a real image stream")


# ---------------------------------------------------------------------------
# 5. Failure Mining Tests
# ---------------------------------------------------------------------------

def test_mine_failures_empty_dir(tmp_path: Path):
    miner = FailureMiner(root=tmp_path)
    patterns = miner.mine()
    assert isinstance(patterns, list)
    assert len(patterns) == 0


def test_failure_pattern_data_structure():
    pattern = FailurePattern(
        pattern_id="fp-01",
        entity="auth.py",
        category="security",
        description="JWT expiration not verified",
        occurrences=3,
        examples=["Session fixation on re-login"],
        suggested_fix="Validate exp claim in token payload",
        confidence=0.95,
    )
    assert pattern.occurrences == 3
    assert pattern.confidence == 0.95


# ---------------------------------------------------------------------------
# 6. SDK Surface Exports
# ---------------------------------------------------------------------------

def test_sdk_surface_bindings():
    expected_attrs = [
        "shared_memory_write",
        "shared_memory_search",
        "classify_effort",
        "steer_output",
        "compress_shell",
    ]
    for attr in expected_attrs:
        assert hasattr(sdk, attr), f"Missing {attr} on entroly.sdk"
        assert callable(getattr(sdk, attr)), f"entroly.sdk.{attr} is not callable"
