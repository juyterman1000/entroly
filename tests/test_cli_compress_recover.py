"""`entroly compress` / `entroly recover` — the first-run journey.

The recovery contract was unreachable from a terminal: a user could get a
digest from the SDK but had no way to turn it into the original bytes without
writing Python. A recovery guarantee nobody can invoke is a claim, not a
feature.

The journey these pin is: compress a file, see what it cost and what was kept,
then recover the original **in a separate process** — because that is the only
version of the promise that matters. An in-process round trip proves nothing
about a store on disk.

Byte fidelity is asserted on bytes, never on text. The first working version of
this command read files with `Path.read_text()`, which translates CRLF to LF on
Windows: a 3,854-byte file was stored as 3,606 bytes, and recovery returned
something faithful to what was compressed but not to the file on disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run(args: list[str], cwd: Path, store: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "ENTROLY_DIR": str(store),
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "entroly", *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
    )


def _payload() -> str:
    return json.dumps(
        {
            "request_id": "req_8f3a21bc",
            "error": {"code": "PAYMENT_DECLINED"},
            "amount_cents": 449900,
            "items": [{"sku": f"SKU-{i:04d}", "price": 1999 + i} for i in range(60)],
        },
        indent=2,
    )


@pytest.fixture
def workspace(tmp_path: Path):
    work = tmp_path / "work"
    store = tmp_path / "store"
    work.mkdir()
    store.mkdir()
    return work, store


def _compress_json(work: Path, store: Path, name: str, *extra: str) -> dict:
    result = _run(["compress", name, "--json", *extra], work, store)
    assert result.returncode == 0, result.stderr[-1500:]
    start = result.stdout.find("{")
    assert start >= 0, f"no JSON receipt in output: {result.stdout[:400]}"
    return json.loads(result.stdout[start:])


def test_compress_reports_a_receipt_with_a_recovery_digest(workspace):
    work, store = workspace
    (work / "payload.json").write_bytes(_payload().encode("utf-8"))

    receipt = _compress_json(work, store, "payload.json")
    assert receipt["codec"] == "json"
    assert receipt["tokens_after"] < receipt["tokens_before"]
    assert receipt["reduction_pct"] > 0
    assert receipt["recovery_digest"], "a lossy result must carry a way back"
    assert receipt["unverified_protected"] == [], (
        "the receipt must not claim evidence the output does not contain"
    )


@pytest.mark.parametrize(
    "label,raw",
    [
        ("LF", _payload().encode("utf-8")),
        ("CRLF", _payload().replace("\n", "\r\n").encode("utf-8")),
        ("no trailing newline", _payload().rstrip("\n").encode("utf-8")),
        ("utf-8 text", json.dumps(
            {"note": "naïve café — 日本語 🚀",
             "items": [{"x": i} for i in range(40)]},
            indent=2, ensure_ascii=False).encode("utf-8")),
    ],
)
def test_recover_returns_the_original_bytes_in_a_new_process(workspace, label, raw):
    work, store = workspace
    (work / "payload.json").write_bytes(raw)

    receipt = _compress_json(work, store, "payload.json")
    digest = receipt["recovery_digest"]
    assert digest, f"{label}: no recovery digest"

    # A separate process, sharing only the store on disk.
    result = _run(["recover", digest, "--out", "restored.json"], work, store)
    assert result.returncode == 0, result.stderr[-1500:]

    restored = (work / "restored.json").read_bytes()
    assert restored == raw, (
        f"{label}: recovered {len(restored)} bytes, original was {len(raw)}"
    )


def test_compressed_output_is_smaller_than_the_original(workspace):
    work, store = workspace
    raw = _payload().encode("utf-8")
    (work / "payload.json").write_bytes(raw)

    _compress_json(work, store, "payload.json", "--out", "small.json")
    assert (work / "small.json").stat().st_size < len(raw)


def test_unknown_digest_fails_loudly_rather_than_returning_nothing(workspace):
    work, store = workspace
    result = _run(["recover", "sha256:" + "0" * 64], work, store)
    assert result.returncode == 1
    assert "No recovery entry" in result.stdout or "No recovery entry" in result.stderr


def test_missing_file_is_reported(workspace):
    work, store = workspace
    result = _run(["compress", "nope.json"], work, store)
    assert result.returncode == 1
    assert "No such file" in result.stdout or "No such file" in result.stderr


def test_content_no_codec_claims_is_left_unchanged(workspace):
    work, store = workspace
    prose = b"A short note about nothing structured at all.\n"
    (work / "note.txt").write_bytes(prose)

    receipt = _compress_json(work, store, "note.txt")
    assert receipt["codec"] in {"none", "passthrough"}
    assert receipt["tokens_after"] == receipt["tokens_before"]
