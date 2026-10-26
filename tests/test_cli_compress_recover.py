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


_PY_SOURCE = "import os\n\n" + "".join(
    f"def handler_{i}(event):\n"
    f"    value = event.get('k{i}')\n"
    "    if value is None:\n"
    "        return 404\n"
    "    return value\n\n"
    for i in range(30)
)


def test_compress_does_not_promise_the_original_for_partial_recovery(workspace):
    # Code recovery holds only the elided bodies, without their positions
    # (pinned by test_recover_states_what_it_returned). The receipt must not
    # tell the user that the digest brings the original file back.
    work, store = workspace
    (work / "app.py").write_text(_PY_SOURCE, encoding="utf-8")

    human = _run(["compress", "app.py"], work, store)
    assert human.returncode == 0, human.stderr[-800:]
    assert "Recover the original" not in human.stdout
    assert "omitted parts" in human.stdout

    receipt = _compress_json(work, store, "app.py")
    assert receipt["codec"] == "code"
    assert receipt["recovery_returns"] == "omitted_parts"


def test_compress_promises_the_original_only_when_recovery_is_complete(workspace):
    work, store = workspace
    (work / "resp.json").write_text(_payload(), encoding="utf-8")

    human = _run(["compress", "resp.json"], work, store)
    assert human.returncode == 0, human.stderr[-800:]
    assert "Recover the original" in human.stdout

    receipt = _compress_json(work, store, "resp.json")
    assert receipt["recovery_returns"] == "complete_original"


@pytest.mark.parametrize(
    "label, raw",
    [
        ("empty", b""),
        ("one-byte", b"x"),
        ("bom-json", b'\xef\xbb\xbf{"a": [1, 2, 3]}\n'),
        ("deep-json", b"[" * 5000 + b"1" + b"]" * 5000),
    ],
    ids=["empty", "one-byte", "bom-json", "deep-json"],
)
def test_out_is_written_even_when_nothing_was_compressed(workspace, label, raw):
    # A script running `entroly compress in --out out` and then reading `out`
    # must find the (unchanged) content, not a missing file.
    work, store = workspace
    (work / "in.dat").write_bytes(raw)

    result = _run(["compress", "in.dat", "--out", "out.dat"], work, store)

    assert result.returncode == 0, result.stderr[-800:]
    assert (work / "out.dat").exists(), f"{label}: --out file was not written"
    assert (work / "out.dat").read_bytes() == raw


def _run_home(args: list[str], cwd: Path, home: Path) -> subprocess.CompletedProcess:
    # Default store resolution: no ENTROLY_DIR, private HOME.
    env = {k: v for k, v in os.environ.items() if k != "ENTROLY_DIR"}
    env.update(HOME=str(home), ENTROLY_DISABLE_UPDATE_CHECK="1")
    return subprocess.run(
        [sys.executable, "-m", "entroly", *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
    )


def test_recover_from_a_subdirectory_says_where_to_run_it(tmp_path):
    # Recovery is bound to the directory `entroly compress` ran in (by design:
    # tests/test_recovery_store_trust.py pins scope isolation). A bare
    # "No recovery entry" made a present record look lost.
    project, home = tmp_path / "project", tmp_path / "home"
    (project / "src").mkdir(parents=True)
    home.mkdir()
    (project / "resp.json").write_text(_payload(), encoding="utf-8")
    done = _run_home(["compress", "resp.json", "--json"], project, home)
    assert done.returncode == 0, done.stderr[-800:]
    digest = json.loads(done.stdout[done.stdout.find("{"):])["recovery_digest"]

    missed = _run_home(["recover", digest], project / "src", home)

    assert missed.returncode == 1
    assert f"cd {project}" in missed.stdout


def test_recover_with_store_from_another_directory_explains_the_scope(workspace, tmp_path):
    work, store = workspace
    (work / "resp.json").write_text(_payload(), encoding="utf-8")
    digest = _compress_json(work, store, "resp.json")["recovery_digest"]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    missed = _run(["recover", digest, "--store", str(store / "recovery.json")], elsewhere, store)

    assert missed.returncode == 1
    assert "different working directory" in missed.stdout


def test_recover_with_a_truncated_digest_says_what_a_digest_looks_like(workspace):
    work, store = workspace
    missed = _run(["recover", "sha256:0ec354cbc96f"], work, store)
    assert missed.returncode == 1
    assert "sha256:<64 hex" in missed.stdout


def test_recovery_docs_and_help_name_the_real_flag_and_default(workspace):
    # docs/recovery-data-security.md named a `--store-path` flag that does not
    # exist, and `--help` gave a default that only holds with ENTROLY_DIR set.
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "recovery-data-security.md").read_text(encoding="utf-8")
    assert "--store-path" not in doc
    work, store = workspace
    for command in ("compress", "recover"):
        helped = _run([command, "--help"], work, store)
        assert "--store" in helped.stdout
        assert "checkpoints" in helped.stdout, f"{command} --help hides the default store"


def test_session_rescue_docs_point_at_the_store_rescue_writes():
    # The rescue store defaults to ~/.entroly/session_rescue_recovery.json and
    # is not readable by `entroly recover`; the docs said otherwise.
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "session-rescue.md").read_text(encoding="utf-8")
    assert '".entroly/session_rescue_recovery.json"' not in doc
    assert "ENTROLY_COMPRESSION_STORE=.entroly/" not in doc
    source = (root / "entroly" / "session_rescue.py").read_text(encoding="utf-8")
    assert "`entroly recover` finds spans" not in source


def _nested_api_response() -> str:
    # GitHub-issues-shaped: nested label arrays, multi-line bodies and
    # per-record ids, which the columnar and schema forms both decline.
    return json.dumps(
        {
            "total_count": 400,
            "items": [
                {
                    "id": i,
                    "login": f"user{i}",
                    "state": "closed" if i % 3 else "open",
                    "title": f"Fix flaky test in module_{i}  (two spaces kept)",
                    "labels": [
                        {"name": "bug" if i % 2 else "ci", "color": "ededed"},
                        {"name": f"p{i % 3}", "color": "ededed"},
                    ],
                    "created_at": f"2026-10-02T10:{i % 60:02d}:00Z",
                    "body": f"Steps to reproduce:\n1. run pytest\n2. observe failure in test_{i}",
                }
                for i in range(400)
            ],
        },
        indent=2,
    )


def test_readme_compress_example_shrinks_a_nested_api_response(workspace):
    # README: `entroly compress response.json --out small.json`. A realistic
    # nested response came back "left unchanged" (42,529 -> 42,530 tokens).
    work, store = workspace
    original = _nested_api_response()
    (work / "response.json").write_text(original, encoding="utf-8")

    receipt = _compress_json(work, store, "response.json", "--out", "small.json")

    assert receipt["codec"] == "json"
    assert receipt["tokens_after"] < receipt["tokens_before"]
    small = (work / "small.json").read_text(encoding="utf-8")
    assert json.loads(small) == json.loads(original)
    assert "(two spaces kept)" in small, "string contents must stay verbatim"
    restored = _run(
        ["recover", receipt["recovery_digest"], "--out", "restored.json"], work, store
    )
    assert restored.returncode == 0, restored.stdout[-800:]
    assert (work / "restored.json").read_bytes() == original.encode("utf-8")


def test_recover_reports_a_corrupt_store_plainly(workspace):
    # A truncated store surfaced as a bare JSON parser error ("Unterminated
    # string ...") with no file name and no hint that the store was damaged.
    work, store = workspace
    (work / "resp.json").write_text(_payload(), encoding="utf-8")
    digest = _compress_json(work, store, "resp.json")["recovery_digest"]
    path = store / "recovery.json"
    path.write_bytes(path.read_bytes()[:300])

    missed = _run(["recover", digest], work, store)

    assert missed.returncode == 1
    assert "could not be read" in missed.stdout
    assert str(path) in missed.stdout


def test_recover_still_reads_a_store_above_the_write_cap(workspace, monkeypatch):
    # The byte cap bounds writes. Lowering it must not lock a user out of
    # records that are already stored.
    work, store = workspace
    (work / "resp.json").write_bytes(_payload().encode("utf-8"))
    digest = _compress_json(work, store, "resp.json")["recovery_digest"]
    monkeypatch.setenv("ENTROLY_RECOVERY_STORE_MAX_BYTES", "1024")

    result = _run(["recover", digest, "--out", "restored.json"], work, store)

    assert result.returncode == 0, result.stdout[-800:]
    assert (work / "restored.json").read_bytes() == (work / "resp.json").read_bytes()
