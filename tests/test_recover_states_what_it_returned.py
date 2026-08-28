"""`recover` must say what it handed back, because that differs by codec.

The command advertised "the exact original bytes". For `json` that is true --
the store holds the complete original. For `code` the store holds the bodies
elided from a skeleton, so a user who ran `recover` on a Python file got
function bodies with no imports and no signatures: a fragment that is not even
syntactically valid, presented as though it were their file.

The store always knew. `RecoveryReference.note` records exactly which of the
two it is, and the command simply never printed it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PY_SOURCE = (
    "import hashlib\n\n\n"
    "def hash_password(password, salt):\n"
    "    return hashlib.sha256((salt + password).encode()).hexdigest()\n"
)


def _run(args, cwd, store):
    return subprocess.run(
        [sys.executable, "-m", "entroly.cli", *args, "--store", str(store)],
        capture_output=True, text=True, cwd=str(cwd), timeout=300,
        env={**__import__("os").environ, "ENTROLY_NO_SELF_HEAL": "1"},
    )


def _compress(work: Path, store: Path, name: str) -> dict:
    result = _run(["compress", name, "--json"], work, store)
    assert result.returncode == 0, result.stderr[-800:]
    start = result.stdout.index("{")
    return json.loads(result.stdout[start:result.stdout.rindex("}") + 1])


@pytest.mark.timeout(300)
def test_partial_code_recovery_says_it_is_partial(tmp_path):
    store = tmp_path / "store"
    (tmp_path / "auth.py").write_text(PY_SOURCE, encoding="utf-8")

    receipt = _compress(tmp_path, store, "auth.py")
    assert receipt["codec"] == "code"

    result = _run(
        ["recover", receipt["recovery_digest"], "--out", str(tmp_path / "out.py")],
        tmp_path, store,
    )
    assert result.returncode == 0

    # The user must be told this is not the whole file.
    assert "elided" in result.stdout, (
        "recover printed only a byte count, so a fragment of a Python file "
        f"looked like the file: {result.stdout!r}"
    )

    # Premise: it really is partial. If this stops holding, the assertion
    # above would be checking a message about a condition that no longer
    # exists.
    recovered = (tmp_path / "out.py").read_bytes()
    assert recovered != PY_SOURCE.encode("utf-8")


@pytest.mark.timeout(300)
def test_complete_json_recovery_is_not_labelled_partial(tmp_path):
    store = tmp_path / "store"
    raw = json.dumps({"items": [{"sku": f"S{i}"} for i in range(30)]}, indent=2)
    (tmp_path / "payload.json").write_text(raw, encoding="utf-8")
    # Read back rather than re-encoding the string: write_text applies the
    # platform newline translation, so on Windows the file holds CRLF while
    # raw.encode() is still LF. Comparing against the string would fail for a
    # reason that has nothing to do with recovery.
    original_bytes = (tmp_path / "payload.json").read_bytes()

    receipt = _compress(tmp_path, store, "payload.json")
    result = _run(
        ["recover", receipt["recovery_digest"], "--out", str(tmp_path / "out.json")],
        tmp_path, store,
    )
    assert result.returncode == 0

    assert (tmp_path / "out.json").read_bytes() == original_bytes
    assert "elided" not in result.stdout
    # A complete recovery must not carry an instruction to combine it with
    # anything -- there is nothing left to combine.
    assert "combine" not in result.stdout.lower()


@pytest.mark.timeout(300)
def test_piped_recovery_keeps_stdout_pure(tmp_path):
    """The note must not corrupt a redirect."""
    store = tmp_path / "store"
    (tmp_path / "auth.py").write_text(PY_SOURCE, encoding="utf-8")
    receipt = _compress(tmp_path, store, "auth.py")

    result = _run(["recover", receipt["recovery_digest"]], tmp_path, store)

    assert result.returncode == 0
    assert "elided" in result.stderr, "the note must still be reported"
    assert "elided" not in result.stdout, (
        "stdout carries recovered bytes and must stay pipeable"
    )
