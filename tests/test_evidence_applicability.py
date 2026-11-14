"""A recorded verification is reused only while everything it read is unchanged.

Each scenario runs a real pytest subprocess under the tracer, then changes one
thing and asks whether the recorded result still applies.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from entroly.evidence_applicability import (
    REUSE,
    REVALIDATE,
    UNKNOWN,
    assess,
    record_digest,
    run_and_record,
    verify_record,
)
from entroly.work_graph import WorkGraph, WorkGraphUnavailableError


def _require_native() -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:  # pragma: no cover - depends on the install
        pytest.skip(str(exc))


def _environment() -> dict[str, str]:
    keep = ("PATH", "HOME", "SYSTEMROOT", "LANG", "TMPDIR", "TEMP", "TMP", "USERPROFILE")
    return {key: os.environ[key] for key in keep if key in os.environ}


def _project(root: Path) -> Path:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["src"]\naddopts = "-q -p no:cacheprovider"\n',
        encoding="utf-8",
    )
    (root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "pkg" / "tax.py").write_text(
        "import json\nfrom pathlib import Path\n\n"
        "_RATES = Path(__file__).with_name('rates.json')\n\n\n"
        "def rate():\n    return json.loads(_RATES.read_text())['r']\n",
        encoding="utf-8",
    )
    (root / "src" / "pkg" / "rates.json").write_text('{"r": 2}\n', encoding="utf-8")
    (root / "src" / "pkg" / "other.py").write_text("def other():\n    return 1\n", encoding="utf-8")
    (root / "tests" / "test_tax.py").write_text(
        "from pkg.tax import rate\n\n\ndef test_rate():\n    assert rate() == 2\n",
        encoding="utf-8",
    )
    return root


def _record(project: Path, work: Path, argv: list[str] | None = None, env: dict | None = None):
    return run_and_record(
        project,
        argv or [sys.executable, "-m", "pytest", "tests/test_tax.py"],
        environ=env or _environment(),
        work_dir=work,
        repository_id="test-repo",
    )


def _decision(record, project: Path, env: dict | None = None) -> dict:
    return assess(record, project, environ=env or _environment())


def test_reuse_until_an_input_the_verification_read_changes(tmp_path: Path) -> None:
    _require_native()
    project = _project(tmp_path / "proj")
    record, _ = _record(project, tmp_path / "work")
    assert record["result"]["verdict"] == "passed"
    assert record["trace"]["complete"], record["trace"]
    assert verify_record(record) == []

    assert _decision(record, project)["decision"] == REUSE

    # A module the verification never imported: still applies.
    other = project / "src" / "pkg" / "other.py"
    other.write_text("def other():\n    return 2\n", encoding="utf-8")
    assert _decision(record, project)["decision"] == REUSE

    # A data file read at run time, not a Python file: no longer applies.
    rates = project / "src" / "pkg" / "rates.json"
    rates.write_text('{"r": 3}\n', encoding="utf-8")
    decision = _decision(record, project)
    assert decision["decision"] == REVALIDATE
    assert {"kind": "file", "key": "src/pkg/rates.json"} in [
        {"kind": item["kind"], "key": item["key"]} for item in decision["changed"]
    ]
    rates.write_text('{"r": 2}\n', encoding="utf-8")
    assert _decision(record, project)["decision"] == REUSE

    # The test runner's configuration file.
    config = project / "pyproject.toml"
    original = config.read_text(encoding="utf-8")
    config.write_text(original.replace("-q", "-q -W error"), encoding="utf-8")
    assert _decision(record, project)["decision"] == REVALIDATE
    config.write_text(original, encoding="utf-8")


def test_a_new_package_that_shadows_an_imported_module_is_detected(tmp_path: Path) -> None:
    """No existing file changes, yet ``import pkg.tax`` would resolve elsewhere."""
    _require_native()
    project = _project(tmp_path / "proj")
    record, _ = _record(project, tmp_path / "work")
    shadow = project / "src" / "pkg" / "tax"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("def rate():\n    return 5\n", encoding="utf-8")
    decision = _decision(record, project)
    assert decision["decision"] == REVALIDATE
    assert any(item["key"] == "src/pkg" for item in decision["changed"])
    shutil.rmtree(shadow)

    # An unrelated new directory beside it is not a module the run asked for.
    (project / "src" / "pkg" / "unrelated_assets").mkdir()
    assert _decision(record, project)["decision"] == REUSE


def test_a_new_entry_in_a_directory_the_runner_scanned_is_conservatively_revalidated(
    tmp_path: Path,
) -> None:
    """Known precision limit, pinned so it cannot change silently.

    pytest's directory collector scans the project root and the test directory
    in full, even for a single-file argument. Narrowing that listing would need
    pytest-specific rules about which siblings it ignores, so any new name
    there revalidates. Editing an existing file nothing read still reuses.
    """
    _require_native()
    project = _project(tmp_path / "proj")
    record, _ = _record(project, tmp_path / "work")
    (project / "README.md").write_text("draft\n", encoding="utf-8")
    assert _decision(record, project)["decision"] == REVALIDATE


def test_a_new_conftest_the_runner_probed_for_is_detected(tmp_path: Path) -> None:
    _require_native()
    project = _project(tmp_path / "proj")
    record, _ = _record(project, tmp_path / "work")
    (project / "tests" / "conftest.py").write_text("import pytest\n", encoding="utf-8")
    decision = _decision(record, project)
    assert decision["decision"] == REVALIDATE
    assert any(item["key"] == "tests/conftest.py" for item in decision["changed"])


def test_environment_variables_read_are_bound_and_others_are_not(tmp_path: Path) -> None:
    _require_native()
    project = _project(tmp_path / "proj")
    record, _ = _record(project, tmp_path / "work")
    keys = {item["key"] for item in record["inputs"] if item["kind"] == "env"}
    assert "PYTEST_ADDOPTS" in keys  # pytest reads it even when it is unset

    env = _environment()
    env["PYTEST_ADDOPTS"] = "-W error"
    decision = _decision(record, project, env)
    assert decision["decision"] == REVALIDATE
    assert [item["key"] for item in decision["changed"]] == ["PYTEST_ADDOPTS"]

    unrelated = _environment()
    unrelated["EDITOR_FOR_THIS_TEST"] = "vi"
    assert _decision(record, project, unrelated)["decision"] == REUSE


def test_the_record_stores_digests_not_values(tmp_path: Path) -> None:
    _require_native()
    project = _project(tmp_path / "proj")
    (project / "tests" / "test_tax.py").write_text(
        "import os\nfrom pkg.tax import rate\n\n\n"
        "def test_rate():\n    assert os.environ.get('SECRET_TOKEN_FOR_TEST')\n    assert rate() == 2\n",
        encoding="utf-8",
    )
    env = _environment()
    env["SECRET_TOKEN_FOR_TEST"] = "s3cr3t-value-never-stored"
    record, output = _record(project, tmp_path / "work", env=env)
    serialized = json.dumps(record)
    assert record["result"]["verdict"] == "passed"
    assert "s3cr3t-value-never-stored" not in serialized
    assert '"r": 2' not in serialized and "def rate" not in serialized
    assert "SECRET_TOKEN_FOR_TEST" in {item["key"] for item in record["inputs"]}
    assert "stdout_tail" in output  # shown to the caller, never stored


def test_a_failed_verification_is_recorded_as_failed(tmp_path: Path) -> None:
    _require_native()
    project = _project(tmp_path / "proj")
    (project / "src" / "pkg" / "rates.json").write_text('{"r": 9}\n', encoding="utf-8")
    record, _ = _record(project, tmp_path / "work")
    assert record["result"]["verdict"] == "failed"
    assert record["result"]["exit_code"] != 0
    assert _decision(record, project)["decision"] == REUSE  # the failure still applies


def test_tampering_or_substitution_makes_applicability_unknown(tmp_path: Path) -> None:
    _require_native()
    project = _project(tmp_path / "proj")
    record, _ = _record(project, tmp_path / "work")
    handed_over = record_digest(record)

    forged = json.loads(json.dumps(record))
    forged["result"]["verdict"] = "failed"
    assert verify_record(forged)
    assert _decision(forged, project)["decision"] == UNKNOWN

    dropped = json.loads(json.dumps(record))
    dropped["inputs"] = [i for i in dropped["inputs"] if i["key"] != "src/pkg/rates.json"]
    assert _decision(dropped, project)["decision"] == UNKNOWN

    # A self-consistent record that is not the one handed over.
    other, _ = _record(project, tmp_path / "work2")
    decision = assess(other, project, environ=_environment(), expected_digest=handed_over)
    assert decision["decision"] == UNKNOWN
    assert "record-not-the-one-handed-off" in decision["reasons"]


def test_a_command_outside_the_tracer_is_unknown(tmp_path: Path) -> None:
    _require_native()
    project = _project(tmp_path / "proj")
    record, _ = _record(project, tmp_path / "work", argv=[sys.executable, "-S", "-c", "pass"])
    assert not record["trace"]["complete"]
    assert "command-not-traced" in record["trace"]["reasons"]
    assert _decision(record, project)["decision"] == UNKNOWN


def test_an_untraced_child_process_makes_the_record_incomplete(tmp_path: Path) -> None:
    _require_native()
    project = _project(tmp_path / "proj")
    script = (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-S', '-c', 'pass'], check=True)\n"
    )
    record, _ = _record(project, tmp_path / "work", argv=[sys.executable, "-c", script])
    assert "untraced-subprocess" in record["trace"]["reasons"]
    assert _decision(record, project)["decision"] == UNKNOWN


def test_the_resolved_executable_is_bound(tmp_path: Path) -> None:
    """The same argv run by a different interpreter is a different verification."""
    _require_native()
    if os.name != "posix":  # pragma: no cover - the launcher below is a shell script
        pytest.skip("POSIX launcher script")
    project = _project(tmp_path / "proj")
    first = tmp_path / "bin-a"
    first.mkdir()
    launcher = first / "verify-python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    env = _environment()
    env["PATH"] = str(first) + os.pathsep + env.get("PATH", "")
    record, _ = _record(
        project, tmp_path / "work", argv=["verify-python", "-m", "pytest", "tests/test_tax.py"], env=env,
    )
    assert record["result"]["verdict"] == "passed"
    assert _decision(record, project, env)["decision"] == REUSE

    second = tmp_path / "bin-b"
    second.mkdir()
    (second / "verify-python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (second / "verify-python").chmod(0o755)
    switched = dict(env, PATH=str(second) + os.pathsep + _environment().get("PATH", ""))
    decision = _decision(record, project, switched)
    assert decision["decision"] == REVALIDATE
    assert any(item["kind"] == "executable" for item in decision["changed"])
