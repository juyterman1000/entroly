"""Create the frozen v5 recovery-resilience revalidation contract.

Temporary stabilization helper. The builder removes this file before creating the
final branch.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

v4_path = ROOT / "benchmarks/recovery_resilience_protocol_v4.json"
v5_path = ROOT / "benchmarks/recovery_resilience_protocol_v5.json"
protocol = json.loads(v4_path.read_text(encoding="utf-8"))
protocol["schema_version"] = "entroly.recovery-resilience-protocol.v5"
protocol["frozen_at"] = "2026-07-25T00:00:00Z"
protocol["comparison"]["entroly"] = "1.0.66 source"
protocol["supersedes"] = {
    "protocol": "benchmarks/recovery_resilience_protocol_v4.json",
    "artifact": "benchmarks/results/recovery_resilience_holdout_revalidation_v4.json",
    "reason": (
        "Re-anchor after the recovery store was corrected to preserve CRLF/LF "
        "and trailing newlines exactly. The implementation hash therefore "
        "changed for a user-visible integrity fix. The v4 protocol and artifact "
        "remain immutable. Re-run the same frozen 66-entry matrix without "
        "changing gates or claim policy; both systems passing remains parity, "
        "not leadership."
    ),
}
protocol["suites"]["recovery_resilience"]["holdout_policy"] = (
    "Run once after the exact line-ending recovery correction to re-anchor the "
    "implementation hash. Both systems passing means parity, not leadership."
)
v5_path.write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

benchmark_path = ROOT / "benchmarks/recovery_resilience.py"
benchmark = benchmark_path.read_text(encoding="utf-8")
old = '''PRIOR_REVALIDATION_PROTOCOL_PATH_V3 = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v3.json"
)
CURRENT_REVALIDATION_PROTOCOL_PATH = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v4.json"
)
KNOWN_PROTOCOL_PATHS = (
    PROTOCOL_PATH,
    REVALIDATION_PROTOCOL_PATH,
    PRIOR_REVALIDATION_PROTOCOL_PATH_V3,
    CURRENT_REVALIDATION_PROTOCOL_PATH,
)
'''
new = '''PRIOR_REVALIDATION_PROTOCOL_PATH_V3 = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v3.json"
)
PRIOR_REVALIDATION_PROTOCOL_PATH_V4 = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v4.json"
)
CURRENT_REVALIDATION_PROTOCOL_PATH = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v5.json"
)
KNOWN_PROTOCOL_PATHS = (
    PROTOCOL_PATH,
    REVALIDATION_PROTOCOL_PATH,
    PRIOR_REVALIDATION_PROTOCOL_PATH_V3,
    PRIOR_REVALIDATION_PROTOCOL_PATH_V4,
    CURRENT_REVALIDATION_PROTOCOL_PATH,
)
'''
if benchmark.count(old) != 1:
    raise SystemExit("recovery-resilience protocol registry changed")
benchmark = benchmark.replace(old, new, 1)

python_command_old = '''        command = [
            str(Path(python).resolve()),
            "-m",
'''
python_command_new = '''        command = [
            # Preserve a virtualenv launcher's symlink path. Resolving it would
            # bypass the virtualenv and execute the base interpreter without the
            # benchmark participant's installed site-packages.
            str(Path(python).absolute()),
            "-m",
'''
if benchmark.count(python_command_old) != 1:
    raise SystemExit("recovery-resilience interpreter command changed")
benchmark = benchmark.replace(python_command_old, python_command_new, 1)
benchmark_path.write_text(benchmark, encoding="utf-8")

artifact_old = "recovery_resilience_holdout_revalidation_v4.json"
artifact_new = "recovery_resilience_holdout_revalidation_v5.json"
for relative in (
    "README.md",
    "docs/public-evidence.md",
    "scripts/readme_proof.py",
    "docs/benchmarks/competitive-evidence-matrix.md",
):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if artifact_old not in text:
        raise SystemExit(f"{relative}: expected v4 artifact reference is missing")
    text = text.replace(artifact_old, artifact_new)
    if relative == "docs/public-evidence.md":
        text = text.replace(
            "The committed v4 recovery-resilience revalidation",
            "The committed v5 recovery-resilience revalidation",
        )
    path.write_text(text, encoding="utf-8")

test_path = ROOT / "tests/test_recovery_resilience.py"
tests = test_path.read_text(encoding="utf-8")
if "import sys\n" not in tests:
    import_anchor = "import json\n"
    if tests.count(import_anchor) != 1:
        raise SystemExit("recovery-resilience import anchor changed")
    tests = tests.replace(import_anchor, import_anchor + "import sys\n", 1)

venv_test = '''

def test_adapter_preserves_virtualenv_launcher_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    try:
        launcher.symlink_to(Path(sys.executable))
    except OSError:
        pytest.skip("platform does not permit symlink creation")

    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = json.dumps({"system": "headroom"})
        stderr = ""

    def fake_run(command: list[str], **kwargs: object) -> _Completed:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(resilience.subprocess, "run", fake_run)

    result = resilience._invoke_adapter(
        str(launcher),
        "headroom",
        {"workers": 1, "entries_per_worker": 1, "seed": 7},
        timeout=1.0,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == str(launcher.absolute())
    assert command[0] != str(launcher.resolve())
    assert result["system"] == "headroom"
'''
venv_anchor = (
    "\n\ndef test_committed_holdout_is_current_verified_and_scoped_in_evidence_policy"
)
if "test_adapter_preserves_virtualenv_launcher_path" not in tests:
    if tests.count(venv_anchor) != 1:
        raise SystemExit("recovery-resilience venv-test anchor changed")
    tests = tests.replace(venv_anchor, venv_test + venv_anchor, 1)

if artifact_old not in tests:
    raise SystemExit("tests/test_recovery_resilience.py: v4 artifact reference is missing")
tests = tests.replace(artifact_old, artifact_new)
test_path.write_text(tests, encoding="utf-8")
