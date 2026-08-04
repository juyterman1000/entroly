"""Guards against a simulated benchmark being mistaken for a measured one.

benchmarks/coding_tasks.py is the harness named by
benchmarks/AGENTIC_TASKS_PREREGISTRATION.md as the thing that will answer the
question the product is actually asked: does compression preserve task success
at lower cost? Its real execution path is not wired yet.

Before this guard existed, running it emitted a plausible-looking JSON into
benchmarks/results/ -- the same directory the README cites -- containing
hardcoded token counts and a random-draw pass rate rigged so that Entroly won.
These tests keep that shape from returning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.coding_tasks import (
    REAL_RESULTS_DIR,
    SIMULATED_RESULTS_DIR,
    TASKS,
    BenchmarkHarness,
)


def test_default_run_fails_closed_instead_of_fabricating(tmp_path: Path) -> None:
    """Without an explicit opt-in, the harness must refuse rather than invent."""
    harness = BenchmarkHarness(tmp_path / "ws", model="test-model")
    assert harness.simulate is False

    with pytest.raises(NotImplementedError) as excinfo:
        harness.run(modes=["raw"], runs_per_task=1)

    message = str(excinfo.value)
    # The error has to name the missing work, not just say "not implemented".
    assert "does not call a model" in message
    assert "AGENTIC_TASKS_PREREGISTRATION.md" in message
    assert harness.results == []


def test_simulated_output_is_redirected_out_of_the_real_results_dir(
    tmp_path: Path,
) -> None:
    """Asking for benchmarks/results/ while simulating must not be honoured."""
    harness = BenchmarkHarness(tmp_path / "ws", model="test-model", simulate=True)

    redirected = harness._guard_output_dir(REAL_RESULTS_DIR)
    assert redirected == SIMULATED_RESULTS_DIR

    # A nested path under the real results dir is redirected too.
    nested = harness._guard_output_dir(REAL_RESULTS_DIR / "sub" / "dir")
    assert nested == SIMULATED_RESULTS_DIR


def test_a_measured_run_is_not_redirected(tmp_path: Path) -> None:
    """The guard must not interfere once a real execution path exists."""
    harness = BenchmarkHarness(tmp_path / "ws", model="test-model", simulate=False)
    assert harness._guard_output_dir(REAL_RESULTS_DIR) == REAL_RESULTS_DIR


def test_simulated_report_labels_itself_as_worthless(tmp_path: Path) -> None:
    """A simulated artifact must be unmistakable on inspection."""
    harness = BenchmarkHarness(tmp_path / "ws", model="test-model", simulate=True)
    harness.run(modes=["raw", "entroly"], runs_per_task=1)
    assert harness.results, "simulation should still exercise the plumbing"

    out_dir = tmp_path / "out"
    harness.generate_report(out_dir)

    written = list(out_dir.glob("*.json"))
    assert len(written) == 1, written
    artifact = written[0]

    # Visible before the file is even opened.
    assert artifact.name.startswith("SIMULATED_NOT_EVIDENCE_")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["metadata"]["simulated"] is True
    assert payload["metadata"]["valid_as_evidence"] is False

    limitations = payload["limitations"]
    assert limitations, "a simulated artifact must carry its own limitations"
    joined = " ".join(limitations).lower()
    assert "fabricated" in joined
    assert "no model was called" in joined

    # Every individual trace carries the flag, so a partial copy stays labelled.
    assert all(trace["simulated"] is True for trace in payload["raw_traces"])


def test_simulated_runs_never_land_in_the_committed_results_dir() -> None:
    """No simulated artifact may already be sitting in benchmarks/results/."""
    if not REAL_RESULTS_DIR.exists():
        pytest.skip("benchmarks/results/ not present in this checkout")

    strays = [
        path
        for path in REAL_RESULTS_DIR.glob("*coding_tasks*.json")
    ]
    assert not strays, (
        "simulated coding-task artifacts found in the real results directory: "
        f"{[str(p) for p in strays]}"
    )

    # Belt and braces: nothing in results/ may self-declare as simulated.
    self_declared: list[str] = []
    for path in REAL_RESULTS_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            metadata = payload.get("metadata")
            if isinstance(metadata, dict) and metadata.get("simulated") is True:
                self_declared.append(str(path))
    assert not self_declared, (
        f"artifacts declaring simulated=True in results/: {self_declared}"
    )


def test_tasks_are_declared_with_a_real_test_oracle() -> None:
    """Each task needs an executable oracle; success cannot be self-asserted."""
    assert TASKS, "no tasks declared"
    for task in TASKS:
        assert task.test_command.strip(), f"{task.id} has no test command"
        assert task.setup_command.strip(), f"{task.id} has no setup command"
