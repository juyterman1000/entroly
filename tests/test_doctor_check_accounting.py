"""`entroly doctor`'s summary must agree with the markers it just printed.

Four warning branches printed a yellow ``!`` and then incremented
``checks_passed``: config weights not summing to 1.0, a stale index, mild
weight drift, and missing hallucination verifiers. Measured differentially --
one doctor run with a deliberately bad ``tuning_config.json`` and one without --
the printed warnings went from 1 to 2 while the summary stayed
``8/8 checks passed, 1 warning(s)``.

An operator who reads only the last line saw all-green while the body reported a
problem. cli.py's own comments call that the false green the split counter
exists to prevent, so the rule is asserted here rather than left to review.

The marker/bucket mapping is: ``+`` and ``-`` count as passed (``-`` is
informational -- an absent proxy is not a fault), ``x`` failed, ``!`` warned.

Note on isolation: ``doctor`` reads its tuning file from a fixed path inside the
installed package (``cli.py`` uses ``Path(__file__).parent /
"tuning_config.json"``), so the tests that need a bad config have to write
there. Each guards with "skip if the file already exists", which keeps a real
config safe and makes a concurrent run skip rather than clobber. It is not a
substitute for running these serially under xdist.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUNING = REPO_ROOT / "entroly" / "tuning_config.json"

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _run_doctor(tmp_state: Path) -> tuple[dict[str, int], int, int, int]:
    env = dict(os.environ)
    env["ENTROLY_DIR"] = str(tmp_state)
    env["NO_COLOR"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "entroly", "doctor"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=600,
    )
    clean = ANSI.sub("", proc.stdout)
    # The banner above the report also uses these glyphs; the report starts at
    # the "Entroly Doctor" header.
    if "Entroly Doctor" not in clean:
        pytest.skip(f"doctor did not produce a report:\n{proc.stdout}\n{proc.stderr}")
    body = clean.split("Entroly Doctor", 1)[1]

    summary_line = next(
        (line for line in body.splitlines() if "checks passed" in line), None
    )
    assert summary_line, f"no summary line in doctor output:\n{body}"
    body = body.split(summary_line)[0]

    markers = {"+": 0, "-": 0, "x": 0, "!": 0}
    for line in body.splitlines():
        stripped = line.strip()
        if len(stripped) > 1 and stripped[0] in markers and stripped[1] == " ":
            markers[stripped[0]] += 1

    counts = re.search(r"(\d+)/(\d+) checks passed", summary_line)
    assert counts, summary_line
    failed = re.search(r"(\d+) failed", summary_line)
    warned = re.search(r"(\d+) warning", summary_line)
    return (
        markers,
        int(counts.group(1)),
        int(failed.group(1)) if failed else 0,
        int(warned.group(1)) if warned else 0,
    )


def _assert_markers_match(markers, passed, failed, warned, label):
    assert markers["!"] == warned, (
        f"{label}: doctor printed {markers['!']} warning marker(s) but the "
        f"summary counted {warned}; a printed warning is being reported as a "
        f"pass"
    )
    assert markers["x"] == failed, (
        f"{label}: printed {markers['x']} failure marker(s), summary says {failed}"
    )
    assert markers["+"] + markers["-"] == passed, (
        f"{label}: printed {markers['+']} '+' and {markers['-']} '-' markers "
        f"but the summary counted {passed} passed"
    )


def test_doctor_summary_matches_its_markers(tmp_path):
    """Baseline: whatever this environment reports, the counts must agree."""
    markers, passed, failed, warned = _run_doctor(tmp_path / "state")
    assert sum(markers.values()) > 0, "doctor printed no check markers at all"
    _assert_markers_match(markers, passed, failed, warned, "baseline")


def test_a_forced_warning_is_counted_as_a_warning_not_a_pass(tmp_path):
    """Non-vacuous: drive a warning branch instead of hoping for one.

    The baseline test above passes trivially on a clean machine that produces no
    warnings at all, which is exactly how this defect survived. This one creates
    the condition.
    """
    if TUNING.exists():
        pytest.skip("a real tuning_config.json is present; refusing to clobber it")

    before = _run_doctor(tmp_path / "state_before")

    TUNING.write_text(
        json.dumps({"weights": {"relevance": 0.3, "entropy": 0.2}}),  # sums to 0.5
        encoding="utf-8",
    )
    try:
        after = _run_doctor(tmp_path / "state_after")
    finally:
        TUNING.unlink(missing_ok=True)

    markers_b, _, _, warned_b = before
    markers_a, passed_a, failed_a, warned_a = after

    assert markers_a["!"] > markers_b["!"], (
        "the bad tuning_config.json did not produce a new warning marker; "
        f"before={markers_b} after={markers_a}"
    )
    _assert_markers_match(markers_a, passed_a, failed_a, warned_a, "forced warning")
    assert warned_a > warned_b, (
        f"an added warning did not raise the counted warnings ({warned_b} -> "
        f"{warned_a}); it was absorbed into the pass count"
    )


@pytest.mark.parametrize(
    "band,weights",
    [
        # Drift is sum(|w - default|) over recency .30, frequency .25,
        # semantic .25, entropy .20. Each set sums to exactly 1.0 so the
        # config-validity check still passes with a `+` and the only new marker
        # comes from drift.
        ("mild", {"recency": 0.36, "frequency": 0.19, "semantic": 0.25, "entropy": 0.20}),
        ("heavy", {"recency": 0.60, "frequency": 0.10, "semantic": 0.10, "entropy": 0.20}),
    ],
)
def test_weight_drift_is_counted_as_a_warning_not_a_pass(tmp_path, band, weights):
    """Cover both drift branches, isolated from the other checks.

    ``mild`` lands in ``[0.1, 0.3)`` (drift 0.12) and ``heavy`` above 0.3
    (drift 0.60). They are separate branches with separate counters, and only
    one of them was wrong -- so testing a single band let a mutation of the
    other survive.
    """
    if TUNING.exists():
        pytest.skip("a real tuning_config.json is present; refusing to clobber it")

    before = _run_doctor(tmp_path / "state_before")

    TUNING.write_text(json.dumps({"weights": weights}), encoding="utf-8")
    try:
        markers_a, passed_a, failed_a, warned_a = _run_doctor(tmp_path / "state_after")
    finally:
        TUNING.unlink(missing_ok=True)

    assert markers_a["!"] > before[0]["!"], (
        f"{band} drift did not produce a warning marker; "
        f"before={before[0]} after={markers_a}"
    )
    _assert_markers_match(markers_a, passed_a, failed_a, warned_a, f"{band} drift")
    assert warned_a > before[3], (
        f"the {band} drift warning did not raise the counted warnings "
        f"({before[3]} -> {warned_a})"
    )


def test_a_stale_index_is_counted_as_a_warning_not_a_pass(tmp_path):
    """Cover a second warning branch, through a different mechanism.

    Forcing only the config branch left the stale-index branch untested: a
    mutation that reverted it to ``checks_passed += 1`` survived the rest of
    this file. ``doctor`` resolves the index under ``ENTROLY_DIR``, so the
    condition can be built directly.
    """
    fresh_state = tmp_path / "fresh"
    (fresh_state / "checkpoints").mkdir(parents=True)
    (fresh_state / "checkpoints" / "index.json").write_text("{}", encoding="utf-8")
    before = _run_doctor(fresh_state)

    stale_state = tmp_path / "stale"
    (stale_state / "checkpoints").mkdir(parents=True)
    stale_file = stale_state / "checkpoints" / "index.json"
    stale_file.write_text("{}", encoding="utf-8")
    two_days_ago = stale_file.stat().st_mtime - 48 * 3600
    os.utime(stale_file, (two_days_ago, two_days_ago))

    markers_a, passed_a, failed_a, warned_a = _run_doctor(stale_state)

    assert markers_a["!"] > before[0]["!"], (
        "a 48h-old index did not produce a warning marker; "
        f"fresh={before[0]} stale={markers_a}"
    )
    _assert_markers_match(markers_a, passed_a, failed_a, warned_a, "stale index")
    assert warned_a > before[3], (
        f"the stale-index warning did not raise the counted warnings "
        f"({before[3]} -> {warned_a})"
    )
