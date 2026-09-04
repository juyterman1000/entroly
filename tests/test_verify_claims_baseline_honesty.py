"""`verify-claims` must measure savings against the same baseline as `simulate`.

Onboarding tells a new user to run `entroly verify-claims` (step 1) and then
`entroly simulate` (step 2). `cmd_simulate` measures against
``min(total_tokens, 32_000)`` -- a context window someone might plausibly fill --
and reports ``baseline_tokens`` alongside the percentage. `verify-claims`
divided by the entire indexed corpus instead, which credits every fragment that
was never going to be sent; ``engine._honest_tokens_saved`` documents that as the
error its own baseline helper exists to prevent.

Measured on this repository before the fix: verify-claims printed 99.4% against
769,339 corpus tokens while simulate reported 89.1% for the same repo, and only
simulate disclosed what it compared against.

CLAUDE.md's benchmark-honesty rule is that a claim carries its baseline, so
these assert both halves: the right baseline, and that it is reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entroly.engine import naive_context_baseline  # noqa: E402
from entroly.verify_claims import run as run_verify_claims  # noqa: E402


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    """A small project so the pass indexes quickly."""
    src = tmp_path / "proj"
    (src / "pkg").mkdir(parents=True)
    for i in range(4):
        (src / "pkg" / f"module_{i}.py").write_text(
            "\n".join(
                [
                    f'"""Module {i} of the main module structure."""',
                    f"def handler_{i}(payload):",
                    f"    # explain the main module structure for {i}",
                    "    total = 0",
                    "    for item in payload:",
                    "        total += item",
                    "    return total",
                    "",
                ]
                * 12
            ),
            encoding="utf-8",
        )
    monkeypatch.chdir(src)
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    return src


def _report(tmp_path, tiny_repo) -> dict:
    out = tmp_path / "verification.json"
    run_verify_claims(output=str(out), max_files=8)
    assert out.exists(), "verify-claims wrote no report"
    return json.loads(out.read_text(encoding="utf-8"))


def test_savings_are_measured_against_the_capped_baseline(tmp_path, tiny_repo):
    report = _report(tmp_path, tiny_repo)

    total = report["total_tokens"]
    expected = naive_context_baseline(total)

    assert "baseline_tokens" in report, (
        "the report states a savings percentage without saying what it is "
        "measured against"
    )
    assert report["baseline_tokens"] == expected, (
        f"baseline is {report['baseline_tokens']} but naive_context_baseline"
        f"({total}) is {expected}; dividing by the whole corpus credits "
        "fragments that were never going to be sent"
    )


def test_the_reported_percentage_matches_the_reported_baseline(tmp_path, tiny_repo):
    """The number and its stated baseline have to be the same calculation."""
    report = _report(tmp_path, tiny_repo)

    used = report["tokens_used"]
    baseline = report["baseline_tokens"]
    stated = report["savings_pct"]

    if used == 0 or baseline == 0:
        assert stated == 0.0, "selecting nothing is not a saving"
        return

    recomputed = round((1 - used / baseline) * 100, 1)
    assert stated == pytest.approx(recomputed, abs=0.1), (
        f"reported {stated}% but {used} tokens against a {baseline}-token "
        f"baseline is {recomputed}%"
    )
    assert 0.0 <= stated <= 100.0, f"savings of {stated}% is not a proportion"


def test_the_two_onboarding_commands_share_one_baseline_rule(tmp_path, tiny_repo):
    """Guard the specific divergence: corpus total vs capped window.

    If ``verify-claims`` regressed to dividing by ``total_tokens``, this catches
    it whenever the corpus exceeds the cap -- so the fixture is padded past
    32,000 tokens to make the two rules give different answers.
    """
    # Unique bodies: repeating one block makes every chunk a near-duplicate of
    # the last, and SimHash dedup collapses them, leaving a corpus far under the
    # cap. Measured with a repeated block: 2,038 tokens instead of tens of
    # thousands, which silently skipped this test.
    for f in range(6):
        (tiny_repo / "pkg" / f"bulk_{f}.py").write_text(
            "\n".join(
                f"def bulk_{f}_{n}(records):\n"
                f"    total_{n} = {n * 7 + f}\n"
                f"    for record in records:\n"
                f"        total_{n} += record.value_{n} * {n % 89 + 3}\n"
                f"    return total_{n} - {n * 17 % 103}\n"
                for n in range(500)
            ),
            encoding="utf-8",
        )

    report = _report(tmp_path, tiny_repo)
    total = report["total_tokens"]
    if total <= 32_000:
        pytest.skip(f"corpus is only {total} tokens; the two rules agree here")

    assert report["baseline_tokens"] == 32_000, (
        f"corpus is {total} tokens but the baseline is "
        f"{report['baseline_tokens']}; the cap was not applied"
    )
    assert report["baseline_tokens"] != total, (
        "the baseline is the whole corpus again"
    )
