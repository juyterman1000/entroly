"""The first-run screen may not carry a savings percentage.

`saved = max(0, baseline - selected_tokens)` with
`baseline = min(total_tokens, 32_000)` in entroly/cli.py pins the figure at
or above 75% for a budget of 8,000 before selection has run. It is budget
arithmetic wearing the costume of a measurement, and a first-run screen is
the widest possible distribution for it.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / ".claude-plugin" / "commands" / "entroly-first-run.md"

PERCENTAGE = re.compile(r"\d{1,3}\s?%")


def test_first_run_command_exists() -> None:
    assert COMMAND.is_file()


def test_first_run_copy_states_no_savings_percentage() -> None:
    matches = PERCENTAGE.findall(COMMAND.read_text(encoding="utf-8"))

    assert matches == [], f"first-run copy must not quote a percentage: {matches}"


def test_first_run_copy_shows_recovery_handles() -> None:
    text = COMMAND.read_text(encoding="utf-8").lower()

    # The differentiator is not that context got smaller. It is that what
    # was left out is named and recoverable.
    assert "recover" in text
    assert "omitted" in text
