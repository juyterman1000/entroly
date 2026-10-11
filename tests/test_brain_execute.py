"""Compute the answer instead of reading the corpus that contains it.

Selection is the wrong instrument for an aggregate question. "How many
exported functions", "which files import this", "what is the largest test
file" each require every candidate file to produce one number, so the model
pays for the whole corpus to learn one line of it. A script answers those
where the files already are and returns only the result.

Because this executes code, the tests below treat the boundaries as the
feature: it is off unless asked for, it cannot be reached through a shell, it
does not hand the script the ambient environment, and it reports a truncated
or timed-out run rather than returning a silent prefix.
"""
from __future__ import annotations

import pytest

from entroly.brain_execute import ENABLE_ENV, brain_execute, brain_execute_enabled


@pytest.fixture()
def workspace(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for index in range(5):
        (src / f"mod{index}.py").write_text(
            "def handler():\n    pass\n" * (index + 2), encoding="utf-8"
        )
    return tmp_path


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")


def test_disabled_unless_asked_for(monkeypatch, workspace) -> None:
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    assert brain_execute_enabled() is False

    result = brain_execute("print('ran')", workspace=workspace)

    assert result.ok is False
    assert "disabled" in (result.error or "")
    assert result.stdout == "", "a disabled tool must not execute anything"


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE"])
def test_only_an_exact_opt_in_enables_it(monkeypatch, value) -> None:
    """A truthy-looking value is not consent; the check is exact."""
    monkeypatch.setenv(ENABLE_ENV, value)
    assert brain_execute_enabled() is False


def test_it_returns_the_conclusion_not_the_corpus(enabled, workspace) -> None:
    result = brain_execute(
        "import pathlib\n"
        "n = sum(f.read_text().count('def ') for f in pathlib.Path('src').glob('*.py'))\n"
        "print(f'{n} functions')\n",
        workspace=workspace,
    )

    assert result.ok, result.error
    assert result.stdout.strip() == "20 functions"

    corpus = sum(
        len(f.read_text(encoding="utf-8")) for f in (workspace / "src").glob("*.py")
    )
    assert len(result.stdout) < corpus / 5, (
        "the point is the ratio between what it reads and what it returns"
    )


def test_the_workspace_is_the_working_directory(enabled, workspace) -> None:
    result = brain_execute("import os; print(os.getcwd())", workspace=workspace)
    assert result.ok
    assert result.stdout.strip() == str(workspace.resolve())


def test_a_failing_script_is_an_observation_not_an_exception(enabled, workspace) -> None:
    """Whatever it printed before failing is still worth returning."""
    result = brain_execute(
        "print('partial result')\nraise ValueError('boom')\n", workspace=workspace
    )

    assert result.ok is False
    assert result.exit_code != 0
    assert "partial result" in result.stdout
    assert "ValueError" in result.stderr


def test_a_slow_script_is_stopped_and_says_so(enabled, workspace) -> None:
    result = brain_execute("import time; time.sleep(30)", workspace=workspace, timeout_s=1)

    assert result.ok is False
    assert "timed out" in (result.error or "")
    assert result.meta.get("timed_out") is True


def test_oversized_output_is_capped_and_reports_the_cap(enabled, workspace) -> None:
    """A silent prefix would read as the whole answer."""
    result = brain_execute("print('x' * 100_000)", workspace=workspace, max_output_bytes=512)

    assert result.truncated is True
    assert "truncated" in result.stdout
    assert "100,001 bytes produced" in result.stdout


def test_the_ambient_environment_is_not_handed_over(enabled, workspace, monkeypatch) -> None:
    """A repository shell is full of credentials the script has no use for."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_example")

    result = brain_execute(
        "import os\n"
        "leaked = [k for k in ('AWS_SECRET_ACCESS_KEY', 'GITHUB_TOKEN') if k in os.environ]\n"
        "print(leaked)\n",
        workspace=workspace,
    )

    assert result.ok, result.error
    assert result.stdout.strip() == "[]"


def test_no_shell_is_involved(enabled, workspace) -> None:
    """The code is a script argument, never a command line."""
    marker = workspace / "pwned.txt"
    result = brain_execute(
        f"print('literal: ' + {str('; touch ' + str(marker))!r})", workspace=workspace
    )

    assert result.ok
    assert not marker.exists(), "shell metacharacters must stay inert"


def test_the_script_is_written_outside_the_workspace(enabled, workspace) -> None:
    """A run must not be able to read or index its own source as repo content."""
    before = {p.name for p in workspace.rglob("*")}
    result = brain_execute(
        "import pathlib\nprint(sorted(p.name for p in pathlib.Path('.').rglob('*.py')))",
        workspace=workspace,
    )

    assert result.ok
    assert "brain_script.py" not in result.stdout
    assert {p.name for p in workspace.rglob("*")} == before


@pytest.mark.parametrize("code", ["", "   ", "\n\t "])
def test_empty_code_is_refused(enabled, workspace, code) -> None:
    assert brain_execute(code, workspace=workspace).error == "code is empty"


def test_oversized_code_is_refused(enabled, workspace) -> None:
    result = brain_execute("#" + "x" * (200 * 1024), workspace=workspace)
    assert result.ok is False
    assert "size limit" in (result.error or "")


def test_a_missing_workspace_is_refused(enabled, tmp_path) -> None:
    result = brain_execute("print(1)", workspace=tmp_path / "nope")
    assert result.ok is False
    assert "workspace" in (result.error or "")
