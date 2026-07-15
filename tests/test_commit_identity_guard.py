from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "commit-identity-guard.yml"
OWNER_EMAIL = "208309368+juyterman1000@users.noreply.github.com"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"


def _guard_script() -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    marker = "          python - <<'PY'\n"
    script = workflow.split(marker, 1)[1].split("\n          PY", 1)[0]
    return textwrap.dedent(script)


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _identity_env(
    *, author_name: str, author_email: str, committer_name: str, committer_email: str
) -> dict[str, str]:
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": author_name,
        "GIT_AUTHOR_EMAIL": author_email,
        "GIT_COMMITTER_NAME": committer_name,
        "GIT_COMMITTER_EMAIL": committer_email,
    }


def _write_release_files(repo: Path, version: str, sha256: str) -> None:
    formula = repo / "packaging" / "homebrew" / "entroly.rb"
    release_test = repo / "tests" / "test_release_surface.py"
    formula.parent.mkdir(parents=True, exist_ok=True)
    release_test.parent.mkdir(parents=True, exist_ok=True)
    formula.write_text(
        "\n".join(
            (
                "class Entroly < Formula",
                f'  url "https://files.pythonhosted.org/packages/source/e/entroly/entroly-{version}.tar.gz"',
                f'  sha256 "{sha256}"',
                "end",
                "",
            )
        ),
        encoding="utf-8",
    )
    release_test.write_text(
        "\n".join(
            (
                f'HOMEBREW_FORMULA_VERSION = "{version}"',
                f'HOMEBREW_FORMULA_SHA256 = "{sha256}"',
                "",
            )
        ),
        encoding="utf-8",
    )


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _write_release_files(repo, "1.0.60", "a" * 64)
    _git(repo, "add", ".")
    owner = _identity_env(
        author_name="juyterman1000",
        author_email=OWNER_EMAIL,
        committer_name="juyterman1000",
        committer_email=OWNER_EMAIL,
    )
    _git(repo, "commit", "-m", "Release baseline", env=owner)
    return repo, _git(repo, "rev-parse", "HEAD")


def _homebrew_bot_commit(
    repo: Path,
    *,
    include_owner_trailer: bool = True,
    include_unrelated_path: bool = False,
) -> str:
    _write_release_files(repo, "1.0.61", "b" * 64)
    if include_unrelated_path:
        (repo / "README.md").write_text("unexpected\n", encoding="utf-8")
    _git(repo, "add", ".")
    args = ["commit", "-m", "Update Homebrew formula for 1.0.61 (#141)"]
    if include_owner_trailer:
        args.extend(["-m", f"Co-authored-by: juyterman1000 <{OWNER_EMAIL}>"])
    bot = _identity_env(
        author_name="github-actions[bot]",
        author_email=BOT_EMAIL,
        committer_name="GitHub",
        committer_email="noreply@github.com",
    )
    _git(repo, *args, env=bot)
    return _git(repo, "rev-parse", "HEAD")


def _run_guard(
    repo: Path,
    tmp_path: Path,
    *,
    event_name: str,
    event: dict[str, object],
) -> subprocess.CompletedProcess[str]:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    env = {
        **os.environ,
        "EXPECTED_LOGIN": "juyterman1000",
        "EXPECTED_NOREPLY_EMAIL": OWNER_EMAIL,
        "EXPECTED_ACCOUNT_EMAIL": "fastrunner10090@gmail.com",
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_EVENT_PATH": str(event_path),
    }
    return subprocess.run(
        [sys.executable, "-c", _guard_script()],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def _push_event(before: str, after: str, *, sender: str = "juyterman1000") -> dict[str, object]:
    return {
        "before": before,
        "after": after,
        "ref": "refs/heads/main",
        "sender": {"login": sender},
    }


def test_owner_merged_homebrew_automation_commit_is_allowed(tmp_path: Path) -> None:
    repo, before = _init_repo(tmp_path)
    after = _homebrew_bot_commit(repo)

    result = _run_guard(
        repo,
        tmp_path,
        event_name="push",
        event=_push_event(before, after),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Commit identity OK" in result.stdout


def test_homebrew_automation_commit_cannot_change_an_unrelated_path(tmp_path: Path) -> None:
    repo, before = _init_repo(tmp_path)
    after = _homebrew_bot_commit(repo, include_unrelated_path=True)

    result = _run_guard(
        repo,
        tmp_path,
        event_name="push",
        event=_push_event(before, after),
    )

    assert result.returncode == 1
    assert "author is github-actions[bot]" in result.stdout


def test_homebrew_automation_commit_requires_owner_coauthor(tmp_path: Path) -> None:
    repo, before = _init_repo(tmp_path)
    after = _homebrew_bot_commit(repo, include_owner_trailer=False)

    result = _run_guard(
        repo,
        tmp_path,
        event_name="push",
        event=_push_event(before, after),
    )

    assert result.returncode == 1
    assert "author is github-actions[bot]" in result.stdout


def test_homebrew_automation_commit_requires_owner_triggered_main_push(tmp_path: Path) -> None:
    repo, before = _init_repo(tmp_path)
    after = _homebrew_bot_commit(repo)

    result = _run_guard(
        repo,
        tmp_path,
        event_name="push",
        event=_push_event(before, after, sender="github-actions[bot]"),
    )

    assert result.returncode == 1
    assert "author is github-actions[bot]" in result.stdout


def test_bot_author_exception_is_never_used_for_pull_requests(tmp_path: Path) -> None:
    repo, before = _init_repo(tmp_path)
    after = _homebrew_bot_commit(repo)
    event = {
        "pull_request": {
            "base": {"sha": before},
            "head": {"sha": after},
        }
    }

    result = _run_guard(
        repo,
        tmp_path,
        event_name="pull_request",
        event=event,
    )

    assert result.returncode == 1
    assert "author is github-actions[bot]" in result.stdout
