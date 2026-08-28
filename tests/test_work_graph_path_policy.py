"""Recovery must not disclose secret-shaped filenames or drown work in vendor trees.

Dogfooded finding. `git status --porcelain` omits git-ignored files, so a tidy
repository looks clean and the gap stays invisible. Against a repository where
`.env`, `id_rsa`, and `node_modules/` were untracked but never ignored, an
interrupted-agent recovery reported:

    changed_paths: ['.env', 'app.py', 'id_rsa', 'node_modules/x.js']

Content was already safe -- the digest layer never returns source bytes, and no
secret value appeared. The filename was the leak: `id_rsa` discloses that a
private key exists, and a vendored tree buries the one file the agent actually
edited.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from entroly.work_graph_path_policy import (
    GENERATED,
    ORDINARY,
    SENSITIVE,
    apply_policy,
    classify,
    sensitive_id,
)


class TestClassification:
    @pytest.mark.parametrize(
        "path",
        [
            ".env", ".env.local", "config/.env.production",
            "id_rsa", "id_ed25519", "deploy/id_rsa.pub",
            "server.pem", "private.key", "certs/app.p12",
            "credentials.json", "gcp-credentials.json",
            ".npmrc", ".pypirc", ".netrc",
            "service-account-prod.json", "kubeconfig",
        ],
    )
    def test_secret_shaped_names_are_withheld(self, path):
        classification, rule = classify(path)
        assert classification == SENSITIVE, f"{path} was not withheld ({rule})"

    @pytest.mark.parametrize(
        "path",
        [
            "node_modules/left-pad/index.js", "dist/bundle.js",
            "build/out.o", "target/debug/app", "vendor/lib/x.go",
            ".venv/lib/site-packages/x.py", "__pycache__/mod.cpython-310.pyc",
            "coverage/lcov.info", "app.min.js", "styles.min.css",
            "bundle.js.map", "schema_pb2.py", "model.freezed.dart",
        ],
    )
    def test_machine_produced_files_are_omitted(self, path):
        classification, rule = classify(path)
        assert classification == GENERATED, f"{path} was not omitted ({rule})"

    @pytest.mark.parametrize(
        "path",
        [
            "app.py", "src/main.rs", "README.md",
            "package-lock.json", "poetry.lock", "Cargo.lock",
            "environment.py", "node_modules_notes.md", "keyboard.py",
            "src/secretary.py",
        ],
    )
    def test_ordinary_work_is_reported_unchanged(self, path):
        classification, _rule = classify(path)
        assert classification == ORDINARY, f"{path} was wrongly reclassified"

    def test_lockfiles_are_work_not_noise(self):
        # A changed lockfile is a real, reviewable change. Dropping it as
        # "generated" would hide a dependency bump from the recovering agent.
        for path in ("package-lock.json", "yarn.lock", "Cargo.lock", "uv.lock"):
            assert classify(path)[0] == ORDINARY

    def test_a_key_inside_a_vendor_tree_is_still_withheld(self):
        # Sensitive must win over generated: dropping it silently is worse
        # than reporting an opaque id.
        classification, _ = classify("node_modules/.bin/id_rsa")
        assert classification == SENSITIVE


class TestOpaqueIdentifiers:
    def test_id_is_stable_for_the_same_path(self):
        assert sensitive_id(".env") == sensitive_id(".env")

    def test_id_differs_between_paths(self):
        assert sensitive_id(".env") != sensitive_id("id_rsa")

    def test_id_does_not_contain_the_path(self):
        token = sensitive_id("config/prod/.env.production")
        for leaked in ("env", "prod", "config", "production"):
            assert leaked not in token.removeprefix("sensitive:")


class TestPolicyAggregate:
    def test_omissions_are_counted_not_silent(self):
        result = apply_policy([
            "app.py", ".env", "id_rsa",
            "node_modules/a.js", "node_modules/b.js", "dist/out.js",
        ])

        assert result.paths == ["app.py"]
        assert len(result.sensitive_ids) == 2
        assert result.generated_omitted == 3

        # Receipt honesty: a caller must be able to tell "nothing else
        # changed" from "three files changed and were dropped".
        disclosure = result.as_disclosure()
        assert disclosure["generated_omitted"] == 3
        assert disclosure["sensitive_withheld"] == 2
        assert disclosure["matched_rules"], "omission happened with no stated rule"

    def test_repeated_sensitive_path_yields_one_identifier(self):
        result = apply_policy([".env", ".env", ".env"])
        assert len(result.sensitive_ids) == 1
        assert result.sensitive_count == 3


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


@pytest.mark.timeout(120)
def test_interrupted_agent_recovery_does_not_disclose_secret_filenames(
    tmp_path, monkeypatch
):
    """End-to-end reproduction of the dogfooded leak.

    Drives the real MCP surface rather than the classifier, because the defect
    was that these names reached the persisted graph -- filtering only at
    render time would leave them durable on disk.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.test")
    _git(repo, "config", "user.name", "t")
    (repo / "app.py").write_text("def go():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
    from entroly.work_graph_mcp import work_claim, work_resume

    claim = work_claim(
        project=str(repo), agent_id="agent-a", task_title="harden auth",
        task_id="task-1", scope_paths=["app.py"],
    )
    assert claim.get("status") == "ok", claim

    # Interrupted mid-edit, with the untidy working directory of a real repo.
    (repo / "app.py").write_text("def go():\n    return 2  # wip\n", encoding="utf-8")
    (repo / ".env").write_text("STRIPE_SECRET_KEY=sk_live_DEADBEEF\n", encoding="utf-8")
    (repo / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n", encoding="utf-8")
    (repo / "node_modules").mkdir()
    for index in range(5):
        (repo / "node_modules" / f"dep{index}.js").write_text("//x\n", encoding="utf-8")

    recovered = json.dumps(work_resume(project=str(repo), max_evidence=64), default=str)

    for name in (".env", "id_rsa", "node_modules"):
        assert name not in recovered, f"{name} disclosed in recovery output"
    assert "sk_live_DEADBEEF" not in recovered
    assert "BEGIN OPENSSH" not in recovered

    # Premise: the recovery actually ran and still reports the real work. If
    # this trips, the assertions above passed vacuously.
    assert "app.py" in recovered, "recovery reported nothing; leak check was vacuous"
    assert "sensitive:" in recovered, "withheld files left no trace at all"
