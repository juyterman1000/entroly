from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "1.0.66"
HOMEBREW_FORMULA_VERSION = "1.0.64"
HOMEBREW_FORMULA_SHA256 = "c259fe1e25311679f54ef356f14047f3f6c1e1a6943e82c27bc00966fcea1a3f"
CANONICAL_MCP_NAME = "io.github.juyterman1000/entroly"
CANONICAL_REPOSITORY = "https://github.com/juyterman1000/entroly"


def _read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _read_project_metadata(path: str) -> dict[str, object]:
    """Read the small pyproject surface guarded by these tests.

    Python 3.10 does not ship ``tomllib``. Keeping this parser local avoids
    making the release guard depend on an extra test dependency.
    """
    metadata: dict[str, object] = {
        "optional-dependencies": {},
    }
    current_section = ""
    current_list_key: str | None = None

    for raw_line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]")
            current_list_key = None
            continue
        if current_section == "project" and line.startswith("version"):
            metadata["version"] = line.split("=", 1)[1].strip().strip('"')
            continue
        if current_section == "project" and line.startswith("readme"):
            metadata["readme"] = line.split("=", 1)[1].strip().strip('"')
            continue
        if current_section == "project" and line.startswith("dependencies"):
            current_list_key = "dependencies"
            metadata[current_list_key] = []
            continue
        if (
            current_section == "project.optional-dependencies"
            and "=" in line
            and not line.startswith('"')
        ):
            key = line.split("=", 1)[0].strip()
            current_list_key = key
            metadata["optional-dependencies"][key] = []
            continue
        if current_list_key and line.startswith('"'):
            value = line.rstrip(",").strip().strip('"')
            if current_section == "project":
                metadata[current_list_key].append(value)
            elif current_section == "project.optional-dependencies":
                metadata["optional-dependencies"][current_list_key].append(value)

    return metadata


def test_public_package_versions_are_1_0_66() -> None:
    assert _read_project_metadata("pyproject.toml")["version"] == RELEASE_VERSION
    assert _read_project_metadata("entroly/pyproject.toml")["version"] == RELEASE_VERSION
    assert _read_json("entroly/npm/package.json")["version"] == RELEASE_VERSION
    assert _read_json("entroly/npm-alias/package.json")["version"] == RELEASE_VERSION
    assert _read_json("entroly-wasm/package.json")["version"] == RELEASE_VERSION
    assert _read_json("integrations/openclaw/package.json")["version"] == RELEASE_VERSION
    assert _read_json(".claude-plugin/manifest.json")["version"] == RELEASE_VERSION
    assert _read_json(".mcpb-build/manifest.json")["version"] == RELEASE_VERSION


def test_bundled_mcpb_manifest_matches_release_source() -> None:
    source = _read_json(".mcpb-build/manifest.json")
    with zipfile.ZipFile(ROOT / "entroly.mcpb") as bundle:
        bundled = json.loads(bundle.read("manifest.json"))

    assert bundled == source


def test_openclaw_install_metadata_identifies_clawhub_target() -> None:
    package = _read_json("integrations/openclaw/package.json")
    manifest = _read_json("integrations/openclaw/openclaw.plugin.json")
    openclaw = package["openclaw"]
    install = openclaw["install"]

    assert openclaw["release"]["publishToClawHub"] is True
    assert install["clawhubSpec"] == f"clawhub:{package['name']}"
    assert install["npmSpec"] == package["name"]
    assert install["defaultChoice"] == "npm"
    assert install["minHostVersion"] == openclaw["compat"]["pluginApi"]
    discovery = manifest["configSchema"]["properties"]["autoDiscoverContextBudget"]
    assert discovery == {"type": "boolean", "default": True}
    assert "No remote discovery is enabled automatically" in manifest["uiHints"][
        "autoDiscoverContextBudget"
    ]["help"]


def test_mcp_registry_manifest_points_at_release_package() -> None:
    manifest = _read_json("server.json")

    assert manifest["version"] == RELEASE_VERSION
    packages = manifest["packages"]
    assert packages
    assert packages[0]["identifier"] == "entroly"
    assert packages[0]["version"] == RELEASE_VERSION


def test_mcp_registry_identity_is_canonical_and_non_squattable() -> None:
    manifest = _read_json("server.json")
    npm_manifest = _read_json("entroly/npm/package.json")

    assert manifest["name"] == CANONICAL_MCP_NAME
    assert manifest["websiteUrl"] == CANONICAL_REPOSITORY
    assert manifest["repository"] == {
        "url": CANONICAL_REPOSITORY,
        "source": "github",
    }
    assert npm_manifest["mcpName"] == CANONICAL_MCP_NAME
    assert _read_project_metadata("pyproject.toml")["readme"] == "PYPI_README.md"
    assert f"mcp-name: {CANONICAL_MCP_NAME}" in (
        ROOT / "PYPI_README.md"
    ).read_text(encoding="utf-8")

    expected = {
        ("pypi", "entroly", "uvx", ()),
        ("npm", "entroly-mcp", "npx", ()),
    }
    actual = {
        (
            package["registryType"],
            package["identifier"],
            package["runtimeHint"],
            tuple(argument["value"] for argument in package.get("packageArguments", [])),
        )
        for package in manifest["packages"]
    }
    assert actual == expected


def test_native_engine_is_optional_for_first_time_install() -> None:
    for path in ("pyproject.toml", "entroly/pyproject.toml"):
        project = _read_project_metadata(path)
        hard_deps = project["dependencies"]
        native_deps = project["optional-dependencies"]["native"]
        full_deps = project["optional-dependencies"]["full"]

        assert not any(dep.startswith("entroly-core") for dep in hard_deps)
        assert f"entroly-core>={RELEASE_VERSION},<2" in native_deps
        assert f"entroly-core>={RELEASE_VERSION},<2" in full_deps


def test_no_stale_package_advertising_versions() -> None:
    stale = []
    for path in (
        "server.json",
        ".mcpb-build/manifest.json",
        ".claude-plugin/manifest.json",
        "entroly/npm/package.json",
        "entroly/npm-alias/package.json",
        "entroly-wasm/package.json",
        "integrations/openclaw/package.json",
        "pyproject.toml",
        "entroly/pyproject.toml",
    ):
        text = (ROOT / path).read_text(encoding="utf-8")
        if re.search(r'"version"\s*:\s*"1\.0\.41"', text):
            stale.append(path)

    assert stale == []


def test_homebrew_formula_targets_release_sdist() -> None:
    text = (ROOT / "packaging/homebrew/entroly.rb").read_text(encoding="utf-8")

    assert f"entroly-{HOMEBREW_FORMULA_VERSION}.tar.gz" in text
    assert "packages/source/e/entroly/" in text
    assert HOMEBREW_FORMULA_SHA256 in text


def test_release_workflow_sanitizes_version_once_and_probes_live_artifacts() -> None:
    text = (ROOT / ".github/workflows/entroly-publish.yml").read_text(
        encoding="utf-8"
    )

    assert text.count("github.event.inputs.release_version") == 1
    assert "SEMVER_RE=" in text
    assert "needs.release-metadata.outputs.version" in text
    assert "probe-pypi-openclaw-bridge:" in text
    assert "probe-npm-openclaw:" in text
    assert "needs: [release-metadata, probe-npm-openclaw]" in text
    assert '"openclaw@2026.6.11" "entroly-openclaw@${RELEASE_VERSION}"' in text
    openclaw_publisher = text.split("  publish-npm-openclaw:", 1)[1].split(
        "  probe-npm-openclaw:", 1
    )[0]
    assert "for attempt in $(seq 1 20)" in openclaw_publisher
    assert "waiting for PyPI propagation" in openclaw_publisher

    clawhub_publisher = text.split("  publish-clawhub-openclaw:", 1)[1].split(
        "  publish-binaries:", 1
    )[0]
    assert "id-token: write" in clawhub_publisher
    assert "package trusted-publisher set entroly-openclaw" in clawhub_publisher
    assert '--workflow-filename "entroly-publish.yml"' in clawhub_publisher
    # ClawHub REQUIRES --manual-override-reason for a package that has a trusted
    # publisher configured: `clawhub package publish` on such a package is
    # treated as a "manual" publish and rejected without it. Observed live in
    # run 29794820445: "Manual publishes for packages with trusted publisher
    # config require manualOverrideReason". Assert the flag is present with a
    # substantive, auditable justification — do NOT remove it, or the release's
    # ClawHub publish fails and the listing stalls on the prior version.
    override_reason = re.search(
        r'--manual-override-reason "([^"]+)"', clawhub_publisher
    )
    assert override_reason is not None, (
        "trusted-publisher ClawHub publish must pass --manual-override-reason"
    )
    assert len(override_reason.group(1)) >= 20, (
        "the override reason must be an auditable justification, not a stub"
    )
    assert '--source-commit "$GITHUB_SHA"' in clawhub_publisher
    assert '--source-ref "entroly-v${RELEASE_VERSION}"' in clawhub_publisher
    assert "Verify exact ClawHub version is public" in clawhub_publisher
    assert "for attempt in range(1, 61)" in clawhub_publisher
    assert "package moderation-status entroly-openclaw --json" in clawhub_publisher
    assert "moderation-status entroly-openclaw --json > \"$raw_status\"" in clawhub_publisher
    assert 'release.get("moderationReason")' not in clawhub_publisher


def test_homebrew_sync_is_single_pinned_release_workflow() -> None:
    assert not (ROOT / ".github/workflows/sync-homebrew-after-release.yml").exists()
    text = (ROOT / ".github/workflows/sync-homebrew-formula.yml").read_text(
        encoding="utf-8"
    )

    assert "github.event.workflow_run.head_sha || github.sha" in text
    assert "git', 'show', f'{release_sha}:pyproject.toml'" in text
    assert "merge-base', '--is-ancestor', release_sha, 'origin/main'" in text
    assert "group: sync-homebrew-main" in text
    assert "if target_tuple < current_tuple:" in text
    assert "refusing to downgrade Homebrew" in text
    assert "pull-requests: write" in text
    assert 'BRANCH="agent/homebrew-${VERSION}"' in text
    assert "gh pr create --base main" in text
    assert "git push origin HEAD:main" not in text


def test_release_artifacts_have_one_quality_gated_publisher() -> None:
    assert not (ROOT / ".github/workflows/docker-publish.yml").exists()

    coordinated = (ROOT / ".github/workflows/entroly-publish.yml").read_text(
        encoding="utf-8"
    )
    assert "platforms: linux/amd64,linux/arm64" in coordinated
    assert (
        "type=raw,value=${{ needs.release-metadata.outputs.version }},"
        "enable=${{ needs.release-metadata.outputs.should_publish == 'true' }}"
        in coordinated
    )
    assert "needs: [release-metadata, quality-gate]" in coordinated
    assert "needs: [release-metadata, quality-gate, release-anchor]" in coordinated
    assert "needs: [release-metadata, quality-gate, github-release]" in coordinated
    assert "tag: ${{ needs.release-metadata.outputs.tag }}" in coordinated
    assert "BEFORE_SHA: ${{ github.event.before }}" in coordinated
    assert 'git show "${BEFORE_SHA}:pyproject.toml"' in coordinated
    assert "refusing non-increasing release transition" in coordinated
    assert "Previous main commit is unavailable; refusing automatic publication." in coordinated
    assert "group: entroly-production-publication" in coordinated
    assert "Create or verify the canonical release tag" in coordinated
    assert "GitHub Actions coordinated Entroly" in coordinated
    anchor = coordinated.split("  release-anchor:", 1)[1].split(
        "  publish-core:", 1
    )[0]
    assert anchor.index("git merge-base --is-ancestor") < anchor.index(
        'if [[ "$SHOULD_PUBLISH" != "true" ]]'
    )

    for workflow in ("publish-core-wheels.yml", "release-binary.yml"):
        definition = (ROOT / ".github/workflows" / workflow).read_text(
            encoding="utf-8"
        )
        triggers = definition.split("jobs:", 1)[0]
        assert "workflow_call:" in triggers
        assert "\n  push:" not in triggers
        assert "\n  workflow_dispatch:" not in triggers


def test_commit_identity_guard_scopes_dependabot_to_dependency_paths() -> None:
    guard = (ROOT / ".github/workflows/commit-identity-guard.yml").read_text(
        encoding="utf-8"
    )

    assert "trusted_dependabot_author" in guard
    assert '== "dependabot[bot]"' in guard
    assert 'str(head.get("ref", "")).startswith("dependabot/")' in guard
    assert 'head_repository.get("full_name")' in guard
    assert 'os.environ["GITHUB_REPOSITORY"]' in guard
    assert "all(dependency_path(path) for path in paths)" in guard
    assert 'path.startswith(".github/workflows/")' in guard
    assert 'event.get("sender", {}).get("login") != expected_login' in guard


def test_commit_identity_guard_accepts_only_path_scoped_dependabot(
    tmp_path: Path,
) -> None:
    guard = (ROOT / ".github/workflows/commit-identity-guard.yml").read_text(
        encoding="utf-8"
    )
    embedded = guard.split("          python - <<'PY'\n", 1)[1].split(
        "\n          PY", 1
    )[0]
    script = textwrap.dedent(embedded)
    compile(script, "commit-identity-guard", "exec")

    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args: str, env: dict[str, str] | None = None) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    owner_env = os.environ.copy()
    owner_env.update(
        {
            "GIT_AUTHOR_NAME": "juyterman1000",
            "GIT_AUTHOR_EMAIL": "208309368+juyterman1000@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "juyterman1000",
            "GIT_COMMITTER_EMAIL": (
                "208309368+juyterman1000@users.noreply.github.com"
            ),
        }
    )
    bot_env = os.environ.copy()
    bot_env.update(
        {
            "GIT_AUTHOR_NAME": "dependabot[bot]",
            "GIT_AUTHOR_EMAIL": (
                "49699333+dependabot[bot]@users.noreply.github.com"
            ),
            "GIT_COMMITTER_NAME": "GitHub",
            "GIT_COMMITTER_EMAIL": "noreply@github.com",
        }
    )

    git("init", "-b", "main")
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "entroly"\nversion = "1.0.66"\n',
        encoding="utf-8",
    )
    git("add", "pyproject.toml")
    git("commit", "-m", "initial", env=owner_env)
    base = git("rev-parse", "HEAD")

    (repository / "pyproject.toml").write_text(
        '[project]\nname = "entroly"\nversion = "1.0.66"\n'
        '[project.optional-dependencies]\nbenchmark = ["tiktoken>=0.9,<0.14"]\n',
        encoding="utf-8",
    )
    git("add", "pyproject.toml")
    git(
        "commit",
        "-m",
        "build(deps-dev): update tiktoken requirement",
        env=bot_env,
    )
    event_path = tmp_path / "event.json"

    def run_guard(base_sha: str, head_sha: str) -> subprocess.CompletedProcess[str]:
        event_path.write_text(
            json.dumps(
                {
                    "pull_request": {
                        "base": {"sha": base_sha},
                        "head": {
                            "sha": head_sha,
                            "ref": "dependabot/pip/tiktoken",
                            "repo": {"full_name": "juyterman1000/entroly"},
                        },
                        "user": {"login": "dependabot[bot]"},
                    }
                }
            ),
            encoding="utf-8",
        )
        guard_env = os.environ.copy()
        guard_env.update(
            {
                "EXPECTED_LOGIN": "juyterman1000",
                "EXPECTED_NOREPLY_EMAIL": (
                    "208309368+juyterman1000@users.noreply.github.com"
                ),
                "EXPECTED_ACCOUNT_EMAIL": "fastrunner10090@gmail.com",
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_EVENT_PATH": str(event_path),
                "GITHUB_REPOSITORY": "juyterman1000/entroly",
            }
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            env=guard_env,
            check=False,
            capture_output=True,
            text=True,
        )

    (repository / "README.md").write_text(
        "# Entroly\n",
        encoding="utf-8",
    )
    git("add", "README.md")
    git("commit", "-m", "test: revalidate dependency update", env=owner_env)
    revalidated_head = git("rev-parse", "HEAD")

    trusted = run_guard(base, revalidated_head)
    assert trusted.returncode == 0, trusted.stdout + trusted.stderr

    (repository / "entroly").mkdir()
    (repository / "entroly" / "server.py").write_text(
        "UNRELATED_CODE = True\n",
        encoding="utf-8",
    )
    git("add", "entroly/server.py")
    git("commit", "-m", "build(deps): change runtime code", env=bot_env)
    untrusted_head = git("rev-parse", "HEAD")

    untrusted = run_guard(revalidated_head, untrusted_head)
    assert untrusted.returncode != 0
    assert "Commit identity guard failed." in untrusted.stdout


def test_benchmark_verifies_frozen_evidence_before_latest_tokenizer_smoke() -> None:
    benchmark = (ROOT / ".github/workflows/benchmark.yml").read_text(
        encoding="utf-8"
    )

    assert 'EVIDENCE_TIKTOKEN_VERSION: "0.9.0"' in benchmark
    assert '"tiktoken==${EVIDENCE_TIKTOKEN_VERSION}"' in benchmark
    assert 'python -m pip install --upgrade "tiktoken>=0.9,<0.14"' in benchmark
    assert benchmark.index("Verify compression evidence artifact") < benchmark.index(
        "Smoke-test the supported tokenizer ceiling"
    )


def test_release_version_dispatch_input_is_validated_via_environment() -> None:
    text = (ROOT / ".github/workflows/sync-release-version.yml").read_text(
        encoding="utf-8"
    )

    assert "DISPATCH_VERSION: ${{ github.event.inputs.version }}" in text
    assert 'version="$DISPATCH_VERSION"' in text
    assert '${{ inputs.version }}' not in text


def test_mcp_registry_follows_every_anchored_parent_release() -> None:
    text = (ROOT / ".github/workflows/publish-mcp-registry.yml").read_text(
        encoding="utf-8"
    )

    assert "github.event.workflow_run.head_branch == 'main'" not in text
    assert 'EXPECTED_TAG="entroly-v${VERSION}"' in text
    assert 'TAG_SHA="$(git rev-parse "${EXPECTED_TAG}^{commit}")"' in text
    assert 'if [[ "$TAG_SHA" != "$SOURCE_SHA" ]]' in text
    assert 'echo "should_publish=false" >> "$GITHUB_OUTPUT"' in text
    assert "if: steps.release_guard.outputs.should_publish == 'true'" in text


def test_clawhub_verifier_follows_coordinated_release() -> None:
    text = (ROOT / ".github/workflows/verify-clawhub-listing.yml").read_text(
        encoding="utf-8"
    )
    triggers = text.split("permissions:", 1)[0]

    assert 'workflows: ["Build and Push Entroly Docker Image"]' in triggers
    assert "workflow_run:" in triggers
    assert "\n  push:" not in triggers
    assert "github.event.workflow_run.conclusion == 'success'" in text
    assert "github.event.workflow_run.head_sha || github.sha" in text
    assert "ref: ${{ env.SOURCE_SHA }}" in text
    assert "sha: process.env.SOURCE_SHA" in text
    assert "const description = process.env.CLAWHUB_DESCRIPTION" in text
    assert "const description = '${{ steps.verify.outputs.description }}'" not in text
    assert (
        "https://clawhub.ai/juyterman1000/plugins/entroly-openclaw" in text
    )
    assert "ClawHub {expected_version} unavailable: {safe_error}" in text


def test_clawhub_reconciliation_is_tag_bound_and_non_destructive_by_default() -> None:
    text = (ROOT / ".github/workflows/publish-openclaw-clawhub.yml").read_text(
        encoding="utf-8"
    )

    assert "publish_if_missing:" in text
    assert "default: false" in text
    assert 'release_tag="entroly-v${REQUESTED_VERSION}"' in text
    assert 'git rev-parse "${release_tag}^{commit}"' in text
    assert "package moderation-status entroly-openclaw --json" in text
    assert "steps.moderation.outputs.decision == 'missing'" in text
    assert "inputs.publish_if_missing == true" in text
    assert "package trusted-publisher set entroly-openclaw" in text
    assert '--source-commit "$RELEASE_SHA"' in text
    assert "package delete" not in text
    assert "clawhub-moderation-before-raw.json" in text
    assert "clawhub-moderation-before-raw.json" not in text.split(
        "name: clawhub-release-reconciliation-${{ inputs.version }}", 1
    )[1]
    assert 'release.get("moderationReason")' not in text
