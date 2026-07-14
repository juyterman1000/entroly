from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "1.0.58"
HOMEBREW_FORMULA_VERSION = "1.0.58"
HOMEBREW_FORMULA_SHA256 = "f2d561e7316cf12c07ffffadeff0b8f26368538564d776eee86b4b76a896959e"
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


def test_public_package_versions_are_1_0_58() -> None:
    assert _read_project_metadata("pyproject.toml")["version"] == RELEASE_VERSION
    assert _read_project_metadata("entroly/pyproject.toml")["version"] == RELEASE_VERSION
    assert _read_json("entroly/npm/package.json")["version"] == RELEASE_VERSION
    assert _read_json("entroly/npm-alias/package.json")["version"] == RELEASE_VERSION
    assert _read_json("entroly-wasm/package.json")["version"] == RELEASE_VERSION
    assert _read_json("integrations/openclaw/package.json")["version"] == RELEASE_VERSION
    assert _read_json(".claude-plugin/manifest.json")["version"] == RELEASE_VERSION
    assert _read_json(".mcpb-build/manifest.json")["version"] == RELEASE_VERSION


def test_openclaw_install_metadata_identifies_clawhub_target() -> None:
    package = _read_json("integrations/openclaw/package.json")
    openclaw = package["openclaw"]
    install = openclaw["install"]

    assert openclaw["release"]["publishToClawHub"] is True
    assert install["clawhubSpec"] == f"clawhub:{package['name']}"
    assert install["npmSpec"] == package["name"]
    assert install["defaultChoice"] == "npm"
    assert install["minHostVersion"] == openclaw["compat"]["pluginApi"]


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
    assert '"Release v${PACKAGE_VERSION}"' in coordinated
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
