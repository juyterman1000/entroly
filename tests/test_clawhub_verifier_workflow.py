from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/verify-clawhub-listing.yml"


def test_clawhub_verification_is_release_commit_scoped() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Determine whether public verification is required" in text
    assert 'release_tag="entroly-v${expected_version}"' in text
    assert 'git rev-parse "${release_tag}^{commit}"' in text
    assert '"$tag_sha" == "$SOURCE_SHA"' in text
    assert "steps.release-context.outputs.verify == 'true'" in text


def test_non_release_pushes_receive_a_successful_required_status() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'echo "verify=false" >> "$GITHUB_OUTPUT"' in text
    assert 'echo "state=success" >> "$GITHUB_OUTPUT"' in text
    assert "ClawHub check not required" in text
    assert (
        "steps.verify.outputs.state || steps.release-context.outputs.state"
        in text
    )
    assert "if (state !== 'success')" in text


def test_manual_dispatch_remains_a_strict_public_diagnostic() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'if [[ "$EVENT_NAME" == "workflow_dispatch" ]]' in text
    assert 'echo "verify=true" >> "$GITHUB_OUTPUT"' in text
