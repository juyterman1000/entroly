from __future__ import annotations

import pytest

from entroly.codecs_builtin import ShellCodec
from entroly.shell_codec import detect_shell_profile, esc_compress


@pytest.mark.parametrize(
    ("tool", "text", "profile", "required"),
    [
        ("pytest", "$ pytest -q\ncollecting ...\nFAILED tests/test_cart.py::test_total\nAssertionError: 3 != 4\n1 failed, 9 passed in 0.4s\n", "pytest", "1 failed, 9 passed"),
        ("cargo test", "$ cargo test\nCompiling cart v0.1.0\nerror[E0308]: mismatched types\ntest result: FAILED. 2 passed; 1 failed\n", "cargo", "error[E0308]"),
        ("npm test", "$ npm test\nPASS a.test.ts\nnpm ERR! code ELIFECYCLE\nTest Suites: 1 failed, 2 passed\n", "node", "Test Suites:"),
        ("git merge", "$ git merge feature\nAuto-merging app.py\nCONFLICT (content): Merge conflict in app.py\nAutomatic merge failed\n", "git", "CONFLICT (content)"),
        ("docker build", "$ docker build .\n[1/9] FROM python\nfailed to solve: process exited with code 1\n", "container", "failed to solve"),
        ("kubectl", "$ kubectl get pods\nNAME READY STATUS\napi-1 0/1 CrashLoopBackOff\n", "kubernetes", "CrashLoopBackOff"),
        ("terraform", "$ terraform plan\nTerraform will perform the following actions:\nPlan: 4 to add, 1 to change, 0 to destroy.\n", "terraform", "Plan: 4 to add"),
        ("gradle", "$ gradle build\n> Task :compile\nBUILD FAILED in 4s\n", "build", "BUILD FAILED"),
    ],
)
def test_profiles_preserve_outcome_evidence(
    tool: str, text: str, profile: str, required: str
) -> None:
    result = esc_compress(text + ("progress noise\n" * 80), budget=12, tool_name=tool)
    assert result.profile == profile
    assert required in result.compressed
    assert text.splitlines()[0] in result.compressed


def test_unknown_command_remains_generic() -> None:
    assert detect_shell_profile("alpha\nbeta\ngamma", "custom-tool") is None
    assert esc_compress("alpha\nbeta\ngamma", tool_name="custom-tool").profile == "generic"


def test_shell_codec_recovery_remains_exact_with_profile() -> None:
    source = "$ pytest -q\n" + ("collecting plugin data\n" * 60) + "1 failed, 9 passed\n"
    codec = ShellCodec()
    reps = codec.representations(source, source_id="pytest", budget=30)
    assert len(reps) == 2
    assert reps[1].recovery is not None
    assert codec.store.get(reps[1].recovery.digest) == source
