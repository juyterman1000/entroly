from __future__ import annotations

import pytest

from scripts import readme_proof


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param(
            "model-recovery",
            "24/24",
            marks=pytest.mark.skipif(
                not readme_proof.model_recovery_tokenizer_available(),
                reason="artifact verification requires its frozen o200k tokenizer",
            ),
        ),
        ("restart-recovery", "66/66"),
    ],
)
def test_artifact_proof_commands_verify_before_display(
    command: str,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert readme_proof.main([command]) == 0
    output = capsys.readouterr().out
    assert "[PASS]" in output
    assert expected in output
    assert "HEADROOM BASELINE" in output
    assert "not a universal" in output.lower()


def test_restart_proof_keeps_failure_mechanics_in_raw_artifact(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert readme_proof.main(["restart-recovery"]) == 0
    output = capsys.readouterr().out
    assert "incomplete worker evidence" in output
    assert "database is locked" not in output
    assert "Worker errors" not in output


def test_model_recovery_proof_explains_missing_optional_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        readme_proof,
        "model_recovery_tokenizer_available",
        lambda: False,
    )

    assert readme_proof.main(["model-recovery"]) == 2
    assert "python -m pip install tiktoken" in capsys.readouterr().err


def test_local_proof_uses_packaged_verifier(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(*, output: str, max_files: int) -> int:
        assert max_files == 40
        report = {
            "passed": 6,
            "failed": 0,
            "results": [
                {"id": claim_id, "status": "PASS"}
                for claim_id in (
                    "SDK-1",
                    "IDX-1",
                    "OPT-1",
                    "CCR-2",
                    "CCR-3",
                    "LOCAL-1",
                )
            ],
        }
        readme_proof.Path(output).write_text(
            readme_proof.json.dumps(report),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr("entroly.verify_claims.run", fake_run)

    assert readme_proof.main(["local"]) == 0
    output = capsys.readouterr().out
    assert "6/6 checks passed" in output
    assert "Trust the result, not this video" in output
