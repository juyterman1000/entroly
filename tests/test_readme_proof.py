from __future__ import annotations

import pytest

from scripts import readme_proof


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("model-recovery", "24/24"),
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
    assert "not a universal" in output.lower()


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
