from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_publish_job_has_timeout() -> None:
    workflow = (ROOT / ".github/workflows/entroly-publish.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(?ms)^  build-and-push:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
    )

    assert match is not None
    assert re.search(r"(?m)^    timeout-minutes:\s*[1-9][0-9]*\s*$", match.group("body"))


def test_docker_quality_gate_exposes_installed_console_scripts() -> None:
    workflow = (ROOT / ".github/workflows/entroly-publish.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"(?ms)^  quality-gate:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
    )

    assert match is not None
    body = match.group("body")
    path_export = 'echo "$PWD/.venv/bin" >> "$GITHUB_PATH"'
    assert path_export in body
    assert body.index(path_export) < body.index(".venv/bin/pytest tests/")
