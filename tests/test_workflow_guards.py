from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_publish_job_has_timeout() -> None:
    workflow = (ROOT / ".github/workflows/docker-publish.yml").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^  build-and-push:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)",
        workflow,
    )

    assert match is not None
    assert re.search(r"(?m)^    timeout-minutes:\s*[1-9][0-9]*\s*$", match.group("body"))
