from __future__ import annotations

import json

import pytest

from entroly.assurance_mcp import _error, _safe_workspace


def test_safe_workspace_requires_existing_directory(tmp_path) -> None:
    assert _safe_workspace(str(tmp_path)) == tmp_path.resolve()
    with pytest.raises(ValueError):
        _safe_workspace(str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        _safe_workspace("\x00")


def test_error_payload_is_bounded_json() -> None:
    payload = json.loads(_error("op", RuntimeError("x" * 5_000)))
    assert payload["status"] == "error"
    assert payload["operation"] == "op"
    assert len(payload["error"]) <= 600


def test_safe_workspace_file_blocks_escape(tmp_path) -> None:
    from entroly.assurance_mcp import _safe_workspace_file

    inside = tmp_path / "inside.txt"
    inside.write_text("ok", encoding="utf-8")
    candidate, relative = _safe_workspace_file(str(tmp_path), "inside.txt")
    assert candidate == inside.resolve()
    assert relative.as_posix() == "inside.txt"

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes workspace"):
        _safe_workspace_file(str(tmp_path), str(outside))
