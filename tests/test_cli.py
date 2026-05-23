"""Unit tests for entroly.cli."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from entroly import cli


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_detect_project_type_unknown(chdir_tmp):
    result = cli._detect_project_type()
    assert result["primary"] == "unknown"
    assert result["languages"] == ["unknown"]
    assert result["frameworks"] == []
    assert result["name"] == chdir_tmp.name


def test_detect_project_type_python(chdir_tmp):
    (chdir_tmp / "pyproject.toml").write_text("[project]\nname='x'\n")
    result = cli._detect_project_type()
    assert "python" in result["languages"]
    assert result["primary"] == "python"


def test_detect_project_type_rust_and_go(chdir_tmp):
    (chdir_tmp / "Cargo.toml").write_text("[package]\n")
    (chdir_tmp / "go.mod").write_text("module x\n")
    result = cli._detect_project_type()
    assert "rust" in result["languages"]
    assert "go" in result["languages"]


def test_detect_project_type_react_via_package_json(chdir_tmp):
    (chdir_tmp / "package.json").write_text(json.dumps({
        "dependencies": {"react": "^18.0.0", "express": "^4.0.0"},
    }))
    result = cli._detect_project_type()
    assert "javascript" in result["languages"]
    assert "React" in result["frameworks"]
    assert "Express" in result["frameworks"]


def test_detect_project_type_nextjs_suppresses_react(chdir_tmp):
    (chdir_tmp / "next.config.js").write_text("module.exports={}")
    (chdir_tmp / "package.json").write_text(json.dumps({
        "dependencies": {"react": "^18.0.0"},
    }))
    result = cli._detect_project_type()
    assert "Next.js" in result["frameworks"]
    assert "React" not in result["frameworks"]


def test_detect_project_type_handles_malformed_package_json(chdir_tmp):
    (chdir_tmp / "package.json").write_text("{not valid json")
    result = cli._detect_project_type()
    assert "javascript" in result["languages"]
    assert result["frameworks"] == []
