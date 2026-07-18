from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from entroly.local_foundation import (
    LocalFoundationConfig,
    LocalFoundationError,
    build_llama_server_command,
    load_config,
    save_config,
    strict_local_environment,
)


def _local_files(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"local-model")
    server = tmp_path / ("llama-server.exe" if os.name == "nt" else "llama-server")
    server.write_text("placeholder", encoding="utf-8")
    return model, server


def test_strict_local_config_round_trip(tmp_path: Path):
    model, server = _local_files(tmp_path)
    config = LocalFoundationConfig.create(
        model_path=model,
        server_executable=server,
        context_size=12_288,
    )
    path = save_config(config, tmp_path / "local_foundation.json", verify_hash=True)

    loaded = load_config(path, verify_hash=True)

    assert loaded.mode == "strict-local"
    assert loaded.base_url == "http://127.0.0.1:9378/v1"
    assert loaded.cloud_fallback is False
    assert loaded.remote_embeddings is False
    assert loaded.remote_reranking is False
    assert loaded.runtime_model_downloads is False


def test_remote_endpoint_is_rejected(tmp_path: Path):
    model, server = _local_files(tmp_path)
    config = LocalFoundationConfig.create(
        model_path=model,
        server_executable=server,
    )
    payload = json.loads(config.to_json())
    payload["base_url"] = "https://api.openai.com/v1"
    remote = LocalFoundationConfig.from_dict(payload)

    with pytest.raises(LocalFoundationError, match="non-loopback"):
        remote.validate()


def test_cloud_fallback_is_rejected(tmp_path: Path):
    model, server = _local_files(tmp_path)
    config = LocalFoundationConfig.create(
        model_path=model,
        server_executable=server,
    )
    payload = json.loads(config.to_json())
    payload["cloud_fallback"] = True
    unsafe = LocalFoundationConfig.from_dict(payload)

    with pytest.raises(LocalFoundationError, match="cloud_fallback must be false"):
        unsafe.validate()


def test_runtime_command_uses_local_model_only(tmp_path: Path):
    model, server = _local_files(tmp_path)
    config = LocalFoundationConfig.create(
        model_path=model,
        server_executable=server,
    )

    command = build_llama_server_command(config)

    assert "-m" in command
    assert str(model.resolve()) in command
    assert "-hf" not in command
    assert not any(value.startswith("https://") for value in command)
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_strict_local_environment_removes_cloud_credentials(tmp_path: Path):
    model, server = _local_files(tmp_path)
    config = LocalFoundationConfig.create(
        model_path=model,
        server_executable=server,
    )
    base = {
        "PATH": "test",
        "OPENAI_API_KEY": "secret",
        "ANTHROPIC_API_KEY": "secret",
        "OPENROUTER_API_KEY": "secret",
        "ENTROLY_ANTHROPIC_BASE": "https://api.anthropic.com",
    }

    env = strict_local_environment(config, base)

    assert env["PATH"] == "test"
    assert env["ENTROLY_LOCAL_ONLY"] == "1"
    assert env["ENTROLY_OPENAI_BASE"] == "http://127.0.0.1:9378"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENROUTER_API_KEY" not in env
    assert "ENTROLY_ANTHROPIC_BASE" not in env
