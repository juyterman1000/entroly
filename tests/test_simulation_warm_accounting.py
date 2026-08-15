from __future__ import annotations

from types import SimpleNamespace

from entroly import auto_index
from entroly import cli
from entroly import server


def _install_engine(monkeypatch, engine) -> None:
    monkeypatch.setattr(server, "EntrolyEngine", lambda: engine)
    monkeypatch.setattr(
        auto_index,
        "auto_index",
        lambda _engine: {
            "status": "skipped",
            "existing_fragments": 120,
            "total_tokens": 226_581,
        },
    )


def test_rust_warm_start_uses_persisted_cumulative_token_total(monkeypatch) -> None:
    rust = SimpleNamespace(
        stats=lambda: {"session": {"total_tokens_tracked": 718_430}}
    )
    engine = SimpleNamespace(_use_rust=True, _rust=rust)
    _install_engine(monkeypatch, engine)

    loaded, files, tokens, status = cli._load_local_simulation_engine(max_files=120)

    assert loaded is engine
    assert (files, tokens, status) == (120, 718_430, "skipped")


def test_python_warm_start_uses_context_budget_token_total(monkeypatch) -> None:
    engine = SimpleNamespace(_use_rust=False, _total_token_count=999_999)
    _install_engine(monkeypatch, engine)

    loaded, files, tokens, status = cli._load_local_simulation_engine(max_files=120)

    assert loaded is engine
    assert (files, tokens, status) == (120, 226_581, "skipped")
