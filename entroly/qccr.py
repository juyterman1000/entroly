"""Compatibility wrapper around the Rust-backed QCCR binding.

The implementation remains the pre-hotfix module, while the public call wrapper
synchronizes mutable hooks before every invocation. This preserves integrations
and tests that monkeypatch ``entroly.qccr._rust_select`` or related bindings.
"""

from __future__ import annotations

from . import qccr_legacy as _legacy
from .qccr_hotfix import attach_sufficiency as _safe_attach_sufficiency
from .sufficiency import _lexical_terms as _lexical_term_set

for _name, _value in vars(_legacy).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_MUTABLE_HOOKS = (
    "_rust_expand_query",
    "_rust_rank_files",
    "_rust_select",
    "_load_rank_weights",
    "_PREFILTER_FILE_FLOOR",
    "logical_source",
)


def _sync_public_hooks() -> None:
    for name in _MUTABLE_HOOKS:
        if name in globals():
            setattr(_legacy, name, globals()[name])
    _legacy._attach_sufficiency = _safe_attach_sufficiency


def select(fragments: list[dict], token_budget: int, query: str = "") -> list[dict]:
    _sync_public_hooks()
    return _legacy.select(fragments, token_budget, query)


def _expanded_query_tokens(query: str) -> frozenset[str]:
    _sync_public_hooks()
    return frozenset(_legacy._rust_expand_query(query))


_attach_sufficiency = _safe_attach_sufficiency
