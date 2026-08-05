"""Operator-supplied recovery-store compatibility contract."""

from __future__ import annotations

from typing import Any

from external_adapter import resolve_symbol


class CompressionStore:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_STORE")
        return implementation(*args, **kwargs)


def set_request_compression_store(store: Any) -> Any:
    implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_SET_REQUEST_STORE")
    return implementation(store)


def clear_request_compression_store() -> Any:
    implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_CLEAR_REQUEST_STORE")
    return implementation()


__all__ = [
    "CompressionStore",
    "set_request_compression_store",
    "clear_request_compression_store",
]
