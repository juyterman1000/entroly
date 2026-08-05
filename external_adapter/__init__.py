"""Operator-supplied adapter contract for neutral external baselines.

The Entroly repository does not bundle, identify, install, or endorse another
context product. Benchmark operators provide import targets explicitly through
environment variables. Importing an external adapter without that configuration
fails closed.
"""

from __future__ import annotations

import importlib
import os
from typing import Any


def resolve_symbol(variable: str) -> Any:
    spec = os.environ.get(variable, "").strip()
    if not spec or ":" not in spec:
        raise RuntimeError(
            f"{variable} must be set to an operator-controlled 'module:symbol' import target"
        )
    module_name, symbol_name = spec.split(":", 1)
    if not module_name or not symbol_name:
        raise RuntimeError(f"invalid external adapter import target in {variable}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as error:
        raise RuntimeError(
            f"external adapter symbol {symbol_name!r} is unavailable in {module_name!r}"
        ) from error


def compress(*args: Any, **kwargs: Any) -> Any:
    implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_COMPRESS")
    return implementation(*args, **kwargs)


__all__ = ["compress", "resolve_symbol"]
