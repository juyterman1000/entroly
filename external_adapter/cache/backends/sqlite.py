"""Operator-supplied persistent-backend class."""

from external_adapter import resolve_symbol


class SQLiteBackend:
    def __new__(cls, *args, **kwargs):
        implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_BACKEND")
        return implementation(*args, **kwargs)
