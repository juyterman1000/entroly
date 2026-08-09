"""Process-start authority for destructive repository-intelligence operations.

Read, analysis, planning, and preview capabilities stay available by default.
Destructive service-level apply operations require an operator-controlled process
environment set before the service instance is constructed. Tool arguments can
acknowledge known incompleteness, but they cannot grant write authority.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

REPOSITORY_WRITE_AUTHORITY_SCHEMA_VERSION = "entroly.repository-write-authority.v1"
REPOSITORY_WRITES_ENV = "ENTROLY_REPOSITORY_WRITES"
_TRUE = frozenset({"1", "true", "yes", "on", "enabled"})


class RepositoryWriteAuthorizationError(PermissionError):
    """Raised when a destructive operation lacks external operator authority."""


@dataclass(frozen=True)
class RepositoryWriteAuthority:
    """Immutable authority snapshot captured when a service starts."""

    enabled: bool
    source: str = "process-start-environment"

    @classmethod
    def from_environment(cls) -> "RepositoryWriteAuthority":
        raw = os.environ.get(REPOSITORY_WRITES_ENV, "").strip().lower()
        return cls(enabled=raw in _TRUE)

    def require(self, operation: str) -> None:
        if self.enabled:
            return
        raise RepositoryWriteAuthorizationError(
            f"repository write operation {operation!r} is disabled for this service "
            f"instance; an operator must set {REPOSITORY_WRITES_ENV}=1 before "
            "starting the repository-intelligence service"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPOSITORY_WRITE_AUTHORITY_SCHEMA_VERSION,
            "enabled": self.enabled,
            "source": self.source,
            "mutable_by_tool_arguments": False,
        }


__all__ = [
    "REPOSITORY_WRITE_AUTHORITY_SCHEMA_VERSION",
    "REPOSITORY_WRITES_ENV",
    "RepositoryWriteAuthority",
    "RepositoryWriteAuthorizationError",
]
