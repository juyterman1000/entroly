"""Repository-intelligence service facade with external write authority.

The full read/analysis/planning implementation is preserved byte-for-byte in
:mod:`service_impl`. This facade changes only destructive service-level apply
methods: tool arguments can acknowledge risk, but cannot authorize writes.
"""
from __future__ import annotations

from .service_impl import *  # noqa: F401,F403
from .service_impl import RepositoryIntelligenceService as _BaseRepositoryIntelligenceService
from .write_authority import RepositoryWriteAuthority


class RepositoryIntelligenceService(_BaseRepositoryIntelligenceService):
    """Base service plus immutable process-start write authority."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._repository_write_authority = RepositoryWriteAuthority.from_environment()

    def write_authority_status(self) -> dict[str, object]:
        """Return non-secret authority state for diagnostics/tests."""
        return self._repository_write_authority.to_dict()

    def rename_apply(self, *args, **kwargs):
        self._repository_write_authority.require("rename")
        return super().rename_apply(*args, **kwargs)

    def safe_delete_apply(self, *args, **kwargs):
        self._repository_write_authority.require("safe-delete")
        return super().safe_delete_apply(*args, **kwargs)

    def file_move_apply(self, *args, **kwargs):
        self._repository_write_authority.require("file-move")
        return super().file_move_apply(*args, **kwargs)
