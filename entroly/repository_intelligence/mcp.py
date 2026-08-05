"""Focused MCP server for bounded repository impact and test localization."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .models import RepositoryLimits
from .service import RepositoryIntelligenceError, RepositoryIntelligenceService


def _configured_root(root: str | os.PathLike[str] | None) -> Path:
    selected = root if root is not None else os.environ.get("ENTROLY_REPOSITORY_ROOT", ".")
    resolved = Path(selected).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def _mcp_safe(payload: dict[str, object]) -> dict[str, object]:
    """Return a copy safe to expose to a model-facing MCP client."""
    safe = dict(payload)
    if "root" in safe:
        safe["root"] = "."
    return safe


def _error(exc: Exception, operation: str) -> str:
    if isinstance(exc, RepositoryIntelligenceError):
        payload = exc.to_dict()
    else:
        # Exception messages can contain absolute paths, environment details,
        # or parser internals. Keep the model-facing error useful but local.
        payload = {
            "schema_version": "entroly.repository-service.v1",
            "error": "repository_operation_failed",
            "detail": "repository operation failed; inspect local server logs",
            "error_type": type(exc).__name__,
        }
    payload["operation"] = operation
    return _json(payload)


def create_repository_mcp_server(
    root: str | os.PathLike[str] | None = None,
    *,
    limits: RepositoryLimits | None = None,
):
    """Create a workspace-fixed MCP server with no caller-controlled root."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise RuntimeError("MCP SDK not installed. Install with: pip install mcp") from None

    service = RepositoryIntelligenceService(_configured_root(root), limits=limits)
    mcp = FastMCP(
        "entroly-repository-intelligence",
        instructions=(
            "Inspect a fixed local repository using bounded symbol, import, and "
            "call graphs. Tools return workspace-relative paths and metadata, "
            "never source bytes or absolute local paths."
        ),
    )

    @mcp.tool()
    def repository_summary(refresh: bool = False) -> str:
        """Return bounded counts and the deterministic repository-index digest."""
        try:
            payload = service.refresh() if refresh else service.summary()
            return _json(_mcp_safe(payload))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_summary")

    @mcp.tool()
    def repository_change_impact(
        changed_paths: list[str],
        max_depth: int = 4,
        limit: int = 500,
    ) -> str:
        """Return reverse file/call impact for known workspace-relative paths."""
        try:
            return _json(
                service.impact(
                    changed_paths,
                    max_depth=max(0, min(int(max_depth), 12)),
                    limit=max(1, min(int(limit), 5_000)),
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_change_impact")

    @mcp.tool()
    def repository_tests_for_changes(
        changed_paths: list[str],
        limit: int = 20,
    ) -> str:
        """Rank tests related to known changed paths without executing them."""
        try:
            return _json(service.tests(changed_paths, limit=max(1, min(int(limit), 100))))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_tests_for_changes")

    @mcp.tool()
    def refresh_repository_index() -> str:
        """Atomically rebuild the fixed repository snapshot."""
        try:
            return _json(_mcp_safe(service.refresh()))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "refresh_repository_index")

    return mcp


def main() -> None:
    create_repository_mcp_server().run()


if __name__ == "__main__":
    main()


__all__ = ["create_repository_mcp_server"]
