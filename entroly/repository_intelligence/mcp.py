"""Focused MCP server for bounded repository impact and test localization."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .models import RepositoryLimits
from .service import (
    RepositoryIntelligenceError,
    RepositoryIntelligenceService,
    VerifiedRefactorError,
)


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


def _configured_lsp_command() -> tuple[str, ...]:
    raw = os.environ.get("ENTROLY_LSP_COMMAND_JSON", "").strip()
    if not raw:
        raise VerifiedRefactorError(
            "LSP orchestration is disabled; operator must set ENTROLY_LSP_COMMAND_JSON"
        )
    try:
        command = json.loads(raw)
    except json.JSONDecodeError:
        raise VerifiedRefactorError(
            "ENTROLY_LSP_COMMAND_JSON must be a JSON argument array"
        ) from None
    if (
        not isinstance(command, list)
        or not 1 <= len(command) <= 32
        or any(not isinstance(item, str) or not item or len(item) > 4096 for item in command)
    ):
        raise VerifiedRefactorError(
            "ENTROLY_LSP_COMMAND_JSON must contain 1 to 32 strings"
        )
    return tuple(command)


def _error(exc: Exception, operation: str) -> str:
    if isinstance(exc, RepositoryIntelligenceError):
        payload = exc.to_dict()
    else:
        # Exception messages can contain absolute paths, environment details,
        # or parser internals. Keep the model-facing error useful but local.
        payload = {
            "schema_version": "entroly.repository-service.v2",
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
    cache_dir: str | os.PathLike[str] | None = None,
):
    """Create a workspace-fixed MCP server with no caller-controlled root."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise RuntimeError("MCP SDK not installed. Install with: pip install mcp") from None

    service = RepositoryIntelligenceService(
        _configured_root(root),
        limits=limits,
        cache_dir=cache_dir,
    )
    mcp = FastMCP(
        "entroly-repository-intelligence",
        instructions=(
            "Inspect a fixed local repository using bounded symbol, import, and "
            "call graphs. Paths are workspace-relative and absolute local paths "
            "are never returned. Source bytes are exposed only by the bounded "
            "verified-context tool, labeled untrusted, with exact hashes. All "
            "operations are read-only except repository_rename_apply, which "
            "requires a prior committed preview, its exact plan hash, and an "
            "explicit acknowledgement that reference completeness is unproven."
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
    def repository_verified_context(
        query: str,
        token_budget: int = 2_000,
        max_hops: int = 2,
        max_fragments: int = 24,
        include_history: bool = False,
        max_history_commits: int = 20,
    ) -> str:
        """Return a partial code graph scoped to one task with a receipt."""
        try:
            return _json(service.context(
                query,
                token_budget=max(128, min(int(token_budget), 32_768)),
                max_hops=max(0, min(int(max_hops), 6)),
                max_fragments=max(1, min(int(max_fragments), 100)),
                include_history=bool(include_history),
                max_history_commits=max(1, min(int(max_history_commits), 100)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_verified_context")

    @mcp.tool()
    def repository_symbol_graph(
        symbol_query: str,
        direction: str = "both",
        max_depth: int = 3,
        limit: int = 200,
    ) -> str:
        """Trace freshness-checked static calls without guessing symbol identity."""
        try:
            return _json(service.symbol_graph(
                symbol_query,
                direction=direction,
                max_depth=max(0, min(int(max_depth), 12)),
                limit=max(1, min(int(limit), 5_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_symbol_graph")

    @mcp.tool()
    def repository_graph_query(
        query: str,
        operation: str = "neighbors",
        target_query: str | None = None,
        direction: str = "both",
        max_depth: int = 4,
        limit: int = 100,
    ) -> str:
        """Query verified typed neighbors, paths, relatedness, or impact."""
        try:
            return _json(service.graph_query(
                query,
                operation=operation,
                target_query=target_query,
                direction=direction,
                max_depth=max(0, min(int(max_depth), 20)),
                limit=max(1, min(int(limit), 5_000)),
                max_visited=10_000,
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_graph_query")

    @mcp.tool()
    def repository_map(
        query: str = "",
        token_budget: int = 2_000,
        max_entries: int = 100,
    ) -> str:
        """Rank a receipt-backed structural map across the fixed repository."""
        try:
            return _json(service.repository_map(
                query,
                token_budget=max(128, min(int(token_budget), 32_768)),
                max_entries=max(1, min(int(max_entries), 1_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_map")

    @mcp.tool()
    def repository_program_graph(
        symbol_query: str,
        limit: int = 1_000,
    ) -> str:
        """Return verified Python control flow and reaching definitions."""
        try:
            return _json(service.program_graph(
                symbol_query,
                limit=max(16, min(int(limit), 10_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_program_graph")

    @mcp.tool()
    def repository_code_health(
        max_findings: int = 500,
        max_symbols: int = 2_000,
    ) -> str:
        """Audit verified complexity, cycles, coupling, and navigability risk."""
        try:
            return _json(service.code_health(
                max_findings=max(1, min(int(max_findings), 10_000)),
                max_symbols=max(1, min(int(max_symbols), 20_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_code_health")

    @mcp.tool()
    def repository_architecture(
        max_components: int = 5_000,
        max_communities: int = 1_000,
        max_dependency_edges: int = 100_000,
        max_hotspots: int = 100,
        max_routes: int = 100,
    ) -> str:
        """Return verified layers, communities, cycles, routes, and hotspots."""
        try:
            return _json(service.architecture(
                max_components=max(1, min(int(max_components), 20_000)),
                max_communities=max(1, min(int(max_communities), 10_000)),
                max_dependency_edges=max(
                    1, min(int(max_dependency_edges), 1_000_000)
                ),
                max_hotspots=max(1, min(int(max_hotspots), 1_000)),
                max_routes=max(1, min(int(max_routes), 1_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_architecture")

    @mcp.tool()
    def repository_architecture_diff(
        before: dict[str, object],
        after: dict[str, object],
        limit: int = 5_000,
    ) -> str:
        """Compare two committed architecture receipts and report structural drift."""
        try:
            if len(json.dumps([before, after]).encode("utf-8")) > 32 * 1024 * 1024:
                raise ValueError("combined architecture inputs must be at most 32 MiB")
            return _json(service.architecture_diff(
                before,
                after,
                limit=max(1, min(int(limit), 50_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_architecture_diff")

    @mcp.tool()
    def repository_git_architecture_diff(
        ref: str = "HEAD",
        max_changes: int = 10_000,
        max_dependency_edges: int = 100_000,
    ) -> str:
        """Compare a local Git commit with the verified worktree without checkout."""
        try:
            return _json(service.git_architecture_diff(
                ref,
                max_changes=max(1, min(int(max_changes), 50_000)),
                max_dependency_edges=max(
                    1, min(int(max_dependency_edges), 1_000_000)
                ),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_git_architecture_diff")

    @mcp.tool()
    def repository_http_routes(
        method: str | None = None,
        path_prefix: str | None = None,
        max_routes: int = 10_000,
        max_conflicts: int = 1_000,
    ) -> str:
        """Discover verified HTTP routes, mounted prefixes, handlers, and collisions."""
        try:
            return _json(service.routes(
                method=method,
                path_prefix=path_prefix,
                max_routes=max(1, min(int(max_routes), 100_000)),
                max_conflicts=max(1, min(int(max_conflicts), 10_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_http_routes")

    @mcp.tool()
    def repository_graph_snapshot() -> str:
        """Export the complete bounded graph as a portable committed snapshot."""
        try:
            return _json(service.graph_snapshot())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_graph_snapshot")

    @mcp.tool()
    def repository_graph_snapshot_check(
        snapshot: dict[str, object],
        limit: int = 10_000,
    ) -> str:
        """Verify a shared snapshot and prove whether it matches current sources."""
        try:
            if len(json.dumps(snapshot).encode("utf-8")) > 256 * 1024 * 1024:
                raise ValueError("graph snapshot must be at most 256 MiB")
            return _json(service.graph_snapshot_check(
                snapshot,
                limit=max(1, min(int(limit), 100_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_graph_snapshot_check")

    @mcp.tool()
    def repository_runtime_overlay(
        events: list[dict[str, object]],
        producer: str = "external-trace",
        max_events: int = 100_000,
    ) -> str:
        """Bind value-free runtime events to fresh source and symbol evidence."""
        try:
            return _json(service.runtime_overlay(
                events,
                producer=producer,
                max_events=max(1, min(int(max_events), 1_000_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_runtime_overlay")

    @mcp.tool()
    def repository_semantic_overlay(
        relationships: list[dict[str, object]],
        provider: str,
        max_relationships: int = 100_000,
    ) -> str:
        """Verify external LSP/compiler ranges before trusting semantic edges."""
        try:
            return _json(service.semantic_overlay(
                relationships,
                provider=provider,
                max_relationships=max(1, min(int(max_relationships), 1_000_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_semantic_overlay")

    @mcp.tool()
    def repository_rename_preview(
        symbol_query: str,
        new_name: str,
        semantic_relationships: list[dict[str, object]] | None = None,
        provider: str = "none",
        max_changes: int = 10_000,
    ) -> str:
        """Preview exact rename edits; performs no writes and reports incompleteness."""
        try:
            return _json(service.rename_preview(
                symbol_query,
                new_name,
                semantic_relationships=semantic_relationships or (),
                provider=provider,
                max_changes=max(1, min(int(max_changes), 100_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_rename_preview")

    @mcp.tool()
    def repository_rename_apply(
        plan: dict[str, object],
        expected_plan_sha256: str,
        acknowledge_incomplete: bool = False,
    ) -> str:
        """Apply a previewed rename after plan-hash and risk acknowledgement."""
        try:
            return _json(service.rename_apply(
                plan,
                expected_plan_sha256=expected_plan_sha256,
                acknowledge_incomplete=bool(acknowledge_incomplete),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_rename_apply")

    @mcp.tool()
    def repository_safe_delete_preview(
        symbol_query: str,
        max_blockers: int = 10_000,
    ) -> str:
        """Preview a headless delete; any known or lexical reference blocks it."""
        try:
            return _json(service.safe_delete_preview(
                symbol_query,
                max_blockers=max(1, min(int(max_blockers), 100_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_safe_delete_preview")

    @mcp.tool()
    def repository_safe_delete_apply(
        plan: dict[str, object],
        expected_plan_sha256: str,
        acknowledge_incomplete: bool = False,
    ) -> str:
        """Apply a blocker-free delete after plan-hash and risk acknowledgement."""
        try:
            return _json(service.safe_delete_apply(
                plan,
                expected_plan_sha256=expected_plan_sha256,
                acknowledge_incomplete=bool(acknowledge_incomplete),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_safe_delete_apply")

    @mcp.tool()
    def repository_file_move_preview(
        source_path: str,
        target_path: str,
        max_changes: int = 10_000,
        max_blockers: int = 10_000,
    ) -> str:
        """Preview a headless Python module move with exact import rewrites."""
        try:
            return _json(service.file_move_preview(
                source_path,
                target_path,
                max_changes=max(1, min(int(max_changes), 100_000)),
                max_blockers=max(1, min(int(max_blockers), 100_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_file_move_preview")

    @mcp.tool()
    def repository_file_move_apply(
        plan: dict[str, object],
        expected_plan_sha256: str,
        acknowledge_incomplete: bool = False,
    ) -> str:
        """Apply a blocker-free module move after commitment and risk checks."""
        try:
            return _json(service.file_move_apply(
                plan,
                expected_plan_sha256=expected_plan_sha256,
                acknowledge_incomplete=bool(acknowledge_incomplete),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_file_move_apply")

    @mcp.tool()
    def repository_lsp_rename_preview(
        symbol_query: str,
        new_name: str,
        language_id: str,
        timeout_seconds: float = 15.0,
        max_relationships: int = 10_000,
    ) -> str:
        """Run the operator-configured LSP and return a committed no-write plan."""
        try:
            return _json(service.lsp_rename_preview(
                symbol_query,
                new_name,
                command=_configured_lsp_command(),
                language_id=language_id,
                timeout_seconds=max(1.0, min(float(timeout_seconds), 30.0)),
                max_relationships=max(1, min(int(max_relationships), 100_000)),
            ))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error(exc, "repository_lsp_rename_preview")

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
