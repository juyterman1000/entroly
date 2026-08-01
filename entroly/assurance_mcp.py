"""Focused MCP server for assurance-gated compression and repo intelligence."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .assurance_sdk import compress_assured, compress_file_assured, compress_messages_assured
from .assurance_telemetry import AssuranceLedger
from .domain_assurance import validate_domain_output
from .repo_intelligence import RepositoryIntelligence

_MAX_TEXT_CHARS = 8_000_000
_MAX_MESSAGES = 10_000
_MAX_PATH_CHARS = 4096


def _default_ledger_path() -> Path:
    configured = os.environ.get("ENTROLY_ASSURANCE_LEDGER")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("ENTROLY_DIR", ".entroly")).expanduser() / "assurance-ledger.sqlite3"


def _error(operation: str, exc: Exception) -> str:
    return json.dumps(
        {"status": "error", "operation": operation, "error": str(exc)[:600]},
        indent=2,
        ensure_ascii=False,
    )


def _safe_workspace(path: str) -> Path:
    if not path or len(path) > _MAX_PATH_CHARS or "\x00" in path:
        raise ValueError("workspace path is not a safe bounded path")
    try:
        candidate = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("workspace must be an existing directory") from exc
    if not candidate.is_dir():
        raise ValueError("workspace must be an existing directory")
    return candidate


def _safe_workspace_file(workspace: str, path: str) -> tuple[Path, Path]:
    root = _safe_workspace(workspace)
    if not path or len(path) > _MAX_PATH_CHARS or "\x00" in path:
        raise ValueError("file path is not a safe bounded path")
    raw = Path(path).expanduser()
    try:
        candidate = (root / raw).resolve(strict=True) if not raw.is_absolute() else raw.resolve(strict=True)
    except OSError as exc:
        raise ValueError("file must exist inside workspace") from exc
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("file path escapes workspace") from exc
    if not candidate.is_file():
        raise ValueError("file must exist inside workspace")
    return candidate, relative


def create_assurance_mcp_server(ledger_path: str | None = None):
    """Create the opt-in assurance and repository-intelligence MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise RuntimeError("MCP SDK not installed. Install with: pip install mcp") from None

    mcp = FastMCP(
        "entroly-assurance",
        instructions=(
            "Use assurance-gated compression when evidence loss is costly. "
            "Structural scope validates exact atomic selection; semantic scope "
            "requires a held-out calibration profile and otherwise fails closed."
        ),
    )
    ledger = AssuranceLedger(ledger_path or _default_ledger_path())

    @mcp.tool()
    def assured_compress_text(
        text: str,
        query: str,
        token_budget: int,
        required_scope: str = "candidate_units",
        fallback: str = "original",
        content_type: str = "auto",
    ) -> str:
        """Compress text with an explicit assurance scope and fallback decision."""
        try:
            if len(text) > _MAX_TEXT_CHARS:
                raise ValueError(f"text exceeds {_MAX_TEXT_CHARS:,} characters")
            result = compress_assured(
                text,
                query=query,
                budget=int(token_budget),
                content_type=content_type,
                required_scope=required_scope,
                fallback=fallback,
                ledger=ledger,
            )
            payload = result.to_dict()
            payload["status"] = "ok"
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error("assured_compress_text", exc)

    @mcp.tool()
    def assured_compress_file(
        workspace: str,
        path: str,
        query: str,
        token_budget: int,
        required_scope: str = "candidate_units",
        fallback: str = "original",
        content_type: str = "auto",
    ) -> str:
        """Compress a UTF-8 workspace file with path-aware preservation guards."""
        try:
            root = _safe_workspace(workspace)
            _candidate, relative = _safe_workspace_file(workspace, path)
            result = compress_file_assured(
                relative,
                workspace=root,
                query=query,
                budget=int(token_budget),
                content_type=content_type,
                required_scope=required_scope,
                fallback=fallback,
                ledger=ledger,
            )
            payload = result.to_dict()
            payload["status"] = "ok"
            payload["path"] = relative.as_posix()
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error("assured_compress_file", exc)

    @mcp.tool()
    def assured_compress_messages(
        messages_json: str,
        token_budget: int,
        query: str = "",
        preserve_last_n: int = 4,
        required_scope: str = "candidate_units",
        fallback: str = "original",
    ) -> str:
        """Compress older conversation messages while preserving recent turns."""
        try:
            messages = json.loads(messages_json)
            if not isinstance(messages, list):
                raise ValueError("messages_json must contain a JSON array")
            if len(messages) > _MAX_MESSAGES:
                raise ValueError(f"messages_json exceeds {_MAX_MESSAGES:,} messages")
            if any(not isinstance(message, dict) for message in messages):
                raise ValueError("each message must be a JSON object")
            result = compress_messages_assured(
                messages,
                budget=int(token_budget),
                query=query or None,
                preserve_last_n=int(preserve_last_n),
                required_scope=required_scope,
                fallback=fallback,
                ledger=ledger,
            )
            payload = result.to_dict()
            payload["status"] = "ok"
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _error("assured_compress_messages", exc)

    @mcp.tool()
    def validate_compressed_output(
        original: str,
        emitted: str,
        content_type: str,
        query: str = "",
    ) -> str:
        """Run a workload-specific validity/evidence oracle without compressing."""
        try:
            if len(original) + len(emitted) > _MAX_TEXT_CHARS:
                raise ValueError("combined payload exceeds the MCP validation limit")
            result = validate_domain_output(
                original, emitted, content_type=content_type, query=query
            )
            return json.dumps(
                {"status": "ok", "validation": result.to_dict()},
                indent=2,
                ensure_ascii=False,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return _error("validate_compressed_output", exc)

    @mcp.tool()
    def assurance_stats(since_unix: float = 0.0) -> str:
        """Return local decision coverage, bypass, savings, and latency statistics."""
        try:
            summary = ledger.summary(since=since_unix or None)
            return json.dumps(
                {"status": "ok", "summary": summary.to_dict()},
                indent=2,
                ensure_ascii=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error("assurance_stats", exc)

    @mcp.tool()
    def repo_impact(
        workspace: str,
        changed_paths_json: str,
        max_depth: int = 3,
        max_files: int = 100,
    ) -> str:
        """Find files and tests transitively affected by changed repository files."""
        try:
            root = _safe_workspace(workspace)
            changed = json.loads(changed_paths_json)
            if not isinstance(changed, list) or any(not isinstance(item, str) for item in changed):
                raise ValueError("changed_paths_json must contain a JSON string array")
            intelligence = RepositoryIntelligence.scan(root)
            report = intelligence.impact_report(
                changed,
                max_depth=max(0, min(int(max_depth), 10)),
                max_files=max(1, min(int(max_files), 1_000)),
            )
            return json.dumps(
                {"status": "ok", "impact": report.to_dict()},
                indent=2,
                ensure_ascii=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _error("repo_impact", exc)

    @mcp.tool()
    def repo_overview(
        workspace: str,
        top_n: int = 20,
    ) -> str:
        """Summarize languages, graph size, instruction files, and important files."""
        try:
            root = _safe_workspace(workspace)
            overview = RepositoryIntelligence.scan(root).overview(
                top_n=max(1, min(int(top_n), 200))
            )
            return json.dumps(
                {"status": "ok", "overview": overview.to_dict()},
                indent=2,
                ensure_ascii=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error("repo_overview", exc)

    @mcp.tool()
    def repo_smells(
        workspace: str,
        max_findings: int = 100,
    ) -> str:
        """Return bounded deterministic structural code-smell findings."""
        try:
            root = _safe_workspace(workspace)
            report = RepositoryIntelligence.scan(root).smell_report(
                max_findings=max(1, min(int(max_findings), 1_000))
            )
            return json.dumps(
                {"status": "ok", "smells": report.to_dict()},
                indent=2,
                ensure_ascii=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error("repo_smells", exc)

    @mcp.tool()
    def repo_context_bundle(
        workspace: str,
        query: str,
        changed_paths_json: str,
        token_budget: int = 8_000,
        max_depth: int = 3,
    ) -> str:
        """Build a whole-line dependency-aware context bundle with exact line ranges."""
        try:
            root = _safe_workspace(workspace)
            changed = json.loads(changed_paths_json)
            if not isinstance(changed, list) or any(not isinstance(item, str) for item in changed):
                raise ValueError("changed_paths_json must contain a JSON string array")
            intelligence = RepositoryIntelligence.scan(root)
            bundle = intelligence.context_bundle(
                query=query,
                changed_paths=changed,
                budget_tokens=max(1, min(int(token_budget), 200_000)),
                max_depth=max(0, min(int(max_depth), 10)),
            )
            payload = bundle.to_dict()
            payload["rendered"] = bundle.render()
            return json.dumps(
                {"status": "ok", "bundle": payload},
                indent=2,
                ensure_ascii=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _error("repo_context_bundle", exc)

    return mcp


def main() -> None:
    create_assurance_mcp_server().run()


if __name__ == "__main__":
    main()


__all__ = ["create_assurance_mcp_server"]
