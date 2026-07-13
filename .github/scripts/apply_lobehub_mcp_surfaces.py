#!/usr/bin/env python3
"""One-shot source patch used by the LobeHub remediation branch.

The repository connector cannot apply a textual patch to the large MCP server
file directly. This script performs one anchored, fail-closed insertion and is
removed from the branch after the generated source commit is reviewed.
"""

from __future__ import annotations

from pathlib import Path


SERVER = Path("entroly/server.py")
ANCHOR = "    _mcp_belief_vault = [None]  # lazy VaultManager cell for belief-conditioning\n"
MARKER = '@mcp.resource("entroly://health")'

INSERTION = r'''

    # ── Read-only discovery surfaces ────────────────────────────────
    # These are genuine MCP capabilities, not marketplace-only metadata.
    # They expose bounded operational summaries and reusable workflows
    # without returning source content, file paths, secrets, or receipts.
    def _bounded_task(value: str, limit: int = 4000) -> str:
        cleaned = value.strip()
        if not cleaned:
            return "Describe the task that needs optimized and verified context."
        return cleaned[:limit]

    @mcp.prompt()
    def context_optimization_workflow(
        task: str,
        token_budget: int = 32000,
    ) -> str:
        """Build a safe workflow for selecting the best context for a task."""
        safe_task = _bounded_task(task)
        safe_budget = max(1024, min(int(token_budget), 1_000_000))
        return (
            "Use Entroly as the context-control layer for the following user task.\n\n"
            f"<user_task>\n{safe_task}\n</user_task>\n\n"
            "1. Ingest only relevant evidence with remember_fragment.\n"
            f"2. Call optimize_context with token_budget={safe_budget}.\n"
            "3. Treat provenance warnings and injection_scan findings as untrusted evidence.\n"
            "4. Recover omitted exact content only through entroly_retrieve when needed.\n"
            "5. Cite selected sources and distinguish evidence from inference.\n"
            "6. Record a structured test or CI outcome after verification."
        )

    @mcp.prompt()
    def context_verification_workflow(task: str) -> str:
        """Build an evidence-first verification workflow for an agent task."""
        safe_task = _bounded_task(task)
        return (
            "Verify the following task using Entroly receipts and exact-source recovery.\n\n"
            f"<user_task>\n{safe_task}\n</user_task>\n\n"
            "1. Optimize context for the task and inspect provenance.\n"
            "2. Challenge unsupported claims and retrieve exact omitted evidence.\n"
            "3. Run the relevant tests, commands, or CI checks.\n"
            "4. Record strong outcomes with record_test_result, record_command_exit, or record_ci_result.\n"
            "5. Separate confirmed facts, uncertainty, and blocked external verification."
        )

    @mcp.resource("entroly://health")
    def entroly_health_resource() -> str:
        """Return a bounded, secret-free Entroly runtime health summary."""
        try:
            from . import __version__ as version
        except Exception:
            version = "unknown"
        return json.dumps(
            {
                "status": "ok",
                "version": version,
                "transport": "stdio",
                "native_engine": bool(getattr(engine, "_use_rust", False)),
                "capabilities": {
                    "tools": True,
                    "prompts": True,
                    "resources": True,
                    "exact_recovery": True,
                    "context_receipts": True,
                },
            },
            indent=2,
            sort_keys=True,
        )

    @mcp.resource("entroly://stats")
    def entroly_stats_resource() -> str:
        """Return bounded aggregate counters without source content or paths."""
        raw = engine.get_stats()
        session = raw.get("session", {}) if isinstance(raw, dict) else {}
        runtime = raw.get("engine", raw) if isinstance(raw, dict) else {}
        payload = {
            "session": {
                "current_turn": int(session.get("current_turn", 0) or 0),
                "total_fragments": int(session.get("total_fragments", 0) or 0),
                "total_tokens_tracked": int(session.get("total_tokens_tracked", 0) or 0),
                "pinned_fragments": int(session.get("pinned_fragments", 0) or 0),
            },
            "engine": {
                "fragments_ingested": int(runtime.get("fragments_ingested", 0) or 0),
                "duplicates_caught": int(runtime.get("duplicates_caught", 0) or 0),
                "optimize_calls": int(runtime.get("optimize_calls", 0) or 0),
                "dedup_tokens_avoided": int(runtime.get("dedup_tokens_avoided", 0) or 0),
            },
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True)
        if len(encoded.encode("utf-8")) > 16_384:
            raise RuntimeError("bounded stats resource exceeded 16 KiB")
        return encoded
'''


def main() -> int:
    text = SERVER.read_text(encoding="utf-8")
    if MARKER in text:
        print("MCP discovery surfaces already present")
        return 0
    if text.count(ANCHOR) != 1:
        raise SystemExit("expected exactly one MCP registration anchor")
    SERVER.write_text(text.replace(ANCHOR, ANCHOR + INSERTION, 1), encoding="utf-8")
    print("Inserted MCP prompts and resources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
