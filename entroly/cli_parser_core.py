"""Setup, proxy, context, and receipt argument definitions."""

from __future__ import annotations

import argparse

def _add_core_commands(subparsers) -> None:
    """Register setup, proxy, context, and receipt commands."""
    # entroly init
    init_parser = subparsers.add_parser(
        "init",
        help="Auto-detect project + AI tool, generate MCP config",
    )
    init_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be generated without writing files",
    )
    init_parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Non-interactive mode; accept defaults (no-op today; reserved for future prompts)",
    )

    # entroly serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the MCP server with auto-indexing",
    )
    serve_parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug-level logging (all subsystem details to stderr)",
    )

    # entroly attach
    attach_parser = subparsers.add_parser(
        "attach",
        help="Grant scoped, expiring MCP access to an existing agent client",
    )
    attach_subparsers = attach_parser.add_subparsers(dest="attach_action", required=True)
    attach_create = attach_subparsers.add_parser("create", help="Create a local attachment grant")
    attach_create.add_argument("--client", choices=["claude", "codex", "openclaw"], required=True)
    attach_create.add_argument("--project", default=".", help="Project root bound to the grant")
    attach_create.add_argument("--session", default=None, help="Optional client session label")
    attach_create.add_argument("--ttl", default="1h", help="Grant lifetime, for example 30m or 4h")
    attach_create.add_argument(
        "--scope",
        action="append",
        default=[],
        choices=[
            "observe",
            "context",
            "receipts",
            "verify",
            "continuity",
            "remember",
            "record",
            "vault",
        ],
        help="Allowed tool group; repeat to combine groups",
    )
    attach_create.add_argument("--install", action="store_true", help="Run the client MCP configuration command")
    attach_create.add_argument("--json", dest="json_output", action="store_true")
    attach_create.add_argument("--state-dir", default=None, help=argparse.SUPPRESS)
    attach_list = attach_subparsers.add_parser("list", help="List attachment grants")
    attach_list.add_argument("--all", action="store_true", help="Include expired and revoked grants")
    attach_list.add_argument("--json", dest="json_output", action="store_true")
    attach_list.add_argument("--state-dir", default=None, help=argparse.SUPPRESS)
    attach_revoke = attach_subparsers.add_parser("revoke", help="Revoke a grant immediately")
    attach_revoke.add_argument("grant_id")
    attach_revoke.add_argument(
        "--uninstall",
        action="store_true",
        help="Also remove the generated MCP entry from the client",
    )
    attach_revoke.add_argument("--json", dest="json_output", action="store_true")
    attach_revoke.add_argument("--state-dir", default=None, help=argparse.SUPPRESS)
    attach_serve = attach_subparsers.add_parser(
        "serve",
        help="Internal grant-bound MCP transport",
    )
    attach_serve.add_argument("--grant-id", required=True)
    attach_serve.add_argument("--token-file", required=True)
    attach_serve.add_argument("--state-dir", required=True)

    # entroly dashboard
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Launch live web dashboard showing all engine metrics",
    )
    dash_parser.add_argument(
        "--force", action="store_true",
        help="Force re-index even if persistent index exists",
    )
    dash_parser.add_argument(
        "--port", type=int, default=9378,
        help="Dashboard port (default: 9378)",
    )

    # entroly health
    health_parser = subparsers.add_parser(
        "health",
        help="Analyze codebase health (grade A-F, clones, dead code, SAST)",
    )
    health_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show details for each finding",
    )
    health_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit the health and security report as JSON",
    )

    # entroly autotune
    autotune_parser = subparsers.add_parser(
        "autotune",
        help="Optimize engine hyperparameters via mutation-based search",
    )
    autotune_parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of optimization iterations (default: 50)",
    )
    autotune_parser.add_argument(
        "--rollback", action="store_true",
        help="Restore previous tuning_config.json (undo last autotune)",
    )

    # entroly go
    go_parser = subparsers.add_parser(
        "go",
        help="One command: auto-detect, init, proxy, and dashboard",
    )
    go_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: 9377)",
    )
    go_parser.add_argument(
        "--quality", type=str, default=None,
        help="Override auto-detected quality (speed|fast|balanced|quality|max)",
    )
    go_parser.add_argument(
        "--force", action="store_true",
        help="Force re-index even if persistent index exists",
    )

    # entroly proxy
    proxy_parser = subparsers.add_parser(
        "proxy",
        help="Start the invisible prompt compiler proxy (any IDE)",
    )
    proxy_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: 9377, or ENTROLY_PROXY_PORT)",
    )
    proxy_parser.add_argument(
        "--host", type=str, default=None,
        help="Bind host (default: 127.0.0.1, or ENTROLY_PROXY_HOST)",
    )
    proxy_parser.add_argument(
        "--quality", type=str, default=None,
        help="Quality: speed|fast|balanced|quality|max or 0.0-1.0",
    )
    proxy_parser.add_argument(
        "--force", action="store_true",
        help="Force re-index even if persistent index exists",
    )
    proxy_parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug-level logging (all subsystem details to stderr)",
    )
    proxy_parser.add_argument(
        "--autotune-daemon", action="store_true",
        help="Start the benchmark autotune daemon alongside the proxy (off by default)",
    )
    proxy_parser.add_argument(
        "--bypass", action="store_true",
        help="Start in bypass mode (forward requests unmodified, no optimization)",
    )
    proxy_parser.add_argument(
        "--witness", choices=["off", "audit", "annotate", "strict"], default=None,
        help="Verify model outputs before returning them: off, audit, annotate, or strict",
    )
    proxy_parser.add_argument(
        "--witness-nli", action="store_true",
        help="Use OpenAI NLI inside WITNESS when OPENAI_API_KEY is available",
    )
    proxy_parser.add_argument(
        "--witness-embed", action="store_true",
        help="Embed WITNESS certificates in provider JSON instead of sidecar headers only",
    )
    proxy_parser.add_argument(
        "--witness-profile",
        choices=["auto", "code", "rag", "qa", "benchmark_qa", "summary", "chat", "dialogue"],
        default=None,
        help="WITNESS suppression profile (default: auto)",
    )

    # entroly optimize
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Generate optimized context snapshot for a task",
    )
    optimize_parser.add_argument(
        "--task", "-t", type=str, default="",
        help="Description of the task to optimize context for",
    )
    optimize_parser.add_argument(
        "--budget", "-b", type=int, default=8192,
        help="Token budget (default: 8192)",
    )
    optimize_parser.add_argument(
        "--format", "-f", type=str, choices=["markdown", "json"], default="markdown",
        help="Output format (default: markdown)",
    )
    optimize_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output (only emit the snapshot)",
    )
    optimize_parser.add_argument(
        "--selector", type=str, choices=["auto", "knapsack", "dopt", "qccr"], default="auto",
        help="Selection objective. auto (default): qccr if task given, else knapsack. knapsack (linear), dopt (BM25 + log-det), qccr (sentence-level query-conditioned extractive + MMR).",
    )
    optimize_parser.add_argument(
        "--exclude", type=str, action="append", default=[],
        help="Substring to exclude from the fragment source path (repeatable; dopt selector only)",
    )

    # entroly ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents for Context Receipts",
    )
    ingest_parser.add_argument("path", type=str, help="Document file or directory (.md, .txt, .rst)")
    ingest_parser.add_argument("--out", type=str, default=None, help="Index JSON path (default: .entroly/receipts/index.json)")
    ingest_parser.add_argument("--chunk-tokens", type=int, default=360, help="Approximate max tokens per chunk")
    ingest_parser.add_argument("--overlap-tokens", type=int, default=32, help="Token overlap for oversized chunks")
    ingest_parser.add_argument("--python", action="store_true", help="Force the Python reference implementation")

    # entroly select
    select_parser = subparsers.add_parser(
        "select",
        help="Select context and write a Context Receipt",
    )
    select_parser.add_argument("--query", "-q", type=str, required=True, help="Question/task to select context for")
    select_parser.add_argument("--budget", "-b", type=int, default=8000, help="Token budget (default: 8000)")
    select_parser.add_argument("--index", type=str, default=None, help="Index JSON path (default: .entroly/receipts/index.json)")
    select_parser.add_argument("--docs", type=str, default=None, help="Ingest this document path on the fly")
    select_parser.add_argument("--receipt", type=str, default=None, help="Receipt JSON output path")
    select_parser.add_argument("--report", type=str, default=None, help="Markdown report output path")
    select_parser.add_argument("--chunk-tokens", type=int, default=360, help="Approximate max tokens per chunk when using --docs")
    select_parser.add_argument("--overlap-tokens", type=int, default=32, help="Token overlap for oversized chunks when using --docs")
    select_parser.add_argument("--python", action="store_true", help="Force the Python reference implementation")

    # entroly receipt
    receipt_parser = subparsers.add_parser(
        "receipt",
        help="Render a Context Receipt as Markdown",
    )
    receipt_parser.add_argument("receipt_path", type=str, help="Context Receipt JSON path")
    receipt_parser.add_argument("--out", type=str, default=None, help="Markdown output path")
    receipt_parser.add_argument("--json", action="store_true", help="Print normalized receipt JSON instead of Markdown")
    receipt_parser.add_argument("--python", action="store_true", help="Force the Python reference implementation")

    context_commit_parser = subparsers.add_parser(
        "context-commit",
        help="Create or verify a portable proof of the exact selected context",
    )
    context_commit_parser.add_argument(
        "path", nargs="?", default=None, help="Document file or directory"
    )
    context_commit_parser.add_argument(
        "--query", "-q", default=None, help="Question/task to select context for"
    )
    context_commit_parser.add_argument(
        "--budget", "-b", type=int, default=8000, help="Token budget (default: 8000)"
    )
    context_commit_parser.add_argument("--out", type=str, default=None, help="Commit JSON output path")
    context_commit_parser.add_argument("--parent", type=str, default=None, help="Parent Context Commit ID")
    context_commit_parser.add_argument("--verify", type=str, default=None, help="Verify an existing commit JSON")
    context_commit_parser.add_argument("--chunk-tokens", type=int, default=360)
    context_commit_parser.add_argument("--overlap-tokens", type=int, default=32)
    context_commit_parser.add_argument("--python", action="store_true", help="Force the Python reference implementation")
    context_commit_parser.add_argument("--json", action="store_true", help="Print the created JSON artifact")

    proof_parser = subparsers.add_parser(
        "proof",
        help="Run the durable proof-guided context and exact-recovery protocol",
    )
    proof_subparsers = proof_parser.add_subparsers(
        dest="proof_action", required=True
    )

    def _add_proof_prepare_arguments(command_parser):
        command_parser.add_argument("path", help="Document file or directory")
        command_parser.add_argument("--query", "-q", required=True)
        command_parser.add_argument("--budget", "-b", type=int, default=8000)
        command_parser.add_argument("--max-rounds", type=int, default=3)
        command_parser.add_argument("--recovery-budget", type=int, default=1200)
        command_parser.add_argument("--max-chunks-per-round", type=int, default=3)
        command_parser.add_argument(
            "--profile",
            choices=["code", "rag", "qa", "benchmark_qa", "summary", "chat", "dialogue"],
            default="rag",
        )
        command_parser.add_argument("--chunk-tokens", type=int, default=360)
        command_parser.add_argument("--overlap-tokens", type=int, default=32)
        command_parser.add_argument("--idempotency-key", default=None)
        command_parser.add_argument("--state-dir", default=None, help=argparse.SUPPRESS)
        command_parser.add_argument("--python", action="store_true")
        command_parser.add_argument(
            "--allow-high-risk",
            action="store_true",
            help="Continue only after explicitly accepting a high-review receipt",
        )

    proof_prepare = proof_subparsers.add_parser(
        "prepare",
        help="Prepare a model request without calling a provider",
    )
    _add_proof_prepare_arguments(proof_prepare)

    proof_advance = proof_subparsers.add_parser(
        "advance",
        help="Verify one model output and produce the next request or final answer",
    )
    proof_advance.add_argument("session_id")
    proof_advance.add_argument("--output", default=None)
    proof_advance.add_argument("--output-file", default=None)
    proof_advance.add_argument("--idempotency-key", required=True)
    proof_advance.add_argument("--state-dir", default=None, help=argparse.SUPPRESS)
    proof_advance.add_argument("--python", action="store_true", help=argparse.SUPPRESS)
    proof_advance.add_argument("--allow-high-risk", action="store_true", help=argparse.SUPPRESS)

    proof_inspect = proof_subparsers.add_parser(
        "inspect",
        help="Inspect the last durable response without advancing it",
    )
    proof_inspect.add_argument("session_id")
    proof_inspect.add_argument("--state-dir", default=None, help=argparse.SUPPRESS)
    proof_inspect.add_argument("--python", action="store_true", help=argparse.SUPPRESS)
    proof_inspect.add_argument("--allow-high-risk", action="store_true", help=argparse.SUPPRESS)

    proof_run = proof_subparsers.add_parser(
        "run",
        help="Run all bounded rounds through an explicit local model command",
    )
    _add_proof_prepare_arguments(proof_run)
    proof_run.add_argument(
        "--model-command",
        required=True,
        help="Command that reads request JSON on stdin and writes model text or {\"output\": ...}",
    )
    proof_run.add_argument("--model-timeout", type=float, default=120.0)
    proof_run.add_argument("--json", dest="json_output", action="store_true")

    # entroly audit
    audit_parser = subparsers.add_parser(
        "audit",
        help="Render a multi-turn session-chain audit report",
    )
    audit_parser.add_argument("session_chain", type=str, help="session_chain.json path")
    audit_parser.add_argument(
        "--taint",
        type=str,
        default=None,
        help="Optional hallucination taint JSON path (default: sibling session_taint.json or taint.json)",
    )
    audit_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    audit_parser.add_argument(
        "--input-price-per-million",
        type=float,
        default=0.0,
        help="Optional input-token price for estimated per-turn cost attribution",
    )
    audit_parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns to show in the text ledger (default: 20; -1 = all, 0 = none)",
    )

    # entroly explain
    explain_parser = subparsers.add_parser(
        "explain",
        help="Explain receipt decisions such as why a chunk was omitted",
    )
    explain_parser.add_argument("--why-omitted", required=True, help="Chunk id to explain")
    explain_parser.add_argument("--receipt", type=str, default=None, help="Receipt JSON path (default: latest)")
    explain_parser.add_argument("--python", action="store_true", help="Force the Python reference implementation")

    # entroly feedback
    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Signal outcome quality to improve future context selection",
    )
    feedback_parser.add_argument(
        "--score", "-s", type=float, default=None,
        help="Quality score: 0.0 (bad) to 1.0 (good). Mutually exclusive with --outcome.",
    )
    feedback_parser.add_argument(
        "--outcome", choices=["success", "good", "fail", "failure", "bad", "neutral"],
        default=None,
        help="Symbolic outcome; mapped to a score (success/good→1.0, fail/bad→0.0, neutral→0.5).",
    )
    feedback_parser.add_argument(
        "--task", type=str, default=None,
        help="Optional task description for audit logs (metadata only).",
    )
