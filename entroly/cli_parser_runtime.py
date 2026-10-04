"""Diagnostic, verification, and daemon arguments."""

from __future__ import annotations

import argparse

def _add_tools_and_runtime_commands(subparsers) -> None:
    """Register diagnostic, verification, and daemon commands."""
    # entroly doctor (Gap #52)
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Diagnose common issues: index, config, proxy, weights",
    )
    doctor_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port to check (default: 9377)",
    )
    doctor_parser.add_argument(
        "--privacy", action="store_true", default=False,
        help="Inspect privacy configuration and source heuristics (not a network audit)",
    )
    doctor_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit a stable machine-readable local diagnostic report",
    )

    # entroly digest (Gap #44)
    digest_parser = subparsers.add_parser(
        "digest",
        help="Show weekly summary of entroly's value (tokens saved, costs, etc.)",
    )
    digest_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: 9377)",
    )

    # entroly migrate (Gap #53)
    subparsers.add_parser(
        "migrate",
        help="Auto-migrate config/index to current version format",
    )

    # entroly role (Gap #49)
    role_parser = subparsers.add_parser(
        "role",
        help="Role-based weight presets (frontend, backend, sre, data, fullstack)",
    )
    role_parser.add_argument(
        "role_action", nargs="?", choices=["list", "apply"], default="list",
        help="List available roles or apply one (default: list)",
    )
    role_parser.add_argument(
        "name", nargs="?", default=None,
        help="Role name to apply",
    )
    role_parser.add_argument(
        "--preset", type=str, default=None,
        help="Shorthand: entroly role --preset backend",
    )

    # entroly completions
    comp_parser = subparsers.add_parser(
        "completions",
        help="Generate shell completion script (bash|zsh|fish)",
    )
    comp_parser.add_argument(
        "shell", choices=["bash", "zsh", "fish"],
        help="Shell type",
    )

    # entroly compile
    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile source code into persistent belief artifacts (Cross-Session Memory)",
    )
    compile_parser.add_argument(
        "directory", nargs="?", default=None,
        help="Directory to scan (default: current directory)",
    )
    compile_parser.add_argument(
        "--max-files", type=int, default=0,
        help="Maximum files to process (default: 0 = unlimited)",
    )
    compile_parser.add_argument(
        "--no-retract", action="store_true",
        help=(
            "Keep beliefs whose source file no longer exists. By default a "
            "compile also retracts them, since they would otherwise be "
            "returned beside live beliefs at full confidence."
        ),
    )

    # entroly verify
    subparsers.add_parser(
        "verify",
        help="Run verification pass on all beliefs (staleness, contradictions)",
    )

    verify_claims_parser = subparsers.add_parser(
        "verify-claims",
        help="Run packaged install and README smoke verification",
    )
    verify_claims_parser.add_argument(
        "--output", "-o", default=None,
        help="Machine-readable report path (default: .entroly_verification.json)",
    )
    verify_claims_parser.add_argument(
        "--max-files", type=int, default=120,
        help="Maximum files to sample for the bounded smoke check (default: 120)",
    )

    # entroly verify-code — statically verify LLM-generated code
    # Pass-through: all remaining args are forwarded to verifiers/cli.py,
    # which has its own arg parser (handles `path`, --repo, --lambda,
    # --threshold, --json, --rebuild, --max-items).
    import argparse as _argparse
    verify_code_parser = subparsers.add_parser(
        "verify-code",
        help="Statically verify LLM-generated code against the codebase symbol manifest",
    )
    verify_code_parser.add_argument(
        "verify_code_args",
        nargs=_argparse.REMAINDER,
        help="Path to source file (use '-' for stdin) plus optional flags",
    )

    # entroly sync
    sync_parser = subparsers.add_parser(
        "sync",
        help="Detect workspace changes and update beliefs (Change-Driven Pipeline)",
    )
    sync_parser.add_argument(
        "directory", nargs="?", default=None,
        help="Directory to scan (default: current directory)",
    )
    sync_parser.add_argument(
        "--max-files", type=int, default=0,
        help="Maximum files to process (default: 0 = unlimited)",
    )
    sync_parser.add_argument(
        "--force", action="store_true",
        help="Force full rescan even if no changes detected",
    )

    # entroly search
    search_parser = subparsers.add_parser(
        "search",
        help="Full-text TF-IDF search across vault beliefs (Rust engine)",
    )
    search_parser.add_argument(
        "query", nargs="+",
        help="Search query (e.g., 'knapsack optimization')",
    )
    search_parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)",
    )

    # entroly docs
    docs_parser = subparsers.add_parser(
        "docs",
        help="Compile markdown docs (README, ARCHITECTURE, docs/) into beliefs",
    )
    docs_parser.add_argument(
        "directory", nargs="?", default=None,
        help="Project root to scan (default: current directory)",
    )
    docs_parser.add_argument(
        "--max-files", type=int, default=0,
        help="Maximum doc files to process (default: 0 = unlimited)",
    )

    # entroly share
    share_parser = subparsers.add_parser(
        "share",
        help="Generate a shareable Context Report Card for your codebase",
    )
    share_parser.add_argument(
        "--output", "-o", default="entroly-report.html",
        help="Output file path (default: entroly-report.html)",
    )

    # entroly finetune
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Export vault beliefs as JSONL training data for LLM finetuning",
    )
    finetune_parser.add_argument(
        "--output", "-o", default="training_data.jsonl",
        help="Output file path (default: training_data.jsonl)",
    )

    # entroly witness
    witness_parser = subparsers.add_parser(
        "witness",
        help="Verify and suppress unsupported factual claims",
    )
    witness_parser.add_argument(
        "--context", default=None,
        help="Evidence/context text used to verify the model output",
    )
    witness_parser.add_argument(
        "--context-file", default=None,
        help="Path to evidence/context text",
    )
    witness_parser.add_argument(
        "--output", default=None,
        help="Model output text to verify (default: stdin)",
    )
    witness_parser.add_argument(
        "--output-file", default=None,
        help="Path to model output text",
    )
    witness_parser.add_argument(
        "--mode", choices=["audit", "annotate", "strict"], default="audit",
        help="Policy to apply after verification (default: audit)",
    )
    witness_parser.add_argument(
        "--profile",
        choices=["auto", "code", "rag", "qa", "benchmark_qa", "summary", "chat", "dialogue"],
        default="auto",
        help="Workload-specific suppression profile (default: auto)",
    )
    witness_parser.add_argument(
        "--nli", action="store_true",
        help="Use OpenAI NLI when OPENAI_API_KEY is available",
    )
    witness_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit machine-readable JSON",
    )

    # entroly ravs
    ravs_parser = subparsers.add_parser(
        "ravs",
        help="RAVS v1 offline evaluation tools",
    )
    ravs_subparsers = ravs_parser.add_subparsers(dest="ravs_action")
    ravs_report_parser = ravs_subparsers.add_parser(
        "report",
        help="Generate offline evaluation report from RAVS event log",
    )
    ravs_report_parser.add_argument(
        "--log", type=str, default=None,
        help="Path to RAVS event JSONL log (default: ~/.entroly/ravs/events.jsonl)",
    )
    ravs_report_parser.add_argument(
        "--format", type=str, choices=["text", "json"], default="text",
        help="Output format: text (human-readable) or json (byte-stable)",
    )
    ravs_report_parser.add_argument(
        "--since", type=str, default=None,
        help="Only include traces since this time (e.g. 7d, 24h, or Unix timestamp)",
    )
    ravs_report_parser.add_argument(
        "--include-weak", action="store_true",
        help="Include weak (agent self-report) signals in headline metrics",
    )

    # entroly ravs capture — called by the PostToolUse hook
    ravs_capture_parser = ravs_subparsers.add_parser(
        "capture",
        help="Capture a tool outcome into the RAVS event log (called by PostToolUse hook)",
    )
    ravs_capture_parser.add_argument(
        "--stdin", action="store_true",
        help="Read Claude Code PostToolUse JSON payload from stdin",
    )
    ravs_capture_parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output (for hook usage)",
    )
    ravs_capture_parser.add_argument(
        "--command", dest="capture_command", type=str, default=None,
        help="Command string (used with --exit-code instead of --stdin)",
    )
    ravs_capture_parser.add_argument(
        "--exit-code", dest="exit_code", type=int, default=None,
        help="Exit code of the command",
    )
    ravs_capture_parser.add_argument(
        "--stdout", dest="stdout_text", type=str, default="",
        help="Last portion of stdout for verdict refinement",
    )
    ravs_capture_parser.add_argument(
        "--log", type=str, default=None,
        help="Override RAVS event log path",
    )

    # entroly ravs hook — install/manage PostToolUse hooks
    ravs_hook_parser = ravs_subparsers.add_parser(
        "hook",
        help="Install RAVS PostToolUse hooks into your IDE",
    )
    ravs_hook_subparsers = ravs_hook_parser.add_subparsers(dest="hook_action")
    ravs_hook_install = ravs_hook_subparsers.add_parser(
        "install",
        help="Install the PostToolUse hook into a supported IDE",
    )
    ravs_hook_install.add_argument(
        "--claude-code", action="store_true",
        help="Install into Claude Code (~/.claude/settings.json)",
    )
    ravs_hook_install.add_argument(
        "--dry-run", action="store_true",
        help="Print the merged settings without writing",
    )

    # ── daemon command ─────────────────────────────────────────────
    cache_parser = subparsers.add_parser(
        "cache",
        help="Inspect EGSC persistent cache (entries, hit rate, on-disk size)",
    )
    cache_subparsers = cache_parser.add_subparsers(dest="cache_action")
    cache_subparsers.add_parser("stats", help="Show persistent cache statistics")

    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Start unified supervisor (proxy + dashboard + MCP + learning)",
    )
    daemon_parser.add_argument("--proxy-port", type=int, default=9377, help="Proxy port")
    daemon_parser.add_argument("--dashboard-port", type=int, default=9378, help="Dashboard port")
    daemon_parser.add_argument("--mcp-port", type=int, default=9379, help="MCP (SSE) port")
    daemon_parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    daemon_parser.add_argument("--no-proxy", action="store_true", help="Skip proxy server")
    daemon_parser.add_argument("--no-mcp", action="store_true", help="Skip MCP server")
    daemon_parser.add_argument("--quality", type=str, default="balanced", help="Quality mode: fast/balanced/max")
    daemon_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
