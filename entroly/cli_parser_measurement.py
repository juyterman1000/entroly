"""Lifecycle, telemetry, and local measurement arguments."""

from __future__ import annotations


def _add_lifecycle_and_measurement_commands(
    subparsers,
    *,
    reason_options: tuple[tuple[str, str], ...],
    benefit_options: tuple[tuple[str, str], ...],
    surface_options: tuple[tuple[str, str], ...],
    duration_options: tuple[tuple[str, str], ...],
) -> None:
    """Register lifecycle, telemetry, and local measurement commands."""
    # entroly uninstall
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Guided Python uninstall with optional structured exit feedback",
    )
    uninstall_parser.add_argument(
        "--reason",
        choices=[value for value, _label in reason_options],
        default=None,
        help="Structured reason for uninstalling (no free text)",
    )
    uninstall_parser.add_argument(
        "--benefit",
        choices=[value for value, _label in benefit_options],
        default=None,
        help="Whether useful token reduction was observed",
    )
    uninstall_parser.add_argument(
        "--surface",
        choices=[value for value, _label in surface_options],
        default=None,
        help="Primary Entroly surface used",
    )
    uninstall_parser.add_argument(
        "--duration",
        choices=[value for value, _label in duration_options],
        default=None,
        help="Coarse use-duration bucket",
    )
    uninstall_parser.add_argument(
        "--send-feedback",
        action="store_true",
        help="Explicitly send the structured response without an interactive prompt",
    )
    uninstall_parser.add_argument(
        "--endpoint",
        default=None,
        help="HTTPS feedback collector for this one-time response",
    )
    uninstall_parser.add_argument(
        "--skip-feedback",
        action="store_true",
        help="Uninstall without collecting or sending an exit response",
    )
    uninstall_parser.add_argument(
        "--delete-remote-telemetry",
        action="store_true",
        help="Request deletion of recent linked telemetry before the one-time survey",
    )
    uninstall_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the survey payload, destination, and pip command without changing anything",
    )
    uninstall_parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="Collect/send feedback and revoke local telemetry without invoking pip",
    )
    uninstall_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Pass --yes to pip uninstall",
    )

    # entroly benchmark
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run competitive benchmark: Entroly vs Raw vs Top-K",
    )
    benchmark_parser.add_argument(
        "--budget", type=int, default=4096,
        help="Token budget per query (default: 4096)",
    )
    benchmark_parser.add_argument(
        "--compare-baseline", dest="compare_baseline", action="store_true",
        help="Print the explicit baseline comparison used for the benchmark table",
    )

    def _add_local_measure_args(p):
        p.add_argument(
            "--budget", type=int, default=4096,
            help="Token budget per query (default: 4096)",
        )
        p.add_argument(
            "--baseline", type=int, default=0,
            help="Baseline tokens/query; default is min(indexed tokens, 32000)",
        )
        p.add_argument(
            "--query", action="append", default=[],
            help="Query to simulate. Repeat for multiple queries.",
        )
        p.add_argument(
            "--json", action="store_true",
            help="Emit machine-readable JSON",
        )
        p.add_argument(
            "--max-files", type=int, default=100,
            help="Maximum files to index for this local smoke estimate (default: 100; 0 = full auto-index limit)",
        )

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Estimate local token savings without calling an LLM",
    )
    _add_local_measure_args(simulate_parser)

    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress a file with the content codecs and print a receipt",
    )
    compress_parser.add_argument("path", help="File to compress")
    compress_parser.add_argument(
        "--out", dest="out_path", default=None,
        help="Write the compressed form here (default: stdout is not used; "
             "only the receipt is printed)",
    )
    compress_parser.add_argument(
        "--store", dest="store_path", default=None,
        help="Recovery store path (default: <ENTROLY_DIR>/recovery.json)",
    )
    compress_parser.add_argument(
        "--query", default="", help="Query to condition document selection on",
    )
    compress_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit the receipt as JSON",
    )

    recover_parser = subparsers.add_parser(
        "recover",
        # Not "the exact original bytes". What a digest recovers depends on the
        # codec that produced it: `json` stores the complete original, `code`
        # stores the bodies elided from a skeleton. Both are exact for what
        # they hold, but only one is the whole file, and promising the file
        # made a partial recovery look like a corrupt one.
        help="Recover the exact bytes a recovery digest commits to "
             "(a whole file, or the parts elided from its compressed form)",
    )
    recover_parser.add_argument(
        "digest", help="Recovery digest from a compress receipt (sha256:...)",
    )
    recover_parser.add_argument(
        "--store", dest="store_path", default=None,
        help="Recovery store path (default: <ENTROLY_DIR>/recovery.json)",
    )
    recover_parser.add_argument(
        "--out", dest="out_path", default=None,
        help="Write recovered bytes here (default: stdout)",
    )

    perf_parser = subparsers.add_parser(
        "perf",
        help="Measure local optimizer savings and latency without calling an LLM",
    )
    _add_local_measure_args(perf_parser)

    value_parser = subparsers.add_parser(
        "value",
        help="Show measured provider value separately from local-only reductions",
    )
    value_parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit a machine-readable Context Value Receipt",
    )

    activation_parser = subparsers.add_parser(
        "activation",
        help="Run or inspect deterministic agent-host activation",
    )
    activation_subparsers = activation_parser.add_subparsers(
        dest="activation_action", required=True
    )
    activation_hook = activation_subparsers.add_parser(
        "hook", help="Read one host hook event from stdin and emit hook JSON"
    )
    activation_hook.add_argument(
        "--host",
        default="auto",
        choices=[
            "auto",
            "codex",
            "claude-code",
            "cursor",
            "gemini",
            "kiro",
            "vscode-copilot",
            "compatible",
        ],
    )
    activation_hook.add_argument("--budget", type=int, default=1200)
    activation_hook.add_argument("--max-files", type=int, default=200)
    activation_hook.add_argument(
        "--output-format",
        choices=["auto", "json", "context"],
        default="auto",
        help="Host output contract; auto emits context text for Kiro and JSON otherwise",
    )
    activation_status_parser = activation_subparsers.add_parser(
        "status", help="Show observed host-hook activations for this project"
    )
    activation_status_parser.add_argument("--project", default=None)
    activation_status_parser.add_argument(
        "--json", dest="json_output", action="store_true"
    )
    for action in ("install", "uninstall"):
        activation_config = activation_subparsers.add_parser(
            action, help=f"{action.capitalize()} a managed project activation hook"
        )
        activation_config.add_argument(
            "--host", choices=["cursor", "kiro"], required=True
        )
        activation_config.add_argument("--project", default=".")
        if action == "install":
            activation_config.add_argument(
                "--force",
                action="store_true",
                help="Back up an existing target before installing",
            )

    # entroly status
    status_parser = subparsers.add_parser(
        "status",
        help="Check if entroly server/proxy is running",
    )
    status_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port to check (default: 9377)",
    )
    status_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit a stable machine-readable local status report",
    )
    status_parser.add_argument(
        "--require-running", action="store_true",
        help="Exit nonzero when the local Entroly proxy is not ready",
    )

    # entroly config
    subparsers.add_parser(
        "config",
        help="Show current configuration",
    )

    # entroly telemetry
    telem_parser = subparsers.add_parser(
        "telemetry",
        help="Manage explicit-consent pseudonymous product-health telemetry",
    )
    telem_parser.add_argument(
        "action", choices=["on", "off", "status", "preview", "flush"],
        nargs="?", default="status",
        help="Manage product-health telemetry (default: status)",
    )
    telem_parser.add_argument(
        "--endpoint",
        default=None,
        help="HTTPS collector URL stored only after `telemetry on`",
    )
    telem_parser.add_argument(
        "--no-error-events",
        action="store_true",
        help="Collect adoption events but not coarse error categories",
    )
    telem_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit machine-readable status or flush output",
    )

    # entroly usage
    usage_parser = subparsers.add_parser(
        "usage",
        help="Query provider usage and spend from the usage ledger",
    )
    usage_parser.add_argument(
        "--since",
        default=None,
        help="Show events after this time (e.g. 1h, 24h, 7d, or ISO timestamp)",
    )
    usage_parser.add_argument(
        "--until",
        default=None,
        help="Show events before this time (ISO timestamp)",
    )
    usage_parser.add_argument(
        "--model", default=None,
        help="Filter by model name",
    )
    usage_parser.add_argument(
        "--provider", default=None,
        help="Filter by provider name",
    )
    usage_parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum events to list (default: 20)",
    )
    usage_parser.add_argument(
        "--csv", dest="csv_output", action="store_true",
        help="Export matching events as CSV",
    )
    usage_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit machine-readable JSON output",
    )
    usage_parser.add_argument(
        "--db",
        default=None,
        help="Path to usage ledger SQLite file",
    )

    # entroly clean
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clear cached state (checkpoints, index, pull cache)",
    )
    clean_parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )

    # entroly export
    export_parser = subparsers.add_parser(
        "export",
        help="Export learned state for sharing with teammates",
    )
    export_parser.add_argument(
        "output_path", nargs="?", default=None,
        help="Positional output path (alternative to --output).",
    )
    export_parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path (default: entroly_export.json).",
    )

    # entroly import
    import_parser = subparsers.add_parser(
        "import",
        help="Import shared learned state from an export file",
    )
    import_parser.add_argument(
        "file", type=str,
        help="Path to entroly_export.json",
    )

    # entroly drift
    subparsers.add_parser(
        "drift",
        help="Detect weight drift / staleness in learned configuration",
    )

    # entroly profile
    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage per-project weight profiles",
    )
    profile_parser.add_argument(
        "profile_action", choices=["save", "load", "list"],
        help="Save current config as profile, load a profile, or list profiles",
    )
    profile_parser.add_argument(
        "name", nargs="?", default=None,
        help="Profile name (defaults to project hash for 'save')",
    )

    # entroly batch
    batch_parser = subparsers.add_parser(
        "batch",
        help="Headless/CI mode: optimize batch queries from stdin or file",
    )
    batch_parser.add_argument(
        "-i", "--input", type=str, default="-",
        help="Input file with one query per line (default: stdin)",
    )
    batch_parser.add_argument(
        "--budget", type=int, default=128000,
        help="Token budget per query (default: 128000)",
    )
    batch_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output results as JSON (for CI pipelines)",
    )
    batch_parser.add_argument(
        "--fail-over-budget", dest="fail_over_budget", action="store_true",
        help="Exit with non-zero status if any query's optimized tokens exceed --budget (for CI gating)",
    )

    # entroly demo (bounded local measurement)
    demo_parser = subparsers.add_parser(
        "demo",
        help="Quick-win demo: before/after comparison showing token savings",
    )
    _add_local_measure_args(demo_parser)
