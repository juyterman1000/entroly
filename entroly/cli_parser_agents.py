"""Agent wrapper, experiment, and governance arguments."""

from __future__ import annotations

import argparse
from collections.abc import Callable

def _add_agent_and_governance_commands(
    subparsers, wrap_agent_names: Callable[[], str]
) -> None:
    """Register agent wrappers, experiments, and governance commands."""
    # entroly wrap
    wrap_parser = subparsers.add_parser(
        "wrap",
        help="Start proxy, configure MCP, or print setup for a coding agent",
    )
    wrap_parser.add_argument(
        "agent", type=str, nargs="?",
        help=f"Agent to wrap: {wrap_agent_names()}",
    )
    wrap_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: 9377)",
    )
    wrap_parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview actions (config writes / proxy start / agent launch) without performing them",
    )
    wrap_parser.add_argument(
        "agent_args", nargs=argparse.REMAINDER,
        help="Additional arguments passed to the agent",
    )

    # Explicit two-arm experiments. Each invocation records one arm so Entroly
    # never runs a stateful/costly agent task twice without separate consent.
    trial_parser = subparsers.add_parser(
        "trial",
        help="Record one matched baseline/optimized agent run or report an experiment",
    )
    trial_parser.add_argument("--experiment", default=None, help="Stable experiment id")
    trial_parser.add_argument(
        "--arm", choices=["baseline", "optimized"], default=None,
        help="Baseline bypasses selection; optimized enables Entroly",
    )
    trial_parser.add_argument("--report", default=None, help="Report an existing experiment id")
    trial_parser.add_argument("--evaluation", default=None, help="External JSON quality evaluation")
    trial_parser.add_argument("--port", type=int, default=9377, help="Entroly proxy port")
    trial_parser.add_argument("--receipt", default=None, help="Override the run receipt path")
    trial_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Emit JSON"
    )
    trial_parser.add_argument(
        "agent_command", nargs=argparse.REMAINDER,
        help="Agent command after --, for example: -- codex exec 'fix the test'",
    )

    shrink_parser = subparsers.add_parser(
        "shrink",
        help="Run a command through a bounded, exactly recoverable output envelope",
    )
    shrink_parser.add_argument("--budget", type=int, default=1200, help="Per-stream token budget")
    shrink_parser.add_argument(
        "--max-bytes", type=int, default=64 * 1024 * 1024,
        help="Per-stream compression cap; larger streams pass through",
    )
    shrink_parser.add_argument("--store", dest="store_path", default=None, help="Recovery store path")
    shrink_parser.add_argument("--receipt", default=None, help="Run receipt path")
    shrink_parser.add_argument(
        "command_args", nargs=argparse.REMAINDER, help="Command after --"
    )

    browser_parser = subparsers.add_parser(
        "browser",
        help="Capture and compress a recoverable accessibility evidence envelope",
    )
    browser_parser.add_argument("url", nargs="?", help="HTTP(S) page URL")
    browser_parser.add_argument("--snapshot", default=None, help="Existing ARIA snapshot path")
    browser_parser.add_argument("--query", default="", help="Evidence query used for selection")
    browser_parser.add_argument("--budget", type=int, default=2000, help="Active token budget")
    browser_parser.add_argument("--timeout", type=float, default=30.0, help="Navigation timeout seconds")
    browser_parser.add_argument(
        "--max-bytes", type=int, default=16 * 1024 * 1024, help="Maximum snapshot file size"
    )
    browser_parser.add_argument(
        "--allow-private-network", action="store_true",
        help="Permit loopback/private/reserved targets for explicit local testing",
    )
    browser_parser.add_argument("--store", dest="store_path", default=None, help="Recovery store path")
    browser_parser.add_argument("--receipt", default=None, help="Receipt output path")
    browser_parser.add_argument(
        "--semantic-model", default=None,
        help="Existing local sentence-transformer directory; remote downloads are refused",
    )
    browser_parser.add_argument(
        "--threshold", type=float, default=None,
        help="Optional ranker score floor (a ranking signal, not proof)",
    )
    browser_parser.add_argument("--calibration-id", default=None)
    browser_parser.add_argument("--max-matches", type=int, default=8)
    browser_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Emit context and receipt as JSON"
    )

    find_parser = subparsers.add_parser(
        "find",
        help="Locate exact source evidence for a natural-language query",
    )
    find_parser.add_argument("source", nargs="?", help="UTF-8 source file; omit to read stdin")
    find_parser.add_argument("--query", "-q", required=True, help="Evidence to locate")
    find_parser.add_argument("--source-id", default=None, help="Provenance label in the receipt")
    find_parser.add_argument("--budget", type=int, default=2000, help="Selected token budget")
    find_parser.add_argument("--max-matches", type=int, default=5)
    find_parser.add_argument("--threshold", type=float, default=None)
    find_parser.add_argument("--calibration-id", default=None)
    find_parser.add_argument(
        "--semantic-model", default=None,
        help="Existing local sentence-transformer directory; remote downloads are refused",
    )
    find_parser.add_argument(
        "--passage-mode", choices=["auto", "paragraph", "line"], default="auto"
    )
    find_parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    find_parser.add_argument("--store", dest="store_path", default=None)
    find_parser.add_argument("--receipt", default=None)
    find_parser.add_argument("--json", dest="json_output", action="store_true")

    # ── Governance control plane ──────────────────────────────────────
    # The `entroly/governance/` package shipped with tests but no entry point,
    # so the repository's own reachability check listed all seven modules as
    # unreachable. This is its product path.
    govern_parser = subparsers.add_parser(
        "govern", help="Inspect the agent governance control plane"
    )
    govern_groups = govern_parser.add_subparsers(dest="govern_group", required=True)

    govern_identity = govern_groups.add_parser("identity", help="Agent identity")
    identity_actions = govern_identity.add_subparsers(dest="identity_action", required=True)
    identity_actions.add_parser("show", help="Resolve the current agent identity")
    identity_create = identity_actions.add_parser("create", help="Mint an agent identity")
    identity_create.add_argument("--agent-id", required=True)
    identity_create.add_argument("--agent-type", default="unknown")
    identity_create.add_argument("--organization", default="")
    identity_create.add_argument("--user", default="")
    identity_create.add_argument("--model", default="")
    identity_create.add_argument("--scope", default="", help="Comma-separated scopes")

    govern_policy = govern_groups.add_parser("policy", help="Policy evaluation")
    policy_actions = govern_policy.add_subparsers(dest="policy_action", required=True)
    policy_list = policy_actions.add_parser("list", help="List loaded policies")
    policy_check = policy_actions.add_parser("check", help="Evaluate one authorization")
    policy_check.add_argument("scope", help="Scope being requested, e.g. tool:write")
    policy_check.add_argument("--resource", default="")
    policy_check.add_argument("--risk", default="low")
    # On both actions: `list` is where an operator inspects a candidate file
    # before checking against it. Registering it only on `check` made the
    # obvious first command fail with "unrecognized arguments".
    for _policy_leaf in (policy_list, policy_check):
        _policy_leaf.add_argument("--policy-file", default=None)

    govern_audit = govern_groups.add_parser("audit", help="Governance audit log")
    audit_actions = govern_audit.add_subparsers(dest="audit_action", required=True)
    audit_tail = audit_actions.add_parser("tail", help="Show recent audit records")
    audit_tail.add_argument("--limit", type=int, default=20)
    audit_actions.add_parser("verify", help="Verify the audit hash chain")

    govern_status = govern_groups.add_parser("status", help="Control-plane status")

    # `--json` on every leaf, not just the groups: argparse only accepts a
    # parent's flag *before* the subcommand, so `govern status --json` would
    # otherwise fail with "unrecognized arguments" — the order an operator
    # actually types.
    for _govern_leaf in (
        identity_actions.choices["show"],
        identity_create,
        policy_actions.choices["list"],
        policy_check,
        audit_tail,
        audit_actions.choices["verify"],
        govern_status,
    ):
        _govern_leaf.add_argument("--json", dest="json_output", action="store_true")

    response_parser = subparsers.add_parser(
        "response", help="Manage reversible response contracts for agent bundles"
    )
    response_subparsers = response_parser.add_subparsers(dest="response_action", required=True)
    for response_action in ("list", "show", "disable"):
        response_action_parser = response_subparsers.add_parser(response_action)
        response_action_parser.add_argument(
            "--scope", choices=["project", "user"], default="project"
        )
        response_action_parser.add_argument(
            "--json", dest="json_output", action="store_true"
        )
    response_set = response_subparsers.add_parser("set")
    response_set.add_argument("name", choices=["concise", "minimal", "evidence", "off"])
    response_set.add_argument("--scope", choices=["project", "user"], default="project")
    response_set.add_argument("--json", dest="json_output", action="store_true")

    # entroly unwrap
    unwrap_parser = subparsers.add_parser(
        "unwrap",
        help="Remove Entroly's persistent integration without touching other tools",
    )
    unwrap_parser.add_argument(
        "agent", type=str, nargs="?",
        help=f"Agent to unwrap: {wrap_agent_names()}",
    )
    unwrap_parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview the resulting config without writing files",
    )

    # entroly learn
    learn_parser = subparsers.add_parser(
        "learn",
        help="Analyze session for failure patterns, write corrections",
    )
    learn_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port to read feedback from (default: 9377)",
    )
    learn_parser.add_argument(
        "--apply", action="store_true",
        help="Write learnings to CLAUDE.md / AGENTS.md",
    )
    learn_parser.add_argument(
        "--history", action="store_true",
        help="Audit local agent histories without emitting their content",
    )
    learn_parser.add_argument(
        "--history-root", action="append", default=None,
        help="Explicit history root (repeatable; overrides known defaults)",
    )
    learn_parser.add_argument("--max-files", type=int, default=200)
    learn_parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    learn_parser.add_argument("--max-file-bytes", type=int, default=8 * 1024 * 1024)
    learn_parser.add_argument("--json", dest="json_output", action="store_true")
    learn_parser.add_argument(
        "--deep", action="store_true",
        help="Deep failure mining: vault, PRISM, evolution, checkpoints",
    )
    learn_parser.add_argument(
        "--min-occurrences", type=int, default=2, dest="min_occurrences",
        help="Minimum occurrences for a pattern to surface (default: 2)",
    )

    # entroly capabilities
    capabilities_parser = subparsers.add_parser(
        "capabilities",
        help="Report installed runtime capabilities without network calls",
    )
    capabilities_parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit a stable machine-readable capability report",
    )

    # entroly hook
    hook_parser = subparsers.add_parser(
        "hook",
        help="Manage shell hook for transparent CLI output compression",
    )
    hook_sub = hook_parser.add_subparsers(dest="hook_action")
    hook_install = hook_sub.add_parser("install", help="Install shell hook")
    hook_install.add_argument("--shell", choices=["bash", "zsh", "fish"], help="Target shell")
    hook_uninstall = hook_sub.add_parser("uninstall", help="Remove shell hook")
    hook_uninstall.add_argument("--shell", choices=["bash", "zsh", "fish"], help="Target shell")
    hook_sub.add_parser("status", help="Show hook installation status")
