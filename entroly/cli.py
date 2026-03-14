"""
Entroly CLI — Zero-friction onboarding for AI coding agents.

Commands:
    entroly init     Auto-detect project + AI tool, generate MCP config
    entroly serve    Start MCP server with auto-indexing
    entroly dashboard   Show live value metrics
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


# ── ANSI colors ──
class C:
    BOLD = "\033[1m"
    GREEN = "\033[38;5;82m"
    CYAN = "\033[38;5;45m"
    YELLOW = "\033[38;5;220m"
    RED = "\033[38;5;196m"
    GRAY = "\033[38;5;240m"
    RESET = "\033[0m"


def _detect_project_type() -> dict:
    """Detect what kind of project this is."""
    cwd = os.getcwd()
    indicators = {
        "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
        "rust": ["Cargo.toml"],
        "javascript": ["package.json"],
        "typescript": ["tsconfig.json"],
        "go": ["go.mod"],
        "java": ["pom.xml", "build.gradle"],
        "ruby": ["Gemfile"],
    }

    detected = []
    for lang, files in indicators.items():
        for f in files:
            if os.path.exists(os.path.join(cwd, f)):
                detected.append(lang)
                break

    project_name = os.path.basename(cwd)
    return {
        "name": project_name,
        "languages": detected or ["unknown"],
        "primary": detected[0] if detected else "unknown",
    }


def _detect_ai_tool() -> dict:
    """Detect which AI coding tool is installed."""
    cwd = os.getcwd()
    tools = []

    # Cursor
    if os.path.exists(os.path.join(cwd, ".cursor")):
        tools.append({
            "name": "Cursor",
            "config_path": os.path.join(cwd, ".cursor", "mcp.json"),
            "config_key": "mcpServers",
        })

    # VS Code (Copilot, Cline, etc.)
    if os.path.exists(os.path.join(cwd, ".vscode")):
        tools.append({
            "name": "VS Code",
            "config_path": os.path.join(cwd, ".vscode", "mcp.json"),
            "config_key": "mcpServers",
        })

    # Windsurf
    if os.path.exists(os.path.join(cwd, ".windsurf")):
        tools.append({
            "name": "Windsurf",
            "config_path": os.path.join(cwd, ".windsurf", "mcp.json"),
            "config_key": "mcpServers",
        })

    # Claude Desktop (global config) — only add if no project-local tool found
    # Avoids overwriting global Claude config when user only uses Cursor/VS Code
    if not tools:
        system = platform.system()
        if system == "Darwin":
            claude_cfg = os.path.expanduser(
                "~/Library/Application Support/Claude/claude_desktop_config.json"
            )
        elif system == "Windows":
            claude_cfg = os.path.join(
                os.environ.get("APPDATA", ""), "Claude", "claude_desktop_config.json"
            )
        else:
            claude_cfg = os.path.expanduser("~/.config/claude/claude_desktop_config.json")

        tools.append({
            "name": "Claude Desktop",
            "config_path": claude_cfg,
            "config_key": "mcpServers",
        })

    return {"tools": tools, "primary": tools[0] if tools else None}


def _generate_mcp_config() -> dict:
    """Generate the MCP server config for Entroly."""
    return {
        "entroly": {
            "command": "entroly",
            "args": ["serve"],
            "env": {},
        }
    }


def _write_config(tool: dict, dry_run: bool = False) -> str:
    """Write MCP config for a specific tool."""
    config_path = tool["config_path"]
    config_key = tool["config_key"]
    entroly_config = _generate_mcp_config()

    # Read existing config if it exists
    existing = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Merge entroly config into existing
    if config_key not in existing:
        existing[config_key] = {}
    existing[config_key].update(entroly_config)

    if dry_run:
        return json.dumps(existing, indent=2)

    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")

    return config_path


def cmd_init(args):
    """entroly init — auto-detect and configure."""
    print(f"""
{C.CYAN}{C.BOLD}  🔬 Entroly — Context Optimizer for AI Coding Agents{C.RESET}
""")

    # Detect project
    project = _detect_project_type()
    print(f"  {C.GRAY}Project:{C.RESET}  {C.BOLD}{project['name']}{C.RESET} ({', '.join(project['languages'])})")

    # Detect AI tools
    tools = _detect_ai_tool()

    if not tools["tools"]:
        print(f"\n  {C.YELLOW}⚠ No AI tool detected.{C.RESET}")
        print(f"  {C.GRAY}Create a .cursor/, .vscode/, or .windsurf/ directory first.{C.RESET}")
        return

    # Show detected tools
    for tool in tools["tools"]:
        exists = os.path.exists(tool["config_path"])
        status = f"{C.GRAY}(config exists){C.RESET}" if exists else ""
        print(f"  {C.GRAY}AI Tool:{C.RESET}  {C.BOLD}{tool['name']}{C.RESET} {status}")

    print()

    # Write configs
    for tool in tools["tools"]:
        if args.dry_run:
            config = _write_config(tool, dry_run=True)
            print(f"  {C.GRAY}Would write to {tool['config_path']}:{C.RESET}")
            print(f"  {config}")
        else:
            path = _write_config(tool)
            print(f"  {C.GREEN}✅ Generated{C.RESET} {path}")

    # Count indexable files
    from entroly.auto_index import _git_ls_files, _should_index
    files = _git_ls_files(os.getcwd())
    indexable = [f for f in files if _should_index(f)]
    print(f"  {C.GREEN}✅ Entroly will auto-index {len(indexable)} files on first run{C.RESET}")

    print(f"""
  {C.BOLD}Next:{C.RESET} Restart your AI tool. Entroly is now active.
  {C.GRAY}The MCP server auto-indexes your codebase on startup.{C.RESET}
  {C.GRAY}Call {C.CYAN}entroly_dashboard{C.GRAY} from your AI to see live value metrics.{C.RESET}
""")


def cmd_serve(args):
    """entroly serve — start MCP server with auto-indexing."""
    # Set env so Docker launcher knows to go native
    os.environ["ENTROLY_NO_DOCKER"] = "1"

    # Import and run
    from entroly.server import main
    main()


def cmd_dashboard(args):
    """entroly dashboard — show value metrics from current session."""
    from entroly.server import EntrolyEngine
    from entroly.auto_index import auto_index

    print(f"\n{C.CYAN}{C.BOLD}  🔬 Entroly Dashboard{C.RESET}\n")

    engine = EntrolyEngine()

    # Auto-index current project
    result = auto_index(engine, force=args.force)

    if result["status"] == "indexed":
        print(f"  {C.GREEN}Indexed {result['files_indexed']} files ({result['total_tokens']:,} tokens) in {result['duration_s']}s{C.RESET}")
    elif result["status"] == "skipped":
        print(f"  {C.GRAY}Using persistent index ({result['existing_fragments']} fragments){C.RESET}")

    # Run an optimize to build stats
    engine.optimize_context(token_budget=128000, query="project overview")

    # Get stats
    stats = engine.get_stats()
    perf = stats.get("performance", {})
    mem = stats.get("memory", {})
    savings = stats.get("savings", {})
    session = stats.get("session", {})

    print(f"""
  {C.BOLD}💰 Cost Analysis{C.RESET}
     Naive cost/call:     ${mem.get('naive_cost_per_call_usd', 0):.4f}
     Optimized cost/call: ${mem.get('optimized_cost_per_call_usd', 0):.4f}
     {C.GREEN}Savings:             {((1 - mem.get('optimized_cost_per_call_usd', 0) / max(mem.get('naive_cost_per_call_usd', 0), 0.0001)) * 100):.0f}% per API call{C.RESET}

  {C.BOLD}⚡ Performance{C.RESET}
     Optimize latency:    {perf.get('avg_optimize_us', 0):.0f}µs ({perf.get('avg_optimize_us', 0)/1000:.2f}ms)
     Context compression: {perf.get('context_compression', 0):.1%}

  {C.BOLD}🧠 Codebase{C.RESET}
     Fragments tracked:   {session.get('total_fragments', 0)}
     Total tokens:        {session.get('total_tokens_tracked', 0):,}
     Memory footprint:    {mem.get('total_kb', 0)} KB
     Avg entropy:         {session.get('avg_entropy', 0):.4f}
""")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="entroly",
        description="🔬 Entroly — Information-theoretic context optimization for AI coding agents",
    )
    subparsers = parser.add_subparsers(dest="command")

    # entroly init
    init_parser = subparsers.add_parser(
        "init",
        help="Auto-detect project + AI tool, generate MCP config",
    )
    init_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be generated without writing files",
    )

    # entroly serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the MCP server with auto-indexing",
    )

    # entroly dashboard
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Show live value metrics for current project",
    )
    dash_parser.add_argument(
        "--force", action="store_true",
        help="Force re-index even if persistent index exists",
    )

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        # Default: if no subcommand, run serve (backward compat)
        cmd_serve(args)


if __name__ == "__main__":
    main()
