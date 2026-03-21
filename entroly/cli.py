"""
Entroly CLI — Zero-friction onboarding for AI coding agents.

Commands:
    entroly init        Auto-detect project + AI tool, generate MCP config
    entroly serve       Start MCP server with auto-indexing
    entroly proxy       Start invisible prompt compiler proxy
    entroly dashboard   Show live value metrics
    entroly health      Analyze codebase health (A-F grade)
    entroly autotune    Optimize hyperparameters
    entroly benchmark   Run competitive comparison
    entroly status      Check if server/proxy is running
    entroly config      Show current configuration
    entroly clean       Clear cached state (checkpoints, index, pull cache)
    entroly telemetry   Manage anonymous usage statistics (opt-in)
    entroly demo        Before/after demo showing token savings
    entroly doctor      Diagnose common issues
    entroly digest      Weekly summary of value delivered
    entroly migrate     Auto-migrate config/index to current version
    entroly role        Role-based weight presets (frontend/backend/sre/data)
    entroly completions Generate shell completion scripts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from entroly import __version__
except ImportError:
    __version__ = "0.5.0"


# ── ANSI colors ──
class C:
    BOLD = "\033[1m"
    GREEN = "\033[38;5;82m"
    CYAN = "\033[38;5;45m"
    YELLOW = "\033[38;5;220m"
    RED = "\033[38;5;196m"
    GRAY = "\033[38;5;240m"
    RESET = "\033[0m"


_ENTROLY_DIR = Path.home() / ".entroly"
_FIRST_RUN_MARKER = _ENTROLY_DIR / ".welcome_shown"


def _check_first_run() -> None:
    """Show a one-time welcome message on first ever invocation.

    Creates ~/.entroly/.welcome_shown as a marker so it only fires once.
    """
    if _FIRST_RUN_MARKER.exists():
        return
    _ENTROLY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"""
{C.CYAN}{C.BOLD}  Welcome to Entroly{C.RESET} — information-theoretic context optimization
  for AI coding agents.

  {C.BOLD}Get started in 60 seconds:{C.RESET}

    {C.BOLD}Step 1:{C.RESET} {C.CYAN}entroly init{C.RESET}       Auto-detect your IDE and generate MCP config
    {C.BOLD}Step 2:{C.RESET} {C.CYAN}entroly proxy{C.RESET}      Start the invisible prompt compiler proxy
    {C.BOLD}Step 3:{C.RESET} Point your IDE's API base URL to {C.CYAN}http://localhost:9377{C.RESET}

  {C.BOLD}See entroly in action:{C.RESET}
    {C.CYAN}entroly demo{C.RESET}        Run a before/after comparison showing token savings

  {C.BOLD}Useful commands:{C.RESET}
    {C.CYAN}entroly status{C.RESET}      Check if server/proxy is running
    {C.CYAN}entroly doctor{C.RESET}      Diagnose common issues
    {C.CYAN}entroly health{C.RESET}      Analyze codebase health (grade A-F)

  {C.GRAY}Documentation: https://github.com/juyterman1000/entroly{C.RESET}
  {C.GRAY}This message appears once. Run entroly --help anytime.{C.RESET}
""")
    try:
        _FIRST_RUN_MARKER.write_text("1")
    except OSError:
        pass


def _check_for_update() -> None:
    """Check PyPI for a newer version (non-blocking, cached for 24h).

    Prints a one-line notice if a newer version exists. Fails silently
    on network errors — never blocks CLI startup.
    """
    cache_file = _ENTROLY_DIR / ".update_check"
    now = __import__("time").time()

    # Only check once per 24 hours
    try:
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            if now - data.get("ts", 0) < 86400:
                if data.get("newer"):
                    print(
                        f"  {C.YELLOW}Update available:{C.RESET} "
                        f"{__version__} -> {data['newer']}  "
                        f"{C.GRAY}(pip install --upgrade entroly){C.RESET}"
                    )
                return
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    # Non-blocking check in a background thread
    import threading

    def _do_check():
        try:
            import urllib.request
            resp = urllib.request.urlopen(
                "https://pypi.org/pypi/entroly/json", timeout=3
            )
            pypi = json.loads(resp.read())
            latest = pypi.get("info", {}).get("version", __version__)

            # Simple version comparison (works for semver)
            newer = None
            if latest != __version__:
                from packaging.version import Version
                try:
                    if Version(latest) > Version(__version__):
                        newer = latest
                except Exception:
                    # packaging not installed — fall back to string compare
                    if latest > __version__:
                        newer = latest

            _ENTROLY_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"ts": now, "newer": newer}))

            if newer:
                print(
                    f"  {C.YELLOW}Update available:{C.RESET} "
                    f"{__version__} -> {newer}  "
                    f"{C.GRAY}(pip install --upgrade entroly){C.RESET}"
                )
        except Exception:
            pass  # Never block or error on update checks

    t = threading.Thread(target=_do_check, daemon=True)
    t.start()


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
            appdata = os.environ.get("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
            claude_cfg = os.path.join(appdata, "Claude", "claude_desktop_config.json")
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
{C.CYAN}{C.BOLD}  Entroly — Context Optimizer for AI Coding Agents{C.RESET}
""")

    # Detect project
    project = _detect_project_type()
    print(f"  {C.GRAY}Project:{C.RESET}  {C.BOLD}{project['name']}{C.RESET} ({', '.join(project['languages'])})")

    # Detect AI tools
    tools = _detect_ai_tool()

    if not tools["tools"]:
        print(f"\n  {C.YELLOW}No AI tool detected.{C.RESET}")
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
            print(f"  {C.GREEN}Generated{C.RESET} {path}")

    # Count indexable files
    from entroly.auto_index import _git_ls_files, _should_index
    files = _git_ls_files(os.getcwd())
    indexable = [f for f in files if _should_index(f)]
    print(f"  {C.GREEN}Entroly will auto-index {len(indexable)} files on first run{C.RESET}")

    print(f"""
  {C.BOLD}Next:{C.RESET} Restart your AI tool. Entroly is now active.
  {C.GRAY}The MCP server auto-indexes your codebase on startup.{C.RESET}
  {C.GRAY}Call {C.CYAN}entroly_dashboard{C.GRAY} from your AI to see live value metrics.{C.RESET}
""")


def cmd_serve(args):
    """entroly serve — start MCP server with auto-indexing."""
    # Set env so Docker launcher knows to go native
    os.environ["ENTROLY_NO_DOCKER"] = "1"

    if getattr(args, "debug", False):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [entroly:%(name)s] %(levelname)s %(message)s",
            stream=sys.stderr,
            force=True,
        )

    # Import and run
    from entroly.server import main
    main()


def cmd_dashboard(args):
    """entroly dashboard — launch live web dashboard at localhost:9378."""
    from entroly.server import EntrolyEngine
    from entroly.auto_index import auto_index
    from entroly.dashboard import start_dashboard

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Value Dashboard{C.RESET}\n")

    engine = EntrolyEngine()
    result = auto_index(engine, force=args.force)

    if result["status"] == "indexed":
        print(f"  {C.GREEN}Indexed {result['files_indexed']} files ({result['total_tokens']:,} tokens) in {result['duration_s']}s{C.RESET}")
    elif result["status"] == "skipped":
        print(f"  {C.GRAY}Using persistent index ({result['existing_fragments']} fragments){C.RESET}")

    # Run an optimize to populate all engine subsystems
    engine.optimize_context(token_budget=128000, query="project overview")

    # Start web dashboard
    start_dashboard(engine=engine, port=args.port, daemon=False)
    print(f"\n  {C.GREEN}{C.BOLD}Dashboard live at http://localhost:{args.port}{C.RESET}")
    print(f"  {C.GRAY}Showing: tokens saved, PRISM weights, health grade, SAST, dep graph, knapsack decisions{C.RESET}")
    print(f"  {C.GRAY}Press Ctrl+C to stop{C.RESET}\n")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n  {C.GRAY}Dashboard stopped.{C.RESET}")


def cmd_health(args):
    """entroly health — analyze codebase health."""
    import json as _json
    from entroly.server import EntrolyEngine
    from entroly.auto_index import auto_index

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Health Analysis{C.RESET}\n")

    engine = EntrolyEngine()
    result = auto_index(engine)

    if result["status"] == "indexed":
        print(f"  {C.GREEN}Indexed {result['files_indexed']} files ({result['total_tokens']:,} tokens){C.RESET}")

    # Run optimize to build dep graph
    engine.optimize_context(token_budget=128000, query="")

    # Get health report
    if engine._use_rust:
        health = _json.loads(engine._rust.analyze_health())
        grade = health.get("health_grade", "?")
        score = health.get("code_health_score", 0)

        grade_colors = {"A": C.GREEN, "B": C.GREEN, "C": C.YELLOW, "D": C.RED, "F": C.RED}
        gc = grade_colors.get(grade, C.GRAY)

        print(f"\n  {C.BOLD}Code Health:{C.RESET}  {gc}{C.BOLD}{grade}{C.RESET} ({score}/100)\n")

        items = [
            ("Clone pairs", health.get("clone_pairs", [])),
            ("Dead symbols", health.get("dead_symbols", [])),
            ("God files", health.get("god_files", [])),
            ("Arch violations", health.get("arch_violations", [])),
            ("Naming issues", health.get("naming_issues", [])),
        ]
        for name, lst in items:
            count = len(lst) if isinstance(lst, list) else 0
            color = C.GREEN if count == 0 else C.YELLOW if count < 5 else C.RED
            sym = "+" if count == 0 else "!"
            print(f"  {color}{sym} {name}: {count}{C.RESET}")
            if count > 0 and args.verbose:
                for item in lst[:5]:
                    detail = item if isinstance(item, str) else str(item)
                    print(f"    {C.GRAY}-> {detail[:80]}{C.RESET}")

        rec = health.get("top_recommendation")
        if rec:
            print(f"\n  {C.YELLOW}{rec}{C.RESET}")

        # Security summary
        sec = _json.loads(engine._rust.security_report())
        total_findings = sec.get("critical_total", 0) + sec.get("high_total", 0)
        if total_findings > 0:
            print(f"\n  {C.RED}{total_findings} security findings ({sec.get('critical_total', 0)} critical, {sec.get('high_total', 0)} high){C.RESET}")
            if sec.get("most_vulnerable_fragment"):
                print(f"    {C.GRAY}Most vulnerable: {sec['most_vulnerable_fragment']}{C.RESET}")
        else:
            print(f"\n  {C.GREEN}No security vulnerabilities detected{C.RESET}")

    print()


def cmd_autotune(args):
    """entroly autotune — optimize engine hyperparameters."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

    # Handle --rollback
    if getattr(args, "rollback", False):
        print(f"\n{C.CYAN}{C.BOLD}  Entroly Autotune — Rollback{C.RESET}\n")
        try:
            from bench.autotune import rollback_config
            config_path = Path(os.path.dirname(__file__)).parent / "tuning_config.json"
            result = rollback_config(config_path)
            if result["status"] == "no_backup_found":
                print(f"  {C.RED}No backup found — nothing to roll back.{C.RESET}\n")
                return
            print(f"  {C.GREEN}Restored from:{C.RESET} {result['restored_from']}")
            print(f"  {C.GREEN}{C.BOLD}Rollback complete.{C.RESET} Previous config is now active.\n")
        except ImportError:
            print(f"  {C.RED}bench.autotune not available{C.RESET}")
        return

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Autotune{C.RESET}\n")
    print(f"  {C.GRAY}Running {args.iterations} iterations of mutation-based optimization...{C.RESET}\n")

    try:
        from bench.autotune import autotune
        result = autotune(iterations=args.iterations)
        print(f"\n  {C.GREEN}{C.BOLD}Best score: {result.get('final_score', 0):.4f}{C.RESET}")
        print(f"  {C.GRAY}Config saved to tuning_config.json{C.RESET}")
        print(f"  {C.GRAY}To undo: entroly autotune --rollback{C.RESET}\n")
    except ImportError:
        # Fallback: run the script directly
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), '..', 'bench', 'autotune.py'), '--iterations', str(args.iterations)])


def _check_upstream(config) -> None:
    """Quick connectivity test against upstream LLM APIs.

    Sends a HEAD request with a 3s timeout to catch DNS/firewall/VPN issues
    before the first real request hangs for 120s. Warns but doesn't block.
    """
    import urllib.request
    endpoints = [
        ("OpenAI", config.openai_base_url),
        ("Anthropic", config.anthropic_base_url),
    ]
    for name, base_url in endpoints:
        try:
            req = urllib.request.Request(base_url, method="HEAD")
            urllib.request.urlopen(req, timeout=3)
            print(f"  {C.GREEN}[OK]{C.RESET} {name} API reachable")
        except Exception:
            print(
                f"  {C.YELLOW}[!!]{C.RESET} {name} API unreachable ({base_url})\n"
                f"       {C.GRAY}Requests will still be forwarded — "
                f"the circuit breaker will handle failures.{C.RESET}"
            )


def cmd_proxy(args):
    """entroly proxy — start the invisible prompt compiler proxy."""
    if getattr(args, "debug", False):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [entroly:%(name)s] %(levelname)s %(message)s",
            stream=sys.stderr,
            force=True,
        )

    # Gap #28: --bypass flag sets env var consumed by proxy
    if getattr(args, "bypass", False):
        os.environ["ENTROLY_BYPASS"] = "1"

    print(f"""
{C.CYAN}{C.BOLD}  Entroly Prompt Compiler Proxy{C.RESET}
{C.GRAY}  Invisible intelligence layer for any AI coding tool{C.RESET}
""")

    from entroly.server import EntrolyEngine
    from entroly.auto_index import auto_index, start_incremental_watcher
    from entroly.proxy import create_proxy_app
    from entroly.proxy_config import ProxyConfig, resolve_quality

    # Load config from environment
    config = ProxyConfig.from_env()
    if args.port:
        config.port = args.port
    if args.host:
        config.host = args.host
    if args.quality is not None:
        quality_val = resolve_quality(args.quality)
        config.quality = quality_val
        config._apply_quality_dial(quality_val)

    # Initialize engine + auto-index codebase (non-blocking for large repos)
    engine = EntrolyEngine()

    import threading

    _index_ready = threading.Event()
    _index_result = {}

    def _bg_index():
        nonlocal _index_result
        _index_result = auto_index(engine, force=args.force)
        _index_ready.set()

    idx_thread = threading.Thread(target=_bg_index, daemon=True, name="entroly-autoindex")
    idx_thread.start()

    # Wait up to 5s for auto-index. If it takes longer, proceed — the proxy
    # is usable immediately (it just won't have context yet).
    if _index_ready.wait(timeout=5.0):
        result = _index_result
        if result["status"] == "indexed":
            print(f"  {C.GREEN}Indexed {result['files_indexed']} files ({result['total_tokens']:,} tokens) in {result['duration_s']}s{C.RESET}")
        elif result["status"] == "skipped":
            print(f"  {C.GRAY}Using persistent index ({result['existing_fragments']} fragments){C.RESET}")
    else:
        print(f"  {C.YELLOW}Auto-indexing in progress...{C.RESET} Proxy starting now (context available shortly)")

    # Start incremental file watcher so new/modified files are picked up
    start_incremental_watcher(engine)

    # Run a warm-up optimize to populate all engine subsystems
    engine.optimize_context(token_budget=128000, query="project overview")

    # Upstream connectivity check — fast-fail if the LLM API is unreachable
    _check_upstream(config)

    # Create the ASGI app (this also starts the dashboard on :9378)
    app = create_proxy_app(engine, config)

    print(f"""
  {C.GREEN}{C.BOLD}Proxy live at http://{config.host}:{config.port}{C.RESET}
  {C.GREEN}{C.BOLD}Dashboard at http://localhost:9378{C.RESET}

  {C.BOLD}To use:{C.RESET} Set your AI tool's API base URL to:
    {C.CYAN}http://localhost:{config.port}/v1{C.RESET}

  {C.GRAY}Every LLM request is intercepted -> optimized (ECC + EGTC + IOS) -> forwarded.{C.RESET}
  {C.GRAY}< 10ms overhead. Press Ctrl+C to stop.{C.RESET}
  {C.GRAY}File watcher active — new/modified files auto-indexed every 120s.{C.RESET}
""")

    try:
        import uvicorn
        uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    except ImportError:
        print(f"  {C.RED}uvicorn not installed. Install with: pip install uvicorn{C.RESET}")
        print(f"  {C.GRAY}Or run directly: uvicorn entroly.proxy:app --port {config.port}{C.RESET}")
    except KeyboardInterrupt:
        print(f"\n  {C.GRAY}Proxy stopped.{C.RESET}")


def cmd_benchmark(args):
    """entroly benchmark — run competitive comparison."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Competitive Benchmark{C.RESET}\n")

    bench_script = os.path.join(os.path.dirname(__file__), "..", "bench", "compare.py")
    if os.path.exists(bench_script):
        subprocess.run([sys.executable, bench_script])
    else:
        print(f"  {C.RED}Benchmark script not found at {bench_script}{C.RESET}")


def cmd_status(args):
    """entroly status — check if server/proxy is running."""
    import urllib.request

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Status{C.RESET}\n")

    port = args.port or 9377
    endpoints = [
        ("Proxy", f"http://127.0.0.1:{port}/health"),
        ("Dashboard", "http://127.0.0.1:9378/health"),
    ]

    for name, url in endpoints:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            data = json.loads(resp.read())
            status_text = data.get("status", "up")
            print(f"  {C.GREEN}[OK]{C.RESET} {name}: {url} -- {status_text}")
        except Exception:
            print(f"  {C.RED}[--]{C.RESET} {name}: not running")

    # Show stats if proxy is up
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=2)
        stats = json.loads(resp.read())
        total = stats.get("requests_total", 0)
        opt = stats.get("requests_optimized", 0)
        rate = stats.get("optimization_rate", "0%")
        breaker = stats.get("circuit_breaker", "closed")
        latency = stats.get("pipeline_latency", {})
        print(f"\n  {C.BOLD}Stats:{C.RESET}")
        print(f"    Requests: {opt}/{total} optimized ({rate})")
        print(f"    Circuit breaker: {breaker}")
        if latency.get("count", 0) > 0:
            print(f"    Pipeline latency: {latency['mean_ms']:.1f}ms avg (+/- {latency['stddev_ms']:.1f}ms)")
    except Exception:
        pass

    print()


def cmd_config(args):
    """entroly config show — display current configuration."""
    from entroly.proxy_config import ProxyConfig, QUALITY_PRESETS

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Configuration{C.RESET}\n")

    config = ProxyConfig.from_env()

    print(f"  {C.BOLD}Quality Presets:{C.RESET} {', '.join(f'{k}={v}' for k, v in QUALITY_PRESETS.items())}\n")

    # Group settings
    groups = {
        "Network": ["port", "host", "openai_base_url", "anthropic_base_url"],
        "Quality": ["quality", "context_fraction"],
        "Features": [k for k in vars(config) if k.startswith("enable_")],
        "ECDB": [k for k in vars(config) if k.startswith("ecdb_")],
        "IOS": [k for k in vars(config) if k.startswith("ios_")],
        "EGTC": ["fisher_scale", "egtc_alpha", "egtc_gamma", "egtc_epsilon",
                  "trajectory_c_min", "trajectory_lambda"],
    }

    for group_name, keys in groups.items():
        print(f"  {C.BOLD}{group_name}:{C.RESET}")
        for key in keys:
            val = getattr(config, key, None)
            if val is not None:
                print(f"    {C.GRAY}{key}:{C.RESET} {val}")
        print()


def cmd_telemetry(args):
    """entroly telemetry — manage anonymous usage statistics."""
    telemetry_file = _ENTROLY_DIR / "telemetry.json"

    if args.action == "on":
        _ENTROLY_DIR.mkdir(parents=True, exist_ok=True)
        telemetry_file.write_text(json.dumps({"enabled": True, "opted_in_at": __import__("time").time()}))
        print(f"  {C.GREEN}Telemetry enabled.{C.RESET} Anonymous usage stats will be collected.")
        print(f"  {C.GRAY}No personal data, API keys, or code content is ever sent.{C.RESET}")
        print(f"  {C.GRAY}To disable: entroly telemetry off{C.RESET}")
    elif args.action == "off":
        if telemetry_file.exists():
            telemetry_file.write_text(json.dumps({"enabled": False}))
        print(f"  {C.GREEN}Telemetry disabled.{C.RESET} No data will be collected.")
    elif args.action == "status":
        enabled = False
        if telemetry_file.exists():
            try:
                data = json.loads(telemetry_file.read_text())
                enabled = data.get("enabled", False)
            except (json.JSONDecodeError, OSError):
                pass
        status = f"{C.GREEN}enabled{C.RESET}" if enabled else f"{C.GRAY}disabled (default){C.RESET}"
        print(f"  Telemetry: {status}")
        print(f"  {C.GRAY}If enabled, only anonymous aggregates are sent:{C.RESET}")
        print(f"  {C.GRAY}  - Proxy vs MCP mode usage{C.RESET}")
        print(f"  {C.GRAY}  - Median codebase size (file count bucket){C.RESET}")
        print(f"  {C.GRAY}  - Feature flags enabled{C.RESET}")
        print(f"  {C.GRAY}  - p95 pipeline latency{C.RESET}")
        print(f"  {C.GRAY}  - OS + Python version{C.RESET}")


def is_telemetry_enabled() -> bool:
    """Check if opt-in telemetry is enabled. Always False by default."""
    telemetry_file = _ENTROLY_DIR / "telemetry.json"
    if not telemetry_file.exists():
        return False
    try:
        return json.loads(telemetry_file.read_text()).get("enabled", False)
    except (json.JSONDecodeError, OSError):
        return False


def cmd_clean(args):
    """entroly clean — clear cached state (checkpoints, index, pull cache)."""
    entroly_dir = Path.home() / ".entroly"

    if not entroly_dir.exists():
        print(f"  {C.GRAY}Nothing to clean — {entroly_dir} does not exist.{C.RESET}")
        return

    # Collect what will be removed
    targets = []
    checkpoint_dir = entroly_dir / "checkpoints"
    if checkpoint_dir.exists():
        count = sum(1 for _ in checkpoint_dir.rglob("*") if _.is_file())
        targets.append(("checkpoints", checkpoint_dir, count))

    index_files = list(entroly_dir.rglob("index.json.gz"))
    if index_files:
        targets.append(("index files", None, len(index_files)))

    pull_cache = entroly_dir / ".last_pull_ts"
    if pull_cache.exists():
        targets.append(("Docker pull cache", pull_cache, 1))

    if not targets:
        print(f"  {C.GRAY}Nothing to clean — no cached state found.{C.RESET}")
        return

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Clean{C.RESET}\n")
    for name, path, count in targets:
        print(f"  {C.YELLOW}{name}:{C.RESET} {count} file(s)")

    if args.yes:
        confirmed = True
    else:
        try:
            answer = input(f"\n  {C.BOLD}Remove all cached state? [y/N]{C.RESET} ").strip().lower()
            confirmed = answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            confirmed = False

    if not confirmed:
        print(f"  {C.GRAY}Aborted.{C.RESET}")
        return

    removed = 0
    # Remove checkpoints directory tree
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        removed += 1
        print(f"  {C.GREEN}Removed{C.RESET} {checkpoint_dir}")

    # Remove index files
    for idx_file in index_files:
        idx_file.unlink()
        removed += 1
        print(f"  {C.GREEN}Removed{C.RESET} {idx_file}")

    # Remove pull cache
    if pull_cache.exists():
        pull_cache.unlink()
        removed += 1
        print(f"  {C.GREEN}Removed{C.RESET} {pull_cache}")

    print(f"\n  {C.GREEN}{C.BOLD}Cleaned {removed} item(s).{C.RESET} Next run will start fresh.\n")


def cmd_export(args):
    """entroly export — export learned state for sharing (Gap #32)."""
    import time as _time

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Export{C.RESET}\n")

    entroly_dir = Path.home() / ".entroly"
    checkpoint_dir = entroly_dir / "checkpoints"
    tuning_config = Path(os.path.dirname(__file__)).parent / "tuning_config.json"

    export_data = {
        "exported_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": __version__,
    }

    # Include tuning config
    if tuning_config.exists():
        try:
            export_data["tuning_config"] = json.loads(tuning_config.read_text())
            print(f"  {C.GREEN}[+]{C.RESET} tuning_config.json")
        except (json.JSONDecodeError, OSError):
            pass

    # Include telemetry prefs
    telem_file = entroly_dir / "telemetry.json"
    if telem_file.exists():
        try:
            export_data["telemetry"] = json.loads(telem_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Write export file
    out_path = Path(args.output) if args.output else Path("entroly_export.json")
    out_path.write_text(json.dumps(export_data, indent=2) + "\n")
    print(f"\n  {C.GREEN}{C.BOLD}Exported to {out_path}{C.RESET}")
    print(f"  {C.GRAY}Share this file with teammates: entroly import {out_path}{C.RESET}\n")


def cmd_import(args):
    """entroly import — import shared learned state (Gap #32)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Import{C.RESET}\n")

    import_path = Path(args.file)
    if not import_path.exists():
        print(f"  {C.RED}File not found: {import_path}{C.RESET}")
        return

    try:
        data = json.loads(import_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  {C.RED}Invalid export file: {e}{C.RESET}")
        return

    print(f"  {C.GRAY}From: {data.get('exported_at', 'unknown')} (v{data.get('version', '?')}){C.RESET}")

    # Restore tuning config
    if "tuning_config" in data:
        tuning_path = Path(os.path.dirname(__file__)).parent / "tuning_config.json"
        tuning_path.write_text(json.dumps(data["tuning_config"], indent=2) + "\n")
        print(f"  {C.GREEN}[+]{C.RESET} tuning_config.json restored")

    print(f"\n  {C.GREEN}{C.BOLD}Import complete.{C.RESET} Restart proxy/serve to apply.\n")


def cmd_drift(args):
    """entroly drift — detect weight drift / staleness (Gap #30)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Drift Detection{C.RESET}\n")

    tuning_path = Path(os.path.dirname(__file__)).parent / "tuning_config.json"
    if not tuning_path.exists():
        print(f"  {C.GRAY}No tuning_config.json found — using defaults (no drift possible).{C.RESET}\n")
        return

    try:
        config = json.loads(tuning_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  {C.RED}Cannot read tuning_config.json: {e}{C.RESET}")
        return

    # Default weights for comparison
    defaults = {"recency": 0.3, "frequency": 0.25, "semantic_sim": 0.25, "entropy": 0.2}
    current = config.get("weights", {})

    total_drift = 0.0
    for key, default_val in defaults.items():
        cur_val = current.get(key, default_val)
        drift = abs(cur_val - default_val)
        total_drift += drift
        indicator = C.GREEN if drift < 0.05 else C.YELLOW if drift < 0.15 else C.RED
        print(f"  {indicator}{key}:{C.RESET} {cur_val:.3f} (default: {default_val:.3f}, drift: {drift:.3f})")

    print(f"\n  {C.BOLD}Total drift:{C.RESET} {total_drift:.3f}")
    if total_drift > 0.3:
        print(f"  {C.RED}Significant drift detected.{C.RESET} Consider resetting:")
        print(f"    {C.CYAN}entroly autotune --rollback{C.RESET}")
    elif total_drift > 0.1:
        print(f"  {C.YELLOW}Moderate drift.{C.RESET} Weights have adapted from defaults.")
    else:
        print(f"  {C.GREEN}Minimal drift.{C.RESET} Weights are close to defaults.")

    # Check config age via file mtime
    import time as _time
    mtime = tuning_path.stat().st_mtime
    age_days = (_time.time() - mtime) / 86400
    if age_days > 30:
        print(f"\n  {C.YELLOW}Config is {age_days:.0f} days old.{C.RESET} Consider re-running autotune.")
    print()


def cmd_profile(args):
    """entroly profile — manage per-project weight profiles (Gap #31)."""
    import hashlib

    profiles_dir = _ENTROLY_DIR / "profiles"

    if args.profile_action == "save":
        # Save current tuning config as a named profile
        tuning_path = Path(os.path.dirname(__file__)).parent / "tuning_config.json"
        if not tuning_path.exists():
            print(f"  {C.RED}No tuning_config.json to save.{C.RESET}")
            return
        profiles_dir.mkdir(parents=True, exist_ok=True)
        name = args.name or hashlib.sha256(os.getcwd().encode()).hexdigest()[:8]
        profile_path = profiles_dir / f"{name}.json"
        import shutil
        shutil.copy2(str(tuning_path), str(profile_path))
        print(f"  {C.GREEN}Profile '{name}' saved{C.RESET} ({profile_path})")

    elif args.profile_action == "load":
        if not args.name:
            print(f"  {C.RED}Specify a profile name: entroly profile load <name>{C.RESET}")
            return
        profile_path = profiles_dir / f"{args.name}.json"
        if not profile_path.exists():
            print(f"  {C.RED}Profile '{args.name}' not found.{C.RESET}")
            return
        tuning_path = Path(os.path.dirname(__file__)).parent / "tuning_config.json"
        import shutil
        shutil.copy2(str(profile_path), str(tuning_path))
        print(f"  {C.GREEN}Profile '{args.name}' loaded.{C.RESET} Restart proxy to apply.")

    elif args.profile_action == "list":
        if not profiles_dir.exists():
            print(f"  {C.GRAY}No profiles saved yet.{C.RESET}")
            return
        for p in sorted(profiles_dir.glob("*.json")):
            name = p.stem
            size = p.stat().st_size
            print(f"  {C.CYAN}{name}{C.RESET} ({size} bytes)")

    else:
        print(f"  Usage: entroly profile {{save|load|list}} [name]")


def cmd_batch(args):
    """entroly batch — headless/CI mode for batch optimization (Gap #33)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Batch Mode{C.RESET}\n")

    from entroly.server import EntrolyEngine
    from entroly.auto_index import auto_index

    engine = EntrolyEngine()
    result = auto_index(engine)

    if result["status"] == "indexed":
        print(f"  {C.GREEN}Indexed {result['files_indexed']} files{C.RESET}")

    # Read queries from stdin or file
    import sys as _sys
    if args.input and args.input != "-":
        with open(args.input) as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = [line.strip() for line in _sys.stdin if line.strip()]

    results = []
    for i, query in enumerate(queries, 1):
        engine.advance_turn()
        opt = engine.optimize_context(
            token_budget=args.budget,
            query=query,
        )
        selected = opt.get("selected_fragments", [])
        total_tokens = sum(f.get("token_count", 0) for f in selected)
        results.append({
            "query": query,
            "fragments_selected": len(selected),
            "tokens_used": total_tokens,
            "budget": args.budget,
        })
        if not args.json_output:
            print(f"  [{i}/{len(queries)}] {query[:60]}... → {len(selected)} fragments, {total_tokens} tokens")

    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n  {C.GREEN}{C.BOLD}Processed {len(queries)} queries.{C.RESET}\n")


def cmd_demo(args):
    """entroly demo — quick-win demo mode: before/after comparison (Gap #41)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Demo{C.RESET} — see the value in 30 seconds\n")

    from entroly.server import EntrolyEngine
    from entroly.auto_index import auto_index

    engine = EntrolyEngine()
    result = auto_index(engine)

    if result["status"] != "indexed" or result.get("files_indexed", 0) == 0:
        print(f"  {C.YELLOW}No files found to index.{C.RESET}")
        print(f"  Run this from a project directory with source files.\n")
        return

    files_indexed = result["files_indexed"]
    total_tokens_raw = result["total_tokens"]
    print(f"  {C.GREEN}Indexed {files_indexed} files ({total_tokens_raw:,} tokens total){C.RESET}\n")

    sample_queries = [
        "How does the authentication flow work?",
        "Find and fix potential SQL injection vulnerabilities",
        "Explain the module structure and dependency graph",
    ]

    budget = 4096
    print(f"  {C.BOLD}Without Entroly:{C.RESET} All {total_tokens_raw:,} tokens sent to LLM (wasteful)")
    print(f"  {C.BOLD}With Entroly:{C.RESET} Optimized context for each query:\n")

    total_saved = 0
    for query in sample_queries:
        engine.advance_turn()
        opt = engine.optimize_context(token_budget=budget, query=query)
        selected = opt.get("selected_fragments", [])
        tokens_used = sum(f.get("token_count", 0) for f in selected)
        saved = total_tokens_raw - tokens_used
        total_saved += saved
        pct = (saved * 100) // max(total_tokens_raw, 1)
        print(f"    {C.CYAN}Q:{C.RESET} {query[:60]}")
        print(f"       {C.GREEN}{len(selected)} fragments, {tokens_used:,} tokens{C.RESET} "
              f"({C.BOLD}{pct}% reduction{C.RESET})\n")

    avg_pct = (total_saved * 100) // max(total_tokens_raw * len(sample_queries), 1)
    print(f"  {C.GREEN}{C.BOLD}Average: {avg_pct}% fewer tokens sent to LLM{C.RESET}")
    print(f"  {C.GRAY}Start using: entroly proxy --quality balanced{C.RESET}\n")


def cmd_doctor(args):
    """entroly doctor — diagnose common issues (Gap #52)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Doctor{C.RESET}\n")

    checks_passed = 0
    checks_total = 0

    # 1. Check Python version
    checks_total += 1
    py_ver = platform.python_version()
    if sys.version_info >= (3, 10):
        print(f"  {C.GREEN}✓{C.RESET} Python {py_ver}")
        checks_passed += 1
    else:
        print(f"  {C.RED}✗{C.RESET} Python {py_ver} (need ≥3.10)")

    # 2. Check Rust engine
    checks_total += 1
    try:
        import entroly_core  # noqa: F401
        print(f"  {C.GREEN}✓{C.RESET} Rust engine (entroly-core) loaded")
        checks_passed += 1
    except ImportError:
        print(f"  {C.RED}✗{C.RESET} Rust engine not installed (pip install entroly-core)")

    # 3. Check config validity
    checks_total += 1
    tuning_path = Path(__file__).parent / "tuning_config.json"
    if tuning_path.exists():
        try:
            with open(tuning_path) as f:
                tc = json.load(f)
            weights = tc.get("weights", {})
            w_sum = sum(weights.values()) if weights else 0
            if abs(w_sum - 1.0) < 0.01:
                print(f"  {C.GREEN}✓{C.RESET} Config valid (weights sum={w_sum:.3f})")
                checks_passed += 1
            else:
                print(f"  {C.YELLOW}!{C.RESET} Config: weights sum={w_sum:.3f} (expected ~1.0)")
                checks_passed += 1  # warning, not failure
        except Exception as e:
            print(f"  {C.RED}✗{C.RESET} Config error: {e}")
    else:
        print(f"  {C.GREEN}✓{C.RESET} Config: using defaults (no tuning_config.json)")
        checks_passed += 1

    # 4. Check proxy reachability
    checks_total += 1
    port = getattr(args, "port", None) or 9377
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
        if resp.status == 200:
            print(f"  {C.GREEN}✓{C.RESET} Proxy reachable at localhost:{port}")
            checks_passed += 1
        else:
            print(f"  {C.YELLOW}!{C.RESET} Proxy responded with status {resp.status}")
    except Exception:
        print(f"  {C.GRAY}-{C.RESET} Proxy not running (localhost:{port})")
        checks_passed += 1  # not running is OK for doctor

    # 5. Check index freshness
    checks_total += 1
    entroly_dir = Path.home() / ".entroly"
    checkpoint_dir = entroly_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.json*"))
        if checkpoint_files:
            newest = max(f.stat().st_mtime for f in checkpoint_files)
            import time as _time
            age_hours = (_time.time() - newest) / 3600
            if age_hours < 24:
                print(f"  {C.GREEN}✓{C.RESET} Index fresh ({age_hours:.1f}h old)")
            else:
                print(f"  {C.YELLOW}!{C.RESET} Index stale ({age_hours:.0f}h old — consider re-indexing)")
            checks_passed += 1
        else:
            print(f"  {C.GRAY}-{C.RESET} No index found (will be created on first run)")
            checks_passed += 1
    else:
        print(f"  {C.GRAY}-{C.RESET} No checkpoints directory")
        checks_passed += 1

    # 6. Check weight drift
    checks_total += 1
    if tuning_path.exists():
        try:
            with open(tuning_path) as f:
                tc = json.load(f)
            defaults = {"recency": 0.30, "frequency": 0.25, "semantic": 0.25, "entropy": 0.20}
            weights = tc.get("weights", defaults)
            drift = sum(abs(weights.get(k, v) - v) for k, v in defaults.items())
            if drift < 0.1:
                print(f"  {C.GREEN}✓{C.RESET} Weights near defaults (drift={drift:.3f})")
            elif drift < 0.3:
                print(f"  {C.YELLOW}!{C.RESET} Weights drifted (drift={drift:.3f})")
            else:
                print(f"  {C.RED}✗{C.RESET} Weights heavily drifted ({drift:.3f}) — consider autotune --rollback")
            checks_passed += 1
        except Exception:
            checks_passed += 1
    else:
        print(f"  {C.GREEN}✓{C.RESET} Weights: defaults (no drift)")
        checks_passed += 1

    # 7. Check Docker (optional)
    checks_total += 1
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  {C.GREEN}✓{C.RESET} Docker available")
        else:
            print(f"  {C.GRAY}-{C.RESET} Docker not running (optional)")
        checks_passed += 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  {C.GRAY}-{C.RESET} Docker not installed (optional)")
        checks_passed += 1

    print(f"\n  {C.BOLD}{checks_passed}/{checks_total} checks passed{C.RESET}\n")


def cmd_digest(args):
    """entroly digest — show weekly summary of entroly's value (Gap #44)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Weekly Digest{C.RESET}\n")

    import time as _time

    # Try to get stats from running proxy
    port = getattr(args, "port", None) or 9377
    stats = None
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"http://localhost:{port}/stats", timeout=2)
        if resp.status == 200:
            stats = json.loads(resp.read())
    except Exception:
        pass

    if stats:
        total = stats.get("requests_total", 0)
        optimized = stats.get("requests_optimized", 0)
        tokens = stats.get("tokens", {})
        saved = tokens.get("saved_total", 0)
        savings_pct = tokens.get("savings_pct", "N/A")
        outcomes = stats.get("outcomes", {})
        error_rate = outcomes.get("error_rate", 0)
        latency = stats.get("pipeline_latency", {})
        mean_ms = latency.get("mean_ms", 0)

        print(f"  {C.BOLD}Requests:{C.RESET} {total:,} total, {optimized:,} optimized")
        print(f"  {C.BOLD}Tokens saved:{C.RESET} {saved:,} ({savings_pct})")
        est_cost = saved * 0.000003
        print(f"  {C.BOLD}Estimated cost saved:{C.RESET} ${est_cost:.2f}")
        print(f"  {C.BOLD}Pipeline latency:{C.RESET} {mean_ms:.1f}ms avg")
        if error_rate > 0:
            color = C.RED if error_rate > 0.1 else C.YELLOW
            print(f"  {C.BOLD}Error rate:{C.RESET} {color}{error_rate:.1%}{C.RESET}")
        else:
            print(f"  {C.BOLD}Error rate:{C.RESET} {C.GREEN}0%{C.RESET}")
    else:
        # Fall back to checkpoint/stats file
        stats_file = Path.home() / ".entroly" / "session_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    data = json.load(f)
                total_saved = data.get("total_tokens_saved", 0)
                total_opt = data.get("total_optimizations", 0)
                print(f"  {C.BOLD}Sessions recorded:{C.RESET} {total_opt:,} optimizations")
                print(f"  {C.BOLD}Tokens saved (lifetime):{C.RESET} {total_saved:,}")
            except Exception:
                print(f"  {C.YELLOW}No stats available.{C.RESET} Start the proxy to begin tracking.")
        else:
            print(f"  {C.YELLOW}No stats available.{C.RESET} Start the proxy to begin tracking.")
    print()


def cmd_migrate(args):
    """entroly migrate — auto-migrate config/index to new format (Gap #53)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Migration Check{C.RESET}\n")

    from entroly import __version__ as current_version

    entroly_dir = Path.home() / ".entroly"
    version_file = entroly_dir / ".version"

    # Check stored version
    stored_version = None
    if version_file.exists():
        try:
            stored_version = version_file.read_text().strip()
        except OSError:
            pass

    if stored_version == current_version:
        print(f"  {C.GREEN}✓{C.RESET} Already on version {current_version}. No migration needed.\n")
        return

    if stored_version:
        print(f"  Upgrading from {C.YELLOW}{stored_version}{C.RESET} to {C.GREEN}{current_version}{C.RESET}\n")
    else:
        print(f"  First run of version {C.GREEN}{current_version}{C.RESET}\n")

    migrated = 0

    # Check tuning_config.json schema
    tuning_path = Path(__file__).parent / "tuning_config.json"
    if tuning_path.exists():
        try:
            with open(tuning_path) as f:
                tc = json.load(f)
            # Ensure all required sections exist
            changed = False
            if "weights" not in tc:
                tc["weights"] = {"recency": 0.30, "frequency": 0.25, "semantic": 0.25, "entropy": 0.20}
                changed = True
            if "decay" not in tc:
                tc["decay"] = {"half_life": 15, "min_relevance": 0.05}
                changed = True
            if "knapsack" not in tc:
                tc["knapsack"] = {"exploration_rate": 0.10}
                changed = True
            if changed:
                with open(tuning_path, "w") as f:
                    json.dump(tc, f, indent=2)
                print(f"  {C.GREEN}✓{C.RESET} Migrated tuning_config.json (added missing sections)")
                migrated += 1
            else:
                print(f"  {C.GREEN}✓{C.RESET} tuning_config.json: schema up to date")
        except Exception as e:
            print(f"  {C.RED}✗{C.RESET} tuning_config.json error: {e}")
    else:
        print(f"  {C.GREEN}✓{C.RESET} No tuning_config.json (using defaults)")

    # Check checkpoint format
    checkpoint_dir = entroly_dir / "checkpoints"
    if checkpoint_dir.exists():
        old_checkpoints = list(checkpoint_dir.glob("*.json"))
        gz_checkpoints = list(checkpoint_dir.glob("*.json.gz"))
        if old_checkpoints and not gz_checkpoints:
            print(f"  {C.YELLOW}!{C.RESET} Found {len(old_checkpoints)} uncompressed checkpoints")
            print(f"       Run {C.CYAN}entroly clean{C.RESET} + re-index for compressed format")
        else:
            print(f"  {C.GREEN}✓{C.RESET} Checkpoints: format OK ({len(gz_checkpoints)} compressed)")
    else:
        print(f"  {C.GREEN}✓{C.RESET} No checkpoints to migrate")

    # Write version marker
    try:
        entroly_dir.mkdir(parents=True, exist_ok=True)
        version_file.write_text(current_version)
        print(f"\n  {C.GREEN}{C.BOLD}Migration complete.{C.RESET} "
              f"({migrated} item{'s' if migrated != 1 else ''} migrated)\n")
    except OSError:
        pass


def cmd_role(args):
    """entroly role — role-based weight presets for different developer types (Gap #49)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Role Presets{C.RESET}\n")

    roles = {
        "frontend": {
            "description": "Frontend developer (React, Vue, CSS)",
            "weights": {"recency": 0.25, "frequency": 0.30, "semantic": 0.30, "entropy": 0.15},
            "note": "Higher semantic + frequency for component reuse patterns",
        },
        "backend": {
            "description": "Backend developer (API, database, services)",
            "weights": {"recency": 0.30, "frequency": 0.20, "semantic": 0.25, "entropy": 0.25},
            "note": "Balanced with higher entropy for diverse system interactions",
        },
        "sre": {
            "description": "SRE / DevOps (infra, CI/CD, monitoring)",
            "weights": {"recency": 0.35, "frequency": 0.15, "semantic": 0.20, "entropy": 0.30},
            "note": "High recency + entropy for fast-changing infra context",
        },
        "data": {
            "description": "Data engineer / ML (SQL, pipelines, notebooks)",
            "weights": {"recency": 0.20, "frequency": 0.30, "semantic": 0.30, "entropy": 0.20},
            "note": "Higher frequency + semantic for repeated query patterns",
        },
        "fullstack": {
            "description": "Full-stack developer (balanced across all areas)",
            "weights": {"recency": 0.25, "frequency": 0.25, "semantic": 0.25, "entropy": 0.25},
            "note": "Equal weights for broad codebase interaction",
        },
    }

    action = getattr(args, "role_action", "list")

    if action == "list":
        for name, info in roles.items():
            w = info["weights"]
            print(f"  {C.CYAN}{name:12s}{C.RESET} {info['description']}")
            print(f"               R={w['recency']:.2f}  F={w['frequency']:.2f}  "
                  f"S={w['semantic']:.2f}  E={w['entropy']:.2f}")
            print(f"               {C.GRAY}{info['note']}{C.RESET}\n")
        print(f"  Apply with: {C.CYAN}entroly role apply <name>{C.RESET}\n")

    elif action == "apply":
        name = getattr(args, "name", None)
        if not name or name not in roles:
            valid = ", ".join(roles.keys())
            print(f"  {C.RED}Unknown role.{C.RESET} Valid: {valid}")
            return
        role = roles[name]
        tuning_path = Path(__file__).parent / "tuning_config.json"
        tc = {}
        if tuning_path.exists():
            try:
                with open(tuning_path) as f:
                    tc = json.load(f)
            except Exception:
                pass
        tc["weights"] = role["weights"]
        with open(tuning_path, "w") as f:
            json.dump(tc, f, indent=2)
        print(f"  {C.GREEN}Applied '{name}' role preset:{C.RESET}")
        w = role["weights"]
        print(f"    R={w['recency']:.2f}  F={w['frequency']:.2f}  "
              f"S={w['semantic']:.2f}  E={w['entropy']:.2f}")
        print(f"    {C.GRAY}{role['note']}{C.RESET}\n")


def cmd_completions(args):
    """entroly completions {bash|zsh|fish} — output shell completion script."""
    shell = args.shell
    commands = [
        "init", "serve", "proxy", "dashboard", "health",
        "autotune", "benchmark", "status", "config", "clean",
        "telemetry", "export", "import", "drift", "profile",
        "batch", "demo", "doctor", "digest", "migrate", "role",
        "completions",
    ]
    cmd_list = " ".join(commands)

    if shell == "bash":
        print(f"""# entroly bash completion — add to ~/.bashrc:
#   eval "$(entroly completions bash)"
_entroly_completions() {{
    local cur="${{COMP_WORDS[COMP_CWORD]}}"
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=($(compgen -W "{cmd_list} --help --version" -- "$cur"))
    elif [ "${{COMP_WORDS[1]}}" = "proxy" ]; then
        COMPREPLY=($(compgen -W "--port --host --quality --force --help" -- "$cur"))
    elif [ "${{COMP_WORDS[1]}}" = "completions" ]; then
        COMPREPLY=($(compgen -W "bash zsh fish" -- "$cur"))
    elif [ "${{COMP_WORDS[1]}}" = "init" ]; then
        COMPREPLY=($(compgen -W "--dry-run --help" -- "$cur"))
    fi
}}
complete -F _entroly_completions entroly""")
    elif shell == "zsh":
        print(f"""# entroly zsh completion — add to ~/.zshrc:
#   eval "$(entroly completions zsh)"
_entroly() {{
    local -a commands
    commands=({cmd_list})
    _arguments '1:command:($commands)' '*::arg:->args'
    case $words[1] in
        proxy) _arguments '--port[Proxy port]:port' '--host[Bind host]:host' '--quality[Quality 0-1]:quality' '--force[Force re-index]' ;;
        completions) _arguments '1:shell:(bash zsh fish)' ;;
        init) _arguments '--dry-run[Preview only]' ;;
    esac
}}
compdef _entroly entroly""")
    elif shell == "fish":
        print(f"""# entroly fish completion — save to ~/.config/fish/completions/entroly.fish
complete -c entroly -n '__fish_use_subcommand' -a '{cmd_list}' -d 'Entroly commands'
complete -c entroly -n '__fish_seen_subcommand_from proxy' -l port -d 'Proxy port'
complete -c entroly -n '__fish_seen_subcommand_from proxy' -l host -d 'Bind host'
complete -c entroly -n '__fish_seen_subcommand_from proxy' -l quality -d 'Quality 0-1'
complete -c entroly -n '__fish_seen_subcommand_from completions' -a 'bash zsh fish'""")
    else:
        print(f"Unknown shell: {shell}. Supported: bash, zsh, fish", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="entroly",
        description="Entroly — Information-theoretic context optimization for AI coding agents",
    )
    parser.add_argument(
        "--version", "-V", action="version",
        version=f"entroly {__version__}",
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
    serve_parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug-level logging (all subsystem details to stderr)",
    )

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
        "--bypass", action="store_true",
        help="Start in bypass mode (forward requests unmodified, no optimization)",
    )

    # entroly benchmark
    subparsers.add_parser(
        "benchmark",
        help="Run competitive benchmark: Entroly vs Raw vs Top-K",
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

    # entroly config
    subparsers.add_parser(
        "config",
        help="Show current configuration",
    )

    # entroly telemetry
    telem_parser = subparsers.add_parser(
        "telemetry",
        help="Manage anonymous usage statistics (opt-in, disabled by default)",
    )
    telem_parser.add_argument(
        "action", choices=["on", "off", "status"],
        help="Enable, disable, or check telemetry status",
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
        "-o", "--output", type=str, default=None,
        help="Output file path (default: entroly_export.json)",
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

    # entroly demo (Gap #41)
    subparsers.add_parser(
        "demo",
        help="Quick-win demo: before/after comparison showing token savings",
    )

    # entroly doctor (Gap #52)
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Diagnose common issues: index, config, proxy, weights",
    )
    doctor_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port to check (default: 9377)",
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
        "role_action", choices=["list", "apply"],
        help="List available roles or apply one",
    )
    role_parser.add_argument(
        "name", nargs="?", default=None,
        help="Role name to apply",
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

    args = parser.parse_args()

    # First-run welcome + update check (non-blocking)
    _check_first_run()
    if args.command not in (None, "completions"):
        _check_for_update()

    _dispatch = {
        "init": cmd_init,
        "serve": cmd_serve,
        "dashboard": cmd_dashboard,
        "health": cmd_health,
        "autotune": cmd_autotune,
        "proxy": cmd_proxy,
        "benchmark": cmd_benchmark,
        "status": cmd_status,
        "config": cmd_config,
        "clean": cmd_clean,
        "telemetry": cmd_telemetry,
        "export": cmd_export,
        "import": cmd_import,
        "drift": cmd_drift,
        "profile": cmd_profile,
        "batch": cmd_batch,
        "demo": cmd_demo,
        "doctor": cmd_doctor,
        "digest": cmd_digest,
        "migrate": cmd_migrate,
        "role": cmd_role,
        "completions": cmd_completions,
    }

    handler = _dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
