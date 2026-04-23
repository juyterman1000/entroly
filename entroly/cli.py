"""
Entroly CLI — Zero-friction onboarding for AI coding agents.
(Test change to trigger cost-check workflow)

Commands:
    entroly optimize    Generate optimized context snapshot for a task
    entroly feedback    Signal outcome quality to improve future context
    entroly go          One command: auto-detect, init, proxy, and dashboard
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
    __version__ = "0.8.2"

# ── Force UTF-8 output on Windows ──
# Windows terminals default to cp1252 which can't encode ✓/✗/─/⚡.
# Reconfigure stdout/stderr to UTF-8 with error replacement so print()
# never raises UnicodeEncodeError.
if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name)
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


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


def _free_port(port: int) -> bool:
    """Kill any stale entroly process occupying *port*. Returns True if the port is now free."""
    import signal
    import socket
    import time as _time

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return True  # port is free

    killed = False
    try:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True, timeout=3,
        )
        pids = result.stdout.strip().split()
        for pid_str in pids:
            pid_str = pid_str.strip()
            if pid_str.isdigit():
                pid = int(pid_str)
                if pid == os.getpid():
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed = True
                except ProcessLookupError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if killed:
        _time.sleep(0.3)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return True
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=3)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        _time.sleep(0.3)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


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
""", file=sys.stderr)
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
                        f"{C.GRAY}(pip install --upgrade entroly){C.RESET}",
                        file=sys.stderr,
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
                    f"{C.GRAY}(pip install --upgrade entroly){C.RESET}",
                    file=sys.stderr,
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

    # Detect JS/TS frameworks for richer project context
    frameworks = []
    framework_indicators = {
        "Next.js": ["next.config.js", "next.config.mjs", "next.config.ts"],
        "Angular": ["angular.json"],
        "Nuxt": ["nuxt.config.ts", "nuxt.config.js"],
        "Remix": ["remix.config.js", "remix.config.ts"],
        "Vite": ["vite.config.ts", "vite.config.js", "vite.config.mts"],
        "Svelte": ["svelte.config.js"],
        "Astro": ["astro.config.mjs"],
        "Gatsby": ["gatsby-config.js", "gatsby-config.ts"],
        "Expo": ["app.json", "expo.json"],
    }
    for fw, fw_files in framework_indicators.items():
        for f in fw_files:
            if os.path.exists(os.path.join(cwd, f)):
                frameworks.append(fw)
                break

    # Check package.json for React/Vue/Angular dependencies
    pkg_path = os.path.join(cwd, "package.json")
    if os.path.isfile(pkg_path):
        try:
            with open(pkg_path) as f:
                pkg = json.load(f)
            all_deps = set(pkg.get("dependencies", {}).keys()) | set(pkg.get("devDependencies", {}).keys())
            if "react" in all_deps and "Next.js" not in frameworks and "Remix" not in frameworks:
                frameworks.append("React")
            if "vue" in all_deps and "Nuxt" not in frameworks:
                frameworks.append("Vue")
            if "@angular/core" in all_deps and "Angular" not in frameworks:
                frameworks.append("Angular")
            if "express" in all_deps:
                frameworks.append("Express")
            if "fastify" in all_deps:
                frameworks.append("Fastify")
            if "nest" in all_deps or "@nestjs/core" in all_deps:
                frameworks.append("NestJS")
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    project_name = os.path.basename(cwd)
    return {
        "name": project_name,
        "languages": detected or ["unknown"],
        "primary": detected[0] if detected else "unknown",
        "frameworks": frameworks,
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
    parse_failed = False
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            parse_failed = True
            existing = {}

    # Merge entroly config into existing
    if config_key not in existing:
        existing[config_key] = {}
    existing[config_key].update(entroly_config)

    if dry_run:
        return json.dumps(existing, indent=2)

    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Backup existing file before overwrite (skip if parse failed — preserve bytes).
    if os.path.exists(config_path):
        backup = config_path + ".entroly-backup"
        try:
            import shutil
            shutil.copy2(config_path, backup)
        except OSError:
            pass
        if parse_failed:
            print(f"  {C.YELLOW if hasattr(C,'YELLOW') else ''}! Existing config at {config_path} was unparseable; original kept at {backup}{C.RESET}")

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
    langs = ', '.join(project['languages'])
    fw = project.get('frameworks', [])
    fw_str = f" + {', '.join(fw)}" if fw else ""
    print(f"  {C.GRAY}Project:{C.RESET}  {C.BOLD}{project['name']}{C.RESET} ({langs}{fw_str})")

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
    from entroly.auto_index import auto_index
    from entroly.dashboard import start_dashboard
    from entroly.server import EntrolyEngine

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Value Dashboard{C.RESET}\n")

    engine = EntrolyEngine()
    result = auto_index(engine, force=args.force)

    if result["status"] == "indexed":
        print(f"  {C.GREEN}Indexed {result['files_indexed']} files ({result['total_tokens']:,} tokens) in {result['duration_s']}s{C.RESET}")
    elif result["status"] == "skipped":
        print(f"  {C.GRAY}Using persistent index ({result['existing_fragments']} fragments){C.RESET}")

    # Run an optimize to populate all engine subsystems
    engine.optimize_context(token_budget=128000, query="project overview")

    # Free dashboard port from any stale process
    if not _free_port(args.port):
        print(f"  {C.RED}Port {args.port} is in use and could not be freed.{C.RESET}")
        print(f"  {C.GRAY}Try: entroly dashboard --port <other-port>{C.RESET}")
        return

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

    from entroly.auto_index import auto_index
    from entroly.server import EntrolyEngine

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
            taint = sec.get("taint_flow_total", 0)
            pat = sec.get("pattern_only_total", 0)
            print(f"\n  {C.RED}{total_findings} security findings ({sec.get('critical_total', 0)} critical, {sec.get('high_total', 0)} high){C.RESET}")
            print(f"    {C.GRAY}{taint} taint-flow (high-confidence) + {pat} pattern-only (review for FPs){C.RESET}")
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
        print(f"\n{C.CYAN}{C.BOLD}  Entroly Autotune -- Rollback{C.RESET}\n")
        try:
            from bench.autotune import rollback_config
            config_path = Path(os.path.dirname(__file__)).parent / "tuning_config.json"
            result = rollback_config(config_path)
            if result["status"] == "no_backup_found":
                print(f"  {C.RED}No backup found -- nothing to roll back.{C.RESET}\n")
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
        # composite_score = 0.50·recall + 0.25·precision + 0.25·context_efficiency
        # on bench/cases.json. Range [0, 1]; higher is better.
        print(f"\n  {C.GREEN}{C.BOLD}Best composite score: {result.get('final_score', 0):.4f} / 1.0{C.RESET}")
        print(f"  {C.GRAY}= 0.50·recall + 0.25·precision + 0.25·efficiency on bench/cases.json{C.RESET}")
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

    from entroly.auto_index import auto_index, start_incremental_watcher
    from entroly.proxy import create_proxy_app
    from entroly.proxy_config import ProxyConfig, resolve_quality
    from entroly.server import EntrolyEngine

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

    # Free ports from any stale entroly processes
    if not _free_port(config.port):
        print(f"  {C.RED}Port {config.port} is in use and could not be freed.{C.RESET}")
        print(f"  {C.GRAY}Try: entroly proxy --port <other-port>{C.RESET}")
        return
    _free_port(9378)  # dashboard port

    # Upstream connectivity check — fast-fail if the LLM API is unreachable
    _check_upstream(config)

    # Create the ASGI app (this also starts the dashboard on :9378)
    app = create_proxy_app(engine, config)

    print(f"""
  {C.GREEN}{C.BOLD}Proxy live at http://{config.host}:{config.port}{C.RESET}
  {C.GREEN}{C.BOLD}Dashboard at http://localhost:9378{C.RESET}

  {C.BOLD}To use:{C.RESET} Set your AI tool's API base URL to:
    {C.CYAN}http://localhost:{config.port}/v1{C.RESET}

  {C.GRAY}Every LLM request is intercepted -> optimized -> forwarded.{C.RESET}
  {C.GRAY}Live pipeline latency: http://localhost:9378 — or `entroly status`.{C.RESET}
  {C.GRAY}File watcher active — new/modified files auto-indexed every 120s.{C.RESET}
  {C.GRAY}Press Ctrl+C to stop.{C.RESET}
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

    from entroly.auto_index import auto_index
    from entroly.server import EntrolyEngine

    engine = EntrolyEngine()
    auto_index(engine)

    queries = [
        "How does authentication work?",
        "Find security vulnerabilities",
        "Explain the data model",
    ]

    # Get total tokens
    if engine._use_rust:
        stats = engine._rust.stats()
        total = stats.get("session", {}).get("total_tokens_tracked", 0)
    else:
        total = getattr(engine, "_total_token_count", 0)

    if total == 0:
        print(f"  {C.YELLOW}No files indexed. Run from a project directory.{C.RESET}")
        return

    # Honest baseline: a 32K "paste matching files into the prompt" dump,
    # capped at what actually exists. Claiming savings vs. the whole repo
    # (7M+ tokens) is marketing, not measurement.
    baseline = min(total, 32_000)
    budget = getattr(args, "budget", 4096)
    print(f"  Codebase: {total:,} total tokens  |  Naive baseline: {baseline:,} tokens (32K dump or repo total)\n")
    print(f"  {'Query':<45} {'Baseline':>9} {'Entroly':>8} {'Saved':>6}")
    print(f"  {'-'*45} {'-'*9} {'-'*8} {'-'*6}")

    total_saved = 0
    for q in queries:
        engine.advance_turn()
        opt = engine.optimize_context(token_budget=budget, query=q)
        selected = opt.get("selected_fragments", []) or opt.get("selected", [])
        used = sum(f.get("token_count", 0) for f in selected)
        saved = max(baseline - used, 0)
        pct = (saved * 100) // max(baseline, 1)
        total_saved += saved
        print(f"  {q:<45} {baseline:>9,} {used:>7,} {pct:>5}%")

    avg_pct = (total_saved * 100) // max(baseline * len(queries), 1)
    print(f"\n  {C.GREEN}{C.BOLD}Average reduction vs. 32K dump: {avg_pct}%{C.RESET}")
    print(f"  {C.GRAY}Budget: {budget:,} tokens. Selector picks whole fragments; no byte-truncation.{C.RESET}\n")


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

    # Show engine stats (resonance, coverage, consolidation) if available
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/engine-stats", timeout=2)
        engine_stats = json.loads(resp.read())
        resonance = engine_stats.get("resonance", {})
        consolidation = engine_stats.get("consolidation", {})
        if resonance:
            pairs = resonance.get("tracked_pairs", 0)
            strength = resonance.get("mean_strength", 0.0)
            w_res = resonance.get("w_resonance", 0.0)
            calibrated = resonance.get("is_calibrated", False)
            cal_str = f"{C.GREEN}calibrated{C.RESET}" if calibrated else f"{C.YELLOW}cold-start{C.RESET}"
            print(f"\n  {C.BOLD}Context Resonance:{C.RESET}")
            print(f"    Pairs tracked: {pairs}  |  Mean strength: {strength:.4f}  |  w_resonance: {w_res:.4f}")
            print(f"    Status: {cal_str}")
        if consolidation:
            total_c = consolidation.get("total_consolidations", 0)
            tokens_c = consolidation.get("tokens_saved", 0)
            if total_c > 0:
                print(f"\n  {C.BOLD}Fragment Consolidation (Maxwell's Demon):{C.RESET}")
                print(f"    Consolidated: {total_c} fragments  |  Tokens saved: {tokens_c}")
        causal = engine_stats.get("causal", {})
        if causal:
            traces = causal.get("total_traces", 0)
            tracked = causal.get("tracked_fragments", 0)
            interventional = causal.get("interventional_fragments", 0)
            temporal = causal.get("temporal_links", 0)
            gravity = causal.get("gravity_sources", 0)
            mass = causal.get("mean_causal_mass", 0.0)
            print(f"\n  {C.BOLD}Causal Context Graph:{C.RESET}")
            print(f"    Traces: {traces}  |  Tracked: {tracked}  |  Interventional: {interventional}")
            print(f"    Temporal links: {temporal}  |  Gravity sources: {gravity}  |  Mean mass: {mass:.4f}")
    except Exception:
        pass

    print()


def cmd_config(args):
    """entroly config show — display current configuration."""
    from entroly.proxy_config import QUALITY_PRESETS, ProxyConfig

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Configuration{C.RESET}\n")

    config = ProxyConfig.from_env()

    print(f"  {C.BOLD}Quality Presets:{C.RESET} {', '.join(f'{k}={v}' for k, v in QUALITY_PRESETS.items())}\n")

    # Group settings
    groups = {
        "Network": ["port", "host", "openai_base_url", "anthropic_base_url", "gemini_base_url"],
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
        print(f"  {C.GRAY}Nothing to clean -- {entroly_dir} does not exist.{C.RESET}")
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
        print(f"  {C.GRAY}Nothing to clean -- no cached state found.{C.RESET}")
        return

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Clean{C.RESET}\n")
    for name, path, count in targets:
        print(f"  {C.YELLOW}{name}:{C.RESET} {count} file(s)")

    if args.yes:
        confirmed = True
    elif not sys.stdin.isatty():
        # Non-interactive (piped/redirected) — skip prompt, default to no
        confirmed = False
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

    # Remove index files (skip those already removed with checkpoints dir)
    for idx_file in index_files:
        if idx_file.exists():
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

    # Write export file. Positional `output_path` takes precedence over -o/--output.
    chosen = getattr(args, "output_path", None) or args.output
    out_path = Path(chosen) if chosen else Path("entroly_export.json")
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
        print(f"  {C.GRAY}No tuning_config.json found -- using defaults (no drift possible).{C.RESET}\n")
        return

    try:
        config = json.loads(tuning_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  {C.RED}Cannot read tuning_config.json: {e}{C.RESET}")
        return

    # Default weights for comparison. Keep keys in sync with cmd_doctor and
    # migrate defaults — mismatched keys silently report bogus drift.
    defaults = {"recency": 0.30, "frequency": 0.25, "semantic": 0.25, "entropy": 0.20}
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
        print("  Usage: entroly profile {save|load|list} [name]")


def cmd_batch(args):
    """entroly batch — headless/CI mode for batch optimization (Gap #33)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Batch Mode{C.RESET}\n")

    from entroly.auto_index import auto_index
    from entroly.server import EntrolyEngine

    engine = EntrolyEngine()
    result = auto_index(engine)

    if result["status"] == "indexed":
        print(f"  {C.GREEN}Indexed {result['files_indexed']} files{C.RESET}")
    elif result["status"] == "skipped":
        print(f"  {C.GRAY}Using persistent index ({result.get('existing_fragments', 0)} fragments){C.RESET}")

    # Read queries from stdin or file
    import sys as _sys
    if args.input and args.input != "-":
        try:
            with open(args.input) as f:
                queries = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"  {C.RED}File not found: {args.input}{C.RESET}")
            return
        except OSError as e:
            print(f"  {C.RED}Cannot read file: {e}{C.RESET}")
            return
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
            print(f"  [{i}/{len(queries)}] {query[:60]}... -> {len(selected)} fragments, {total_tokens} tokens")

    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n  {C.GREEN}{C.BOLD}Processed {len(queries)} queries.{C.RESET}\n")


# ── Wrap: One-command agent launcher ─────────────────────────────────


_WRAP_AGENTS = {
    "claude": {
        "cmd": ["claude"],
        "env_key": "ANTHROPIC_BASE_URL",
        "env_val": "http://localhost:{port}",
        "name": "Claude Code",
    },
    "codex": {
        "cmd": ["codex"],
        "env_key": "OPENAI_BASE_URL",
        "env_val": "http://localhost:{port}/v1",
        "name": "OpenAI Codex CLI",
    },
    "aider": {
        "cmd": ["aider"],
        "env_key": "OPENAI_API_BASE",
        "env_val": "http://localhost:{port}/v1",
        "name": "Aider",
    },
    "cursor": {
        "cmd": None,
        "env_key": None,
        "env_val": None,
        "name": "Cursor",
    },
}


def cmd_wrap(args):
    """entroly wrap <agent> — start proxy + launch agent in one command.

    Starts the Entroly proxy as a daemon, sets the agent's base URL
    env var, and launches it. Zero-config context optimization.
    """
    agent = args.agent.lower()
    if agent not in _WRAP_AGENTS:
        print(f"\n  {C.RED}Unknown agent: {agent}{C.RESET}")
        print(f"  {C.GRAY}Supported: {', '.join(_WRAP_AGENTS.keys())}{C.RESET}\n")
        return

    spec = _WRAP_AGENTS[agent]
    port = args.port or 9377

    print(f"\n{C.CYAN}{C.BOLD}  Entroly Wrap — {spec['name']}{C.RESET}\n")

    # Check if proxy is already running
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        proxy_running = s.connect_ex(("127.0.0.1", port)) == 0

    if not proxy_running:
        print(f"  {C.GRAY}Starting proxy on port {port}...{C.RESET}")
        proxy_cmd = [sys.executable, "-m", "entroly", "proxy", "--port", str(port)]
        subprocess.Popen(
            proxy_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time as _time
        started = False
        for _ in range(30):
            _time.sleep(0.2)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    started = True
                    break
        if not started:
            print(f"  {C.RED}Proxy failed to start on port {port} within 6s.{C.RESET}")
            print(f"  {C.GRAY}Try: entroly proxy --port {port} in another terminal to see the error.{C.RESET}\n")
            return
        print(f"  {C.GREEN}Proxy running at http://localhost:{port}{C.RESET}")
    else:
        print(f"  {C.GREEN}Proxy already running at http://localhost:{port}{C.RESET}")

    if agent == "cursor":
        print(f"\n  {C.BOLD}Cursor Configuration:{C.RESET}")
        print("  Open Cursor Settings → Models → Override OpenAI Base URL:")
        print(f"    {C.CYAN}http://localhost:{port}/v1{C.RESET}")
        print(f"\n  {C.GRAY}All Cursor requests will be automatically optimized.{C.RESET}\n")
        return

    env = os.environ.copy()
    env[spec["env_key"]] = spec["env_val"].format(port=port)
    print(f"  {C.GRAY}Set {spec['env_key']}={spec['env_val'].format(port=port)}{C.RESET}")
    print(f"  {C.GREEN}Launching {spec['name']}...{C.RESET}\n")

    try:
        agent_cmd = spec["cmd"] + args.agent_args
        subprocess.run(agent_cmd, env=env)
    except FileNotFoundError:
        print(f"\n  {C.RED}{spec['name']} not found.{C.RESET}")
        print(f"  {C.GRAY}Install it first, then run: entroly wrap {agent}{C.RESET}\n")
    except KeyboardInterrupt:
        print(f"\n  {C.GRAY}{spec['name']} stopped.{C.RESET}")


# ── Learn: Failure pattern analysis ──────────────────────────────────


def cmd_learn(args):
    """entroly learn — analyze session for failure patterns."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Learn — Failure Pattern Analysis{C.RESET}\n")

    import urllib.request
    port = args.port or 9377

    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=2)
        stats = json.loads(resp.read())
        feedback = stats.get("implicit_feedback", {})
    except Exception:
        feedback = {}

    if not feedback:
        print(f"  {C.YELLOW}No feedback data available.{C.RESET}")
        print(f"  {C.GRAY}Run the proxy and make some requests first.{C.RESET}\n")
        return

    confusion = feedback.get("confusion_detections", 0)
    confidence = feedback.get("confidence_detections", 0)
    rephrases = feedback.get("rephrase_detections", 0)
    total = feedback.get("total_assessed", 0)
    trend = feedback.get("quality_trend", "stable")

    print(f"  {C.BOLD}Session Analysis:{C.RESET}")
    print(f"    Total assessed:     {total}")
    print(f"    Confident:          {C.GREEN}{confidence}{C.RESET}")
    print(f"    Confused:           {C.RED if confusion > 0 else C.GRAY}{confusion}{C.RESET}")
    print(f"    Rephrases (retries):{C.YELLOW if rephrases > 0 else C.GRAY} {rephrases}{C.RESET}")
    print(f"    Quality trend:      {C.GREEN if trend == 'stable' else C.RED}{trend}{C.RESET}")

    if total > 0:
        success_rate = (confidence / total) * 100
        print(f"    Success rate:       {success_rate:.0f}%")

    # Learnings are thresholded observations from this session, not advice.
    # We write the raw counts so the user (or their agent) can decide what to do.
    learnings: list[str] = []
    if total > 5:
        if confusion > total * 0.3:
            learnings.append(
                f"Confusion on {confusion}/{total} requests ({confusion*100//total}%). "
                f"Above the 30% threshold — consider `entroly proxy --quality max` or increasing budget."
            )
        if rephrases > total * 0.2:
            learnings.append(
                f"Rephrases detected on {rephrases}/{total} requests ({rephrases*100//total}%). "
                f"Above the 20% threshold — inspect dropped fragments at /explain."
            )
        if trend == "declining":
            learnings.append(
                f"Quality trend: declining (confident={confidence}/{total})."
            )
    if total <= 5:
        learnings.append(f"Only {total} requests assessed — need >5 for reliable signal.")

    print(f"\n  {C.BOLD}Observations:{C.RESET}")
    for msg in learnings or ["No thresholds crossed."]:
        print(f"    {C.YELLOW}• {msg}{C.RESET}")

    if getattr(args, "apply", False) and learnings and total > 5:
        for fname in ["CLAUDE.md", "AGENTS.md"]:
            fpath = Path.cwd() / fname
            if fpath.exists():
                existing = fpath.read_text(encoding="utf-8", errors="replace")
                if "## Entroly Learnings" not in existing:
                    import time as _time
                    stamp = _time.strftime("%Y-%m-%d")
                    section = f"\n\n## Entroly Learnings ({stamp}, n={total})\n\n"
                    section += "\n".join(f"- {msg}" for msg in learnings) + "\n"
                    fpath.write_text(existing + section, encoding="utf-8")
                    print(f"\n  {C.GREEN}Written learnings to {fname}{C.RESET}")
                else:
                    print(f"\n  {C.GRAY}{fname} already has learnings section — remove the old one to refresh.{C.RESET}")
                break

    print()


def _recommend_quality(project: dict, file_count: int) -> str:
    """Recommend a starting quality preset from project characteristics.

    The heuristic has an information-theoretic basis, not a measured one —
    treat it as a starting point and run `entroly autotune` to calibrate.

    Size thresholds (5000 / 2000 / 500 files) are unvalidated defaults that
    trade pipeline latency for selection fidelity.
    """
    lang = project.get("primary", "unknown")
    langs = project.get("languages", [])

    # Larger codebases blow the latency budget on deeper presets; favor speed.
    if file_count > 5000:
        return "speed"
    if file_count > 2000:
        return "fast"

    # Heterogeneous repos mix semantic distributions (e.g. C infra + TS UI);
    # a single retrieval pass over mixed embeddings benefits from more effort.
    if len(langs) >= 3:
        return "quality"

    # Typed signatures act as compressed semantic metadata: a Rust fn's
    # `-> Result<Session, AuthError>` encodes intent the retriever can use
    # without reading the body. Dynamic languages lack that, so retrieval
    # precision at a fixed budget improves more from deeper analysis.
    if lang in ("rust", "go", "java"):
        return "balanced"
    if lang in ("python", "javascript", "typescript"):
        return "quality" if file_count > 500 else "balanced"

    return "balanced"


def cmd_go(args):
    """entroly go — one command to rule them all: init + proxy + dashboard."""
    print(f"""
{C.CYAN}{C.BOLD}  ⚡ Entroly Go{C.RESET} — full setup in one command
""")

    # Step 1: Detect project
    project = _detect_project_type()
    langs = ', '.join(project['languages'])
    fw = project.get('frameworks', [])
    fw_str = f" + {', '.join(fw)}" if fw else ""
    print(f"  {C.GRAY}Project:{C.RESET}  {C.BOLD}{project['name']}{C.RESET} ({langs}{fw_str})")

    # Step 2: Auto-detect AI tools and write configs
    tools = _detect_ai_tool()
    if tools["tools"]:
        for tool in tools["tools"]:
            try:
                path = _write_config(tool)
                print(f"  {C.GREEN}Configured{C.RESET} {tool['name']} ({path})")
            except Exception as e:
                print(f"  {C.YELLOW}Skipped{C.RESET} {tool['name']}: {e}")
    else:
        print(f"  {C.GRAY}No AI tool detected -- proxy mode works with any tool{C.RESET}")

    # Step 3: Initialize engine + auto-index
    from entroly.auto_index import auto_index, start_incremental_watcher
    from entroly.proxy import create_proxy_app
    from entroly.proxy_config import ProxyConfig, resolve_quality
    from entroly.server import EntrolyEngine

    engine = EntrolyEngine()
    result = auto_index(engine, force=getattr(args, "force", False))

    file_count = 0
    if result["status"] == "indexed":
        file_count = result["files_indexed"]
        print(f"  {C.GREEN}Indexed {file_count} files ({result['total_tokens']:,} tokens) in {result['duration_s']}s{C.RESET}")
    elif result["status"] == "skipped":
        file_count = result.get("existing_fragments", 0)
        print(f"  {C.GRAY}Using persistent index ({file_count} fragments){C.RESET}")

    # Step 4: Smart quality recommendation
    config = ProxyConfig.from_env()
    recommended = _recommend_quality(project, file_count)
    quality_val = resolve_quality(getattr(args, "quality", None) or recommended)
    config.quality = quality_val
    config._apply_quality_dial(quality_val)
    if args.port:
        config.port = args.port

    print(f"  {C.CYAN}Quality:{C.RESET}  {recommended} (auto-detected for {file_count} files)")

    # Start file watcher
    start_incremental_watcher(engine)

    # Warm up engine
    engine.optimize_context(token_budget=128000, query="project overview")

    # Free ports from any stale entroly processes
    if not _free_port(config.port):
        print(f"  {C.RED}Port {config.port} is in use and could not be freed.{C.RESET}")
        return
    _free_port(9378)  # dashboard port

    # Start proxy + dashboard
    app = create_proxy_app(engine, config)

    print(f"""
  {C.GREEN}{C.BOLD}Ready!{C.RESET}

  {C.GREEN}Proxy:{C.RESET}      http://localhost:{config.port}/v1
  {C.GREEN}Dashboard:{C.RESET}  http://localhost:9378

  {C.BOLD}Point your AI tool's API base URL to the proxy URL above.{C.RESET}
  {C.GRAY}Every request: intercepted → optimized → forwarded. Live latency on the dashboard.{C.RESET}
  {C.GRAY}Press Ctrl+C to stop.{C.RESET}
""")

    try:
        import uvicorn
        uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    except ImportError:
        print(f"  {C.RED}uvicorn not installed. Install with: pip install uvicorn{C.RESET}")
    except KeyboardInterrupt:
        print(f"\n  {C.GRAY}Entroly stopped.{C.RESET}")


def cmd_demo(args):
    """entroly demo — quick-win demo mode: before/after comparison (Gap #41)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Demo{C.RESET} -- 3 sample queries, real measurements\n")

    from entroly.auto_index import auto_index
    from entroly.server import EntrolyEngine
    from entroly.value_tracker import estimate_cost

    engine = EntrolyEngine()
    result = auto_index(engine)

    if result["status"] == "indexed":
        files_indexed = result["files_indexed"]
        total_tokens_raw = result["total_tokens"]
        if files_indexed == 0:
            print(f"  {C.YELLOW}No files found to index.{C.RESET}")
            print("  Run this from a project directory with source files.\n")
            return
        print(f"  {C.GREEN}Indexed {files_indexed} files ({total_tokens_raw:,} tokens total){C.RESET}\n")
    elif result["status"] == "skipped":
        existing = result.get("existing_fragments", 0)
        if existing == 0:
            print(f"  {C.YELLOW}No files found to index.{C.RESET}")
            print("  Run this from a project directory with source files.\n")
            return
        if engine._use_rust:
            stats = engine._rust.stats()
            total_tokens_raw = stats.get("session", {}).get("total_tokens_tracked", 0)
        else:
            total_tokens_raw = getattr(engine, "_total_token_count", 0)
        files_indexed = existing
        print(f"  {C.GREEN}Using persistent index: {files_indexed} fragments ({total_tokens_raw:,} tokens total){C.RESET}\n")
    else:
        print(f"  {C.YELLOW}No files found to index.{C.RESET}")
        print("  Run this from a project directory with source files.\n")
        return

    # Smart quality recommendation
    project = _detect_project_type()
    recommended = _recommend_quality(project, files_indexed)
    print(f"  {C.GRAY}Recommended quality preset: {C.CYAN}{recommended}{C.GRAY} for this project{C.RESET}\n")

    sample_queries = [
        "How does the authentication flow work?",
        "Find and fix potential SQL injection vulnerabilities",
        "Explain the module structure and dependency graph",
    ]

    # Model cost estimates for dollar impact
    models = ["gpt-4o", "claude-sonnet-4", "gemini-2.5-pro"]

    budget = 4096
    # Honest baseline: a 32K "paste matching files until you fill the window" dump,
    # not the whole repo. Claiming savings vs. total_tokens_raw (7M+ for AutoGPT)
    # is theatrical — nobody sends their entire codebase.
    BASELINE_PER_QUERY = min(total_tokens_raw, 32_000)
    print(f"  {C.BOLD}Naive baseline:{C.RESET} dump ~{BASELINE_PER_QUERY:,} tokens of matching files per query")
    print(f"  {C.BOLD}With Entroly:{C.RESET} selected context for each query:\n")

    total_saved = 0
    for query in sample_queries:
        engine.advance_turn()
        opt = engine.optimize_context(token_budget=budget, query=query)
        selected = opt.get("selected_fragments", [])
        tokens_used = sum(f.get("token_count", 0) for f in selected)
        saved = max(0, BASELINE_PER_QUERY - tokens_used)
        total_saved += saved
        pct = (saved * 100) // max(BASELINE_PER_QUERY, 1)

        # Per-query cost estimate
        cost_gpt4o = estimate_cost(saved, "gpt-4o")

        # Show top selected files
        top_files = [f.get("source", f.get("id", "?")).split("/")[-1].split("\\")[-1]
                     for f in selected[:3]]
        top_str = ", ".join(top_files) if top_files else "none"

        print(f"    {C.CYAN}Q:{C.RESET} {query[:60]}")
        print(f"       {C.GREEN}{len(selected)} fragments, {tokens_used:,} tokens{C.RESET} "
              f"({C.BOLD}{pct}% reduction{C.RESET}, ~${cost_gpt4o:.4f} saved)")
        print(f"       {C.GRAY}Top files: {top_str}{C.RESET}\n")

    avg_pct = (total_saved * 100) // max(BASELINE_PER_QUERY * len(sample_queries), 1)

    # Show projected savings across popular models
    if avg_pct == 0 and total_tokens_raw < 4096:
        print(f"  {C.GREEN}{C.BOLD}Your entire codebase fits within the token budget!{C.RESET}")
        print(f"  {C.GRAY}Entroly shines on larger codebases (>4K tokens) where it selects{C.RESET}")
        print(f"  {C.GRAY}only relevant fragments instead of sending everything.{C.RESET}\n")
    else:
        print(f"  {C.GREEN}{C.BOLD}Average: {avg_pct}% fewer tokens per request{C.RESET}\n")
    per_query_saved = total_saved // max(len(sample_queries), 1)
    print(f"  {C.BOLD}Per-query savings{C.RESET} (today's input rates, {per_query_saved:,} tokens saved/query):")
    for model in models:
        per_query_cost = estimate_cost(per_query_saved, model)
        print(f"    {C.CYAN}{model:25s}{C.RESET} ${per_query_cost:.4f}/query")
    print(f"  {C.GRAY}Multiply by your actual request volume. We don't know what that is.{C.RESET}")

    print(f"""
  {C.GREEN}{C.BOLD}Get started:{C.RESET}
    {C.CYAN}entroly go{C.RESET}                One command: init + proxy + dashboard
    {C.CYAN}entroly proxy --quality {recommended}{C.RESET}  Start optimizing
""")



def cmd_share(args):
    """entroly share — generate a shareable Context Report Card."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Share{C.RESET} -- generate your Context Report Card\n")

    from entroly.auto_index import auto_index
    from entroly.server import EntrolyEngine
    from entroly.value_tracker import estimate_cost, get_tracker

    engine = EntrolyEngine()
    result = auto_index(engine)

    if result["status"] == "indexed":
        files_indexed = result["files_indexed"]
        total_tokens_raw = result["total_tokens"]
    elif result["status"] == "skipped":
        files_indexed = result.get("existing_fragments", 0)
        if engine._use_rust:
            stats = engine._rust.stats()
            total_tokens_raw = stats.get("session", {}).get("total_tokens_tracked", 0)
        else:
            total_tokens_raw = getattr(engine, "_total_token_count", 0)
    else:
        print(f"  {C.YELLOW}No files found. Run from a project directory.{C.RESET}\n")
        return

    if files_indexed == 0:
        print(f"  {C.YELLOW}No files found. Run from a project directory.{C.RESET}\n")
        return

    print(f"  {C.GREEN}Indexed {files_indexed} files ({total_tokens_raw:,} tokens){C.RESET}")

    # Run sample queries to compute real stats
    sample_queries = [
        "How does the authentication flow work?",
        "Find and fix potential SQL injection vulnerabilities",
        "Explain the module structure and dependency graph",
    ]

    # Honest baseline for *token reduction* (headline only): a 32K "paste matching
    # files until the window fills" dump, capped at what actually exists. Comparing
    # against total_tokens_raw (7M+ for AutoGPT) is marketing, not measurement.
    BUDGET = 4096
    BUDGET_RELAXED = 8192  # 2× for stability measurement
    BASELINE_PER_QUERY = min(total_tokens_raw, 32_000)

    # --- Context Score: geometric mean of three defined quantities per query ---
    # Stability(q) = Jaccard(selection@B, selection@2B). A well-ordered selector
    #                 puts its most important items first; doubling budget adds,
    #                 doesn't replace. Random selection → ~0.
    # Coverage(q)  = fraction of non-stopword query terms present in selected
    #                content. Sanity check: is the selection actually about q?
    # Respect(q)   = 1 iff tokens_used <= B. Budget is a hard contract.
    # ContextScore = 100 * (∏_q Stability·Coverage·Respect)^(1/(3N))
    # No floors, no arbitrary caps. Bad on any axis → low score.
    import re as _re
    _STOPWORDS = {"how","does","do","the","a","an","is","are","what","why","when",
                  "where","and","or","but","to","of","in","on","for","with","by",
                  "it","this","that","these","those","be","will","can","could",
                  "would","should","you","your","i","we","my","our"}
    def _coverage(q: str, frags: list) -> float:
        terms = {w.lower() for w in _re.findall(r"[a-zA-Z_]{3,}", q)
                 if w.lower() not in _STOPWORDS}
        if not terms:
            return 1.0
        blob = " ".join(f.get("content", "") for f in frags).lower()
        return sum(1 for t in terms if t in blob) / len(terms)
    def _ids(frags: list) -> set:
        return {f.get("id") or f.get("source", "") for f in frags}
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / max(len(a | b), 1)

    total_saved = 0
    query_results = []
    stabilities: list[float] = []
    coverages: list[float] = []
    respects: list[float] = []

    for query in sample_queries:
        engine.advance_turn()
        opt_b = engine.optimize_context(token_budget=BUDGET, query=query)
        sel_b = opt_b.get("selected_fragments", [])
        engine.advance_turn()
        opt_2b = engine.optimize_context(token_budget=BUDGET_RELAXED, query=query)
        sel_2b = opt_2b.get("selected_fragments", [])

        tokens_used = sum(f.get("token_count", 0) for f in sel_b)
        saved = max(0, BASELINE_PER_QUERY - tokens_used)
        total_saved += saved
        pct = (saved * 100) // max(BASELINE_PER_QUERY, 1)

        stabilities.append(_jaccard(_ids(sel_b), _ids(sel_2b)))
        coverages.append(_coverage(query, sel_b))
        respects.append(1.0 if tokens_used <= BUDGET else 0.0)

        query_results.append({
            "query": query,
            "fragments": len(sel_b),
            "tokens": tokens_used,
            "saved_pct": pct,
        })

    avg_pct = (total_saved * 100) // max(BASELINE_PER_QUERY * len(sample_queries), 1)

    # Geometric mean. A single 0 on any axis zeros the score — by design.
    import math as _math
    factors = stabilities + coverages + respects
    if any(f <= 0 for f in factors):
        context_score = 0
    else:
        log_sum = sum(_math.log(f) for f in factors)
        context_score = int(round(100 * _math.exp(log_sum / len(factors))))
    context_score = max(0, min(100, context_score))

    avg_stability = sum(stabilities) / len(stabilities)
    avg_coverage = sum(coverages) / len(coverages)
    avg_respect = sum(respects) / len(respects)
    # Per-query $ at today's GPT-4o rate — no fake 100-req/day multiplier.
    per_query_savings_usd = estimate_cost(total_saved // len(sample_queries), "gpt-4o")

    # Detect project name
    project_name = Path.cwd().name

    # Pull lifetime stats if available
    tracker = get_tracker()
    lifetime = tracker.get_lifetime()
    lifetime_tokens = lifetime.get("tokens_saved", 0)
    lifetime_cost = lifetime.get("cost_saved_usd", 0.0)
    lifetime_requests = lifetime.get("requests_optimized", 0)

    # Honest per-query tokens shown instead of "AI sees ALL".
    avg_selected_frags = sum(qr["fragments"] for qr in query_results) // max(len(query_results), 1)
    avg_tokens_used = sum(qr["tokens"] for qr in query_results) // max(len(query_results), 1)

    # Generate HTML report
    html = _generate_report_html(
        project_name=project_name,
        files=files_indexed,
        total_tokens=total_tokens_raw,
        avg_reduction=avg_pct,
        context_score=context_score,
        per_query_savings_usd=per_query_savings_usd,
        avg_stability=avg_stability,
        avg_coverage=avg_coverage,
        avg_respect=avg_respect,
        avg_selected_frags=avg_selected_frags,
        avg_tokens_used=avg_tokens_used,
        query_results=query_results,
        lifetime_tokens=lifetime_tokens,
        lifetime_cost=lifetime_cost,
        lifetime_requests=lifetime_requests,
    )

    out_path = Path(args.output) if hasattr(args, "output") and args.output else Path("entroly-report.html")
    out_path.write_text(html, encoding="utf-8")

    print(f"\n  {C.GREEN}{C.BOLD}Context Report Card{C.RESET}")
    print("  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  {C.BOLD}PROJECT:{C.RESET}  {project_name:<42s}│")
    print(f"  │  {C.BOLD}CONTEXT SCORE:{C.RESET}  {C.GREEN}{context_score}/100{C.RESET}  "
          f"{C.GRAY}(stab·cov·respect)^⅓{C.RESET}      │")
    print(f"  │    {C.GRAY}stability {avg_stability:.2f}  coverage {avg_coverage:.2f}  respect {avg_respect:.2f}{C.RESET}   │")
    print(f"  │  {C.BOLD}PER-QUERY:{C.RESET}  ~{avg_selected_frags} frags, {avg_tokens_used:,} tokens (from {files_indexed:,} files)")
    print(f"  │  {C.BOLD}TOKENS vs 32K DUMP:{C.RESET}  {C.GREEN}{avg_pct}%{C.RESET} smaller")
    print(f"  │  {C.BOLD}SAVED / QUERY:{C.RESET}  {C.GREEN}${per_query_savings_usd:.4f}{C.RESET} (GPT-4o, today)   │")
    print("  └─────────────────────────────────────────────────────┘")
    print(f"\n  {C.GREEN}Report saved:{C.RESET} {out_path.resolve()}")
    print(f"\n  {C.CYAN}Share it!{C.RESET} Post your Context Score on Twitter/LinkedIn.")
    print(f"  {C.GRAY}\"My codebase scores {context_score}/100 on Entroly. What's yours?\"{C.RESET}\n")


def _generate_report_html(
    project_name: str,
    files: int,
    total_tokens: int,
    avg_reduction: int,
    context_score: int,
    per_query_savings_usd: float,
    avg_stability: float,
    avg_coverage: float,
    avg_respect: float,
    avg_selected_frags: int,
    avg_tokens_used: int,
    query_results: list,
    lifetime_tokens: int = 0,
    lifetime_cost: float = 0.0,
    lifetime_requests: int = 0,
) -> str:
    """Generate a beautiful, shareable HTML report card."""
    # Build query rows
    query_rows = ""
    for qr in query_results:
        query_rows += f"""
        <div class="qr">
          <div class="qr-query">{qr['query']}</div>
          <div class="qr-stats">
            <span class="qr-frags">{qr['fragments']} fragments</span>
            <span class="qr-tokens">{qr['tokens']:,} tokens</span>
            <span class="qr-saved">{qr['saved_pct']}% saved</span>
          </div>
        </div>"""

    # Score color
    if context_score >= 80:
        score_color = "#34d399"
    elif context_score >= 60:
        score_color = "#fbbf24"
    else:
        score_color = "#f87171"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entroly Context Report — {project_name}</title>
<meta name="description" content="{project_name} scores {context_score}/100 on Entroly Context Score. {avg_reduction}% token reduction, {files:,} files optimized.">
<meta property="og:title" content="{project_name} — Context Score {context_score}/100">
<meta property="og:description" content="{avg_reduction}% smaller than a 32K naive dump. {files:,} files indexed, ~{avg_selected_frags} selected per query. Powered by Entroly.">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{project_name} — Context Score {context_score}/100 | Entroly">
<meta name="twitter:description" content="Context Score {context_score}/100 (stability·coverage·respect)^⅓. ~{avg_tokens_used:,} tokens/query from {files:,} files.">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#09090b;color:#fafafa;font-family:'Inter',system-ui,sans-serif;display:flex;justify-content:center;align-items:center;min-height:100vh;padding:24px}}
.card{{background:#18181b;border:1px solid #27272a;border-radius:20px;max-width:560px;width:100%;overflow:hidden;box-shadow:0 25px 80px rgba(0,0,0,0.5)}}
.header{{background:linear-gradient(135deg,#312e81,#1e1b4b);padding:32px;text-align:center;position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle,rgba(129,140,248,0.1) 0%,transparent 50%);animation:pulse 4s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.05)}}}}
.logo{{font-weight:800;font-size:14px;letter-spacing:2px;text-transform:uppercase;color:#818cf8;margin-bottom:8px}}
.project{{font-size:24px;font-weight:800;letter-spacing:-0.5px;margin-bottom:4px}}
.subtitle{{color:#a1a1aa;font-size:13px}}
.score-section{{padding:32px;text-align:center;border-bottom:1px solid #27272a}}
.score-ring{{width:140px;height:140px;margin:0 auto 16px;position:relative}}
.score-ring svg{{transform:rotate(-90deg)}}
.score-ring .bg{{stroke:#27272a;fill:none;stroke-width:8}}
.score-ring .fg{{fill:none;stroke-width:8;stroke-linecap:round;stroke:{score_color};stroke-dasharray:{context_score * 4.4} 440;transition:stroke-dasharray 1.5s ease}}
.score-num{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:42px;font-weight:900;color:{score_color}}}
.score-label{{color:#a1a1aa;font-size:13px;font-weight:500}}
.stats{{display:grid;grid-template-columns:1fr 1fr 1fr;border-bottom:1px solid #27272a}}
.stat{{padding:20px;text-align:center;border-right:1px solid #27272a}}
.stat:last-child{{border-right:none}}
.stat-val{{font-size:22px;font-weight:800;color:#34d399}}
.stat-label{{font-size:11px;color:#a1a1aa;text-transform:uppercase;letter-spacing:1px;margin-top:4px}}
.queries{{padding:24px}}
.queries h3{{font-size:13px;color:#a1a1aa;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px}}
.qr{{background:#09090b;border:1px solid #27272a;border-radius:10px;padding:14px;margin-bottom:10px}}
.qr-query{{font-size:13px;font-weight:600;margin-bottom:8px}}
.qr-stats{{display:flex;gap:12px;font-size:12px}}
.qr-frags{{color:#818cf8}}
.qr-tokens{{color:#a1a1aa}}
.qr-saved{{color:#34d399;font-weight:700}}
.footer{{padding:20px 32px;text-align:center;border-top:1px solid #27272a}}
.footer p{{font-size:12px;color:#a1a1aa}}
.footer a{{color:#818cf8;text-decoration:none;font-weight:600}}
.cta{{display:inline-block;background:#818cf8;color:#fff;padding:10px 24px;border-radius:8px;font-weight:700;font-size:13px;text-decoration:none;margin-top:12px}}
.cta:hover{{opacity:0.9}}
.lifetime{{padding:16px 32px;border-bottom:1px solid #27272a;display:flex;justify-content:center;gap:24px;font-size:12px;color:#71717a}}
.lifetime strong{{color:#a1a1aa}}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="logo">Entroly Context Report</div>
    <div class="project">{project_name}</div>
    <div class="subtitle">Generated with entroly share</div>
  </div>

  <div class="score-section">
    <div class="score-ring">
      <svg viewBox="0 0 160 160" width="140" height="140">
        <circle class="bg" cx="80" cy="80" r="70"/>
        <circle class="fg" cx="80" cy="80" r="70"/>
      </svg>
      <div class="score-num">{context_score}</div>
    </div>
    <div class="score-label">CONTEXT SCORE</div>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-val">{avg_selected_frags:,}</div>
      <div class="stat-label">Frags / Query<br><span style="font-size:10px;text-transform:none">of {files:,} indexed</span></div>
    </div>
    <div class="stat">
      <div class="stat-val">{avg_tokens_used:,}</div>
      <div class="stat-label">Tokens / Query<br><span style="font-size:10px;text-transform:none">{avg_reduction}% &lt; 32K dump</span></div>
    </div>
    <div class="stat">
      <div class="stat-val">${per_query_savings_usd:.4f}</div>
      <div class="stat-label">Saved / Query<br><span style="font-size:10px;text-transform:none">GPT-4o, today</span></div>
    </div>
  </div>

  <div class="stats" style="grid-template-columns:1fr 1fr 1fr">
    <div class="stat">
      <div class="stat-val" style="color:#a1a1aa;font-size:18px">{avg_stability:.2f}</div>
      <div class="stat-label">Stability<br><span style="font-size:10px;text-transform:none">Jaccard(S@B, S@2B)</span></div>
    </div>
    <div class="stat">
      <div class="stat-val" style="color:#a1a1aa;font-size:18px">{avg_coverage:.2f}</div>
      <div class="stat-label">Coverage<br><span style="font-size:10px;text-transform:none">query terms hit</span></div>
    </div>
    <div class="stat">
      <div class="stat-val" style="color:#a1a1aa;font-size:18px">{avg_respect:.2f}</div>
      <div class="stat-label">Respect<br><span style="font-size:10px;text-transform:none">tokens ≤ budget</span></div>
    </div>
  </div>

  {"<div class='lifetime'><span>Lifetime: <strong>" + f"{lifetime_tokens:,}" + "</strong> tokens saved</span><span><strong>" + f"${lifetime_cost:,.2f}" + "</strong> saved</span><span><strong>" + f"{lifetime_requests:,}" + "</strong> requests</span></div>" if lifetime_requests > 0 else ""}

  <div class="queries">
    <h3>Sample Queries</h3>
    {query_rows}
  </div>

  <div class="footer">
    <p>My codebase scores <strong>{context_score}/100</strong> on Entroly. What's yours?</p>
    <a href="https://github.com/juyterman1000/entroly" class="cta">Get Your Score</a>
    <p style="margin-top:12px"><a href="https://github.com/juyterman1000/entroly">github.com/juyterman1000/entroly</a></p>
  </div>
</div>
</body>
</html>"""


def cmd_doctor(args):
    """entroly doctor — diagnose common issues (Gap #52)."""
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Doctor{C.RESET}\n")

    checks_passed = 0
    checks_total = 0

    # 1. Check Python version
    checks_total += 1
    py_ver = platform.python_version()
    if sys.version_info >= (3, 10):
        print(f"  {C.GREEN}+{C.RESET} Python {py_ver}")
        checks_passed += 1
    else:
        print(f"  {C.RED}x{C.RESET} Python {py_ver} (need >=3.10)")

    # 2. Check Rust engine
    checks_total += 1
    try:
        import entroly_core  # noqa: F401
        print(f"  {C.GREEN}+{C.RESET} Rust engine (entroly-core) loaded")
        checks_passed += 1
    except ImportError:
        print(f"  {C.RED}x{C.RESET} Rust engine not installed (pip install entroly-core)")

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
                print(f"  {C.GREEN}+{C.RESET} Config valid (weights sum={w_sum:.3f})")
                checks_passed += 1
            else:
                print(f"  {C.YELLOW}!{C.RESET} Config: weights sum={w_sum:.3f} (expected ~1.0)")
                checks_passed += 1  # warning, not failure
        except Exception as e:
            print(f"  {C.RED}x{C.RESET} Config error: {e}")
    else:
        print(f"  {C.GREEN}+{C.RESET} Config: using defaults (no tuning_config.json)")
        checks_passed += 1

    # 4. Check proxy reachability
    checks_total += 1
    port = getattr(args, "port", None) or 9377
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
        if resp.status == 200:
            print(f"  {C.GREEN}+{C.RESET} Proxy reachable at localhost:{port}")
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
                print(f"  {C.GREEN}+{C.RESET} Index fresh ({age_hours:.1f}h old)")
            else:
                print(f"  {C.YELLOW}!{C.RESET} Index stale ({age_hours:.0f}h old -- consider re-indexing)")
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
                print(f"  {C.GREEN}+{C.RESET} Weights near defaults (drift={drift:.3f})")
            elif drift < 0.3:
                print(f"  {C.YELLOW}!{C.RESET} Weights drifted (drift={drift:.3f})")
            else:
                print(f"  {C.RED}x{C.RESET} Weights heavily drifted ({drift:.3f}) -- consider autotune --rollback")
            checks_passed += 1
        except Exception:
            checks_passed += 1
    else:
        print(f"  {C.GREEN}+{C.RESET} Weights: defaults (no drift)")
        checks_passed += 1

    # 7. Check Docker (optional)
    checks_total += 1
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  {C.GREEN}+{C.RESET} Docker available")
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
        from entroly.value_tracker import estimate_cost
        est_cost = estimate_cost(saved, "gpt-4o")
        print(f"  {C.BOLD}Estimated cost saved:{C.RESET} ${est_cost:.2f} {C.GRAY}(at GPT-4o input rate){C.RESET}")
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
        print(f"  {C.GREEN}+{C.RESET} Already on version {current_version}. No migration needed.\n")
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
                print(f"  {C.GREEN}+{C.RESET} Migrated tuning_config.json (added missing sections)")
                migrated += 1
            else:
                print(f"  {C.GREEN}+{C.RESET} tuning_config.json: schema up to date")
        except Exception as e:
            print(f"  {C.RED}x{C.RESET} tuning_config.json error: {e}")
    else:
        print(f"  {C.GREEN}+{C.RESET} No tuning_config.json (using defaults)")

    # Check checkpoint format
    checkpoint_dir = entroly_dir / "checkpoints"
    if checkpoint_dir.exists():
        old_checkpoints = list(checkpoint_dir.glob("*.json"))
        gz_checkpoints = list(checkpoint_dir.glob("*.json.gz"))
        if old_checkpoints and not gz_checkpoints:
            print(f"  {C.YELLOW}!{C.RESET} Found {len(old_checkpoints)} uncompressed checkpoints")
            print(f"       Run {C.CYAN}entroly clean{C.RESET} + re-index for compressed format")
        else:
            print(f"  {C.GREEN}+{C.RESET} Checkpoints: format OK ({len(gz_checkpoints)} compressed)")
    else:
        print(f"  {C.GREEN}+{C.RESET} No checkpoints to migrate")

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

    # Starting-point presets. The rationale under each is an information-
    # theoretic argument, not a measurement — run `entroly autotune` after
    # applying to calibrate against bench/cases.json for your codebase.
    roles = {
        "frontend": {
            "description": "Frontend developer (React, Vue, CSS)",
            "weights": {"recency": 0.25, "frequency": 0.30, "semantic": 0.30, "entropy": 0.15},
            "note": "Component reuse is a repetition pattern — frequency captures it; "
                    "semantic similarity finds the right component class at retrieval.",
        },
        "backend": {
            "description": "Backend developer (API, database, services)",
            "weights": {"recency": 0.30, "frequency": 0.20, "semantic": 0.25, "entropy": 0.25},
            "note": "Backend work spans heterogeneous subsystems (HTTP, DB, auth, queues); "
                    "entropy diversifies selection across them rather than collapsing into one.",
        },
        "sre": {
            "description": "SRE / DevOps (infra, CI/CD, monitoring)",
            "weights": {"recency": 0.35, "frequency": 0.15, "semantic": 0.20, "entropy": 0.30},
            "note": "Infra state drifts fast (recency) and configs come from many distinct "
                    "sources — k8s, Terraform, GHA, Dockerfiles — (entropy).",
        },
        "data": {
            "description": "Data engineer / ML (SQL, pipelines, notebooks)",
            "weights": {"recency": 0.20, "frequency": 0.30, "semantic": 0.30, "entropy": 0.20},
            "note": "Pipelines reuse table/column/schema names heavily; frequency picks up "
                    "the repetition, semantic matches the right transform or model.",
        },
        "fullstack": {
            "description": "Full-stack developer (balanced across all areas)",
            "weights": {"recency": 0.25, "frequency": 0.25, "semantic": 0.25, "entropy": 0.25},
            "note": "Uniform priors when workload mix is unknown — maximum-entropy default.",
        },
    }

    # Support both `entroly role list` and `entroly role --preset backend`
    preset = getattr(args, "preset", None)
    if preset:
        # --preset is shorthand for "apply <name>"
        action = "apply"
        args.name = preset
    else:
        action = getattr(args, "role_action", "list")

    if action == "list":
        for name, info in roles.items():
            w = info["weights"]
            print(f"  {C.CYAN}{name:12s}{C.RESET} {info['description']}")
            print(f"               R={w['recency']:.2f}  F={w['frequency']:.2f}  "
                  f"S={w['semantic']:.2f}  E={w['entropy']:.2f}")
            print(f"               {C.GRAY}{info['note']}{C.RESET}\n")
        print(f"  {C.GRAY}Rationales are information-theoretic starting points, not measurements.{C.RESET}")
        print(f"  {C.GRAY}Run {C.CYAN}entroly autotune{C.GRAY} after applying to calibrate on your codebase.{C.RESET}")
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
        print(f"    {C.GRAY}{role['note']}{C.RESET}")
        print(f"    {C.GRAY}Run {C.CYAN}entroly autotune{C.GRAY} to calibrate on your codebase.{C.RESET}\n")


def cmd_completions(args):
    """entroly completions {bash|zsh|fish} — output shell completion script."""
    shell = args.shell
    commands = [
        "init", "go", "serve", "proxy", "dashboard", "health",
        "autotune", "benchmark", "status", "config", "clean",
        "telemetry", "export", "import", "drift", "profile",
        "batch", "wrap", "learn", "share", "demo",
        "doctor", "digest", "migrate", "role", "completions",
        "optimize", "feedback", "compile", "verify", "sync",
        "search", "docs", "finetune",
    ]
    cmd_list = " ".join(commands)

    if shell == "bash":
        print(f"""# entroly bash completion -- add to ~/.bashrc:
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
        print(f"""# entroly zsh completion -- add to ~/.zshrc:
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
        print(f"""# entroly fish completion -- save to ~/.config/fish/completions/entroly.fish
complete -c entroly -n '__fish_use_subcommand' -a '{cmd_list}' -d 'Entroly commands'
complete -c entroly -n '__fish_seen_subcommand_from proxy' -l port -d 'Proxy port'
complete -c entroly -n '__fish_seen_subcommand_from proxy' -l host -d 'Bind host'
complete -c entroly -n '__fish_seen_subcommand_from proxy' -l quality -d 'Quality 0-1'
complete -c entroly -n '__fish_seen_subcommand_from completions' -a 'bash zsh fish'""")
    else:
        print(f"Unknown shell: {shell}. Supported: bash, zsh, fish", file=sys.stderr)
        sys.exit(1)


def cmd_optimize(args):
    """entroly optimize — generate an optimized context snapshot for a specific task.

    Indexes the codebase and selects files under a token budget using the
    knapsack approximation (0/1 DP when feasible, density-greedy otherwise —
    not exact optimum; knapsack is NP-hard). Outputs a markdown snapshot
    suitable for injection into a subagent prompt.
    """
    from entroly.auto_index import auto_index
    from entroly.server import EntrolyEngine

    task = getattr(args, "task", "") or ""
    budget = getattr(args, "budget", 8192)
    output_format = getattr(args, "format", "markdown")
    quiet = getattr(args, "quiet", False)

    if not quiet:
        print(f"\n{C.CYAN}{C.BOLD}  Entroly Optimize{C.RESET}", file=sys.stderr)
        if task:
            print(f"  Task: {C.GREEN}{task}{C.RESET}", file=sys.stderr)
        print(f"  Budget: {budget:,} tokens\n", file=sys.stderr)

    engine = EntrolyEngine()
    result = auto_index(engine)

    if result["status"] == "indexed":
        files_indexed = result["files_indexed"]
        total_tokens = result["total_tokens"]
        if not quiet:
            print(f"  Indexed {C.GREEN}{files_indexed}{C.RESET} files ({total_tokens:,} tokens)\n", file=sys.stderr)
    elif result["status"] == "skipped":
        existing = result.get("existing_fragments", 0)
        if existing == 0:
            print(f"  {C.YELLOW}No files found.{C.RESET} Run from a project directory.", file=sys.stderr)
            return
        files_indexed = existing
        if not quiet:
            print(f"  Using persistent index: {C.GREEN}{files_indexed}{C.RESET} fragments\n", file=sys.stderr)
    else:
        print(f"  {C.YELLOW}No files found.{C.RESET} Run from a project directory.", file=sys.stderr)
        return

    engine.advance_turn()
    selector = getattr(args, "selector", "knapsack")
    if selector == "dopt":
        # Bayesian D-optimal re-selection over the full fragment store.
        # recall() caps at ~25 items and biases toward stale stubs; we bypass
        # it and run log-det greedy directly on export_fragments. See
        # entroly/dopt_selector.py for the objective and algorithm.
        from entroly.dopt_selector import select as dopt_select
        candidates = [dict(f) for f in engine._rust.export_fragments()]
        exclude_patterns = getattr(args, "exclude", []) or []
        if exclude_patterns:
            candidates = [
                c for c in candidates
                if not any(p in (c.get("source") or "") for p in exclude_patterns)
            ]
        selected = dopt_select(candidates, token_budget=budget, query=task)
        opt = {
            "selected_fragments": selected,
            "total_tokens": sum(f.get("token_count") or (len(f.get("content", "")) // 4) for f in selected),
            "recommended_budget": budget,
            "task_type": "",
        }
    else:
        opt = engine.optimize_context(token_budget=budget, query=task)
        selected = opt.get("selected_fragments", []) or opt.get("selected", [])

    # Deduplicate fragments by source path
    seen_sources = set()
    deduped = []
    for f in selected:
        src = f.get("source", "")
        if src not in seen_sources:
            seen_sources.add(src)
            deduped.append(f)
    selected = deduped
    # Source of truth for tokens_used is the engine's own accounting
    # (opt["total_tokens"]). It reflects the resolution-aware cost the
    # knapsack actually charged against the budget. Summing per-fragment
    # token_count in Python can drift by a handful of tokens because
    # Reference/Skeleton/Belief variants report different costs than the
    # knapsack internally charges.
    tokens_used = opt.get("total_tokens", sum(f.get("token_count", 0) for f in selected))
    recommended_budget = opt.get("recommended_budget", budget)
    task_type = opt.get("task_type", "")

    if not quiet:
        print(f"  Selected {C.GREEN}{len(selected)}{C.RESET} fragments ({tokens_used:,} tokens)\n", file=sys.stderr)
        if recommended_budget > budget and task_type:
            print(
                f"  {C.YELLOW}Note:{C.RESET} task classified as {C.CYAN}{task_type}{C.RESET} — "
                f"a wider budget of {C.GREEN}{recommended_budget:,}{C.RESET} is recommended for best coverage.\n"
                f"  Pass {C.GREEN}--budget {recommended_budget}{C.RESET} to opt in.\n",
                file=sys.stderr,
            )

    if output_format == "json":
        import json as _json
        output = {
            "task": task,
            "budget": budget,
            "tokens_used": tokens_used,
            "fragments": [
                {
                    "source": f.get("source", ""),
                    "token_count": f.get("token_count", 0),
                    "content": f.get("content", f.get("preview", "")),
                }
                for f in selected
            ],
        }
        print(_json.dumps(output, indent=2))
    else:
        # Markdown output — designed for injection into agent prompts
        lines = []
        lines.append("# Codebase Context Snapshot")
        lines.append("")
        if task:
            lines.append(f"**Task:** {task}")
        lines.append(f"**Files:** {len(selected)} | **Tokens:** {tokens_used:,} / {budget:,}")
        lines.append("")
        for frag in selected:
            source = frag.get("source", "unknown")
            tc = frag.get("token_count", 0)
            content = frag.get("content", "") or frag.get("preview", "")
            # If content is truncated (ends with ...), read full file from disk
            if content.endswith("...") and source:
                file_path = source.replace("file:", "") if source.startswith("file:") else source
                resolved = Path(file_path)
                if not resolved.is_absolute():
                    resolved = Path.cwd() / resolved
                try:
                    content = resolved.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass  # keep truncated preview as fallback
            # Detect language from file extension
            ext = source.rsplit(".", 1)[-1] if "." in source else ""
            lang_map = {
                "py": "python", "rs": "rust", "js": "javascript",
                "ts": "typescript", "go": "go", "rb": "ruby",
                "java": "java", "c": "c", "cpp": "cpp", "h": "c",
                "toml": "toml", "yaml": "yaml", "yml": "yaml",
                "json": "json", "md": "markdown", "sh": "bash",
            }
            lang = lang_map.get(ext, "")
            lines.append(f"## `{source}` ({tc} tokens)")
            lines.append(f"```{lang}")
            lines.append(content.rstrip())
            lines.append("```")
            lines.append("")
        print("\n".join(lines))

    # Save last optimization for feedback command
    state_file = Path.home() / ".entroly" / "last_optimize.json"
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(state_file, "w") as f:
            _json.dump({
                "fragment_ids": [fr.get("id", fr.get("fragment_id", "")) for fr in selected],
                "task": task,
                "tokens_used": tokens_used,
            }, f)
    except Exception:
        pass


def cmd_feedback(args):
    """entroly feedback — signal outcome quality to improve future context selection.

    After an agent completes a task using optimized context, run this to
    tell Entroly whether the context was helpful. Entroly uses this to
    adjust fragment relevance scores for future optimizations.
    """
    from entroly.server import EntrolyEngine

    score = getattr(args, "score", None)
    outcome = getattr(args, "outcome", None)
    if score is None and outcome is not None:
        score = {"success": 1.0, "good": 1.0,
                 "fail": 0.0, "failure": 0.0, "bad": 0.0,
                 "neutral": 0.5}.get(outcome)
    if score is None:
        print(f"  {C.RED}Provide --score (0.0-1.0) or --outcome (success|fail|neutral).{C.RESET}")
        return

    # Load the last optimization state
    state_file = Path.home() / ".entroly" / "last_optimize.json"
    if not state_file.exists():
        print(f"  {C.YELLOW}No previous optimization found.{C.RESET}")
        print(f"  Run {C.CYAN}entroly optimize --task \"...\" {C.RESET}first.")
        return

    try:
        with open(state_file) as f:
            state = json.load(f)
    except Exception:
        print(f"  {C.RED}Could not read last optimization state.{C.RESET}")
        return

    fragment_ids = state.get("fragment_ids", [])
    task = state.get("task", "")

    if not fragment_ids:
        print(f"  {C.YELLOW}No fragments to provide feedback for.{C.RESET}")
        return

    engine = EntrolyEngine()

    if score >= 0.7:
        engine.record_success(fragment_ids)
        icon = f"{C.GREEN}✓{C.RESET}"
        label = "positive"
    elif score <= 0.3:
        engine.record_failure(fragment_ids)
        icon = f"{C.RED}✗{C.RESET}"
        label = "negative"
    else:
        engine.record_reward(fragment_ids, score - 0.5)
        icon = f"{C.YELLOW}~{C.RESET}"
        label = "neutral"

    print(f"\n  {icon} Recorded {label} feedback (score={score:.1f}) for {len(fragment_ids)} fragments")
    if task:
        print(f"  Task: {C.GRAY}{task}{C.RESET}")
    print(f"  {C.GRAY}Entroly will adjust future context selections based on this signal.{C.RESET}\n")


def cmd_compile(args):
    """entroly compile -- compile source code into persistent belief artifacts.

    Scans a directory for source files, extracts code entities, resolves
    dependencies, and writes belief artifacts to the vault. This is the
    foundation of Cross-Session Memory: compile once, start warm forever.
    """
    import os

    from entroly.belief_compiler import BeliefCompiler
    from entroly.vault import VaultConfig, VaultManager

    target = getattr(args, "directory", None) or os.getcwd()
    max_files = getattr(args, "max_files", 0)

    vault_base = os.environ.get(
        "ENTROLY_VAULT",
        os.path.join(os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")), "vault"),
    )
    vault = VaultManager(VaultConfig(base_path=vault_base))
    vault.ensure_structure()
    compiler = BeliefCompiler(vault)

    print(f"\n  {C.CYAN}{C.BOLD}Compiling Beliefs{C.RESET}")
    print(f"  {C.GRAY}Source:  {target}{C.RESET}")
    print(f"  {C.GRAY}Vault:   {vault_base}{C.RESET}")
    cap_desc = "unlimited" if max_files <= 0 else str(max_files)
    print(f"  {C.GRAY}Max files: {cap_desc}{C.RESET}\n")

    result = compiler.compile_directory(target, max_files)
    if max_files > 0 and result.files_processed >= max_files:
        print(f"  {C.YELLOW}! Hit --max-files cap ({max_files}); some files not indexed. "
              f"Re-run with --max-files 0 for unlimited.{C.RESET}")

    print(f"  {C.GREEN}Files processed:{C.RESET}    {result.files_processed}")
    print(f"  {C.GREEN}Entities extracted:{C.RESET} {result.entities_extracted}")
    print(f"  {C.GREEN}Beliefs written:{C.RESET}    {result.beliefs_written}")
    if result.errors:
        print(f"  {C.YELLOW}Errors:{C.RESET}             {len(result.errors)}")
        for e in result.errors[:5]:
            print(f"    {C.GRAY}- {e}{C.RESET}")

    coverage = vault.coverage_index()
    print(f"\n  {C.CYAN}Vault coverage:{C.RESET} {coverage['total_beliefs']} beliefs")
    print(f"  {C.CYAN}Avg confidence:{C.RESET} {coverage['average_confidence']:.2f}")
    print(f"\n  {C.GREEN}Beliefs persisted.{C.RESET} Next session starts warm.\n")


def cmd_verify(args):
    """entroly verify -- run verification pass on all beliefs.

    Checks for staleness, contradictions, confidence divergence. Writes
    verification artifacts to vault/verification/. This promotes beliefs
    from 'inferred' to 'verified' or flags them as 'stale'.
    """
    import os

    from entroly.vault import VaultConfig, VaultManager
    from entroly.verification_engine import VerificationEngine

    vault_base = os.environ.get(
        "ENTROLY_VAULT",
        os.path.join(os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")), "vault"),
    )
    vault = VaultManager(VaultConfig(base_path=vault_base))
    vault.ensure_structure()
    verifier = VerificationEngine(vault, freshness_hours=24.0, min_confidence=0.5)

    print(f"\n  {C.CYAN}{C.BOLD}Verifying Beliefs{C.RESET}")
    print(f"  {C.GRAY}Vault: {vault_base}{C.RESET}\n")

    report = verifier.full_verification_pass()
    rd = report.to_dict()

    print(f"  {C.GREEN}Beliefs checked:{C.RESET}     {rd.get('total_beliefs_checked', 0)}")
    print(f"  {C.GREEN}Stale:{C.RESET}               {rd.get('stale_count', 0)}")
    print(f"  {C.GREEN}Contradictions:{C.RESET}      {rd.get('contradiction_count', 0)}")
    print(f"  {C.GREEN}Low confidence:{C.RESET}      {rd.get('low_confidence_count', 0)}")

    stale = rd.get("stale", [])
    if stale:
        print(f"\n  {C.YELLOW}Stale beliefs:{C.RESET}")
        for s in stale[:10]:
            ent = s.get("entity", "unknown") if isinstance(s, dict) else str(s)
            print(f"    - {ent}")

    print(f"\n  {C.GREEN}Verification artifacts written to vault/verification/{C.RESET}\n")


def cmd_sync(args):
    """entroly sync -- detect workspace changes and update beliefs.

    Scans for new/modified/deleted files since last sync, marks affected
    beliefs stale, recompiles changed files, and runs verification.
    This is the Change-Driven Pipeline (Flow 4).
    """
    import os

    from entroly.belief_compiler import BeliefCompiler
    from entroly.change_listener import WorkspaceChangeListener
    from entroly.change_pipeline import ChangePipeline
    from entroly.vault import VaultConfig, VaultManager
    from entroly.verification_engine import VerificationEngine

    target = getattr(args, "directory", None) or os.getcwd()
    max_files = getattr(args, "max_files", 100)
    force = getattr(args, "force", False)

    vault_base = os.environ.get(
        "ENTROLY_VAULT",
        os.path.join(os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")), "vault"),
    )
    vault = VaultManager(VaultConfig(base_path=vault_base))
    vault.ensure_structure()
    compiler = BeliefCompiler(vault)
    verifier = VerificationEngine(vault, freshness_hours=24.0, min_confidence=0.5)
    change_pipe = ChangePipeline(vault, verifier)

    listener = WorkspaceChangeListener(
        vault=vault, compiler=compiler, verifier=verifier,
        change_pipe=change_pipe, project_dir=target,
    )

    print(f"\n  {C.CYAN}{C.BOLD}Syncing Workspace{C.RESET}")
    print(f"  {C.GRAY}Source: {target}{C.RESET}")
    print(f"  {C.GRAY}Vault:  {vault_base}{C.RESET}\n")

    result = listener.scan_once(force=force, max_files=max_files)
    rd = result.to_dict()

    print(f"  {C.GREEN}Files scanned:{C.RESET}    {rd.get('scanned_files', 0)}")
    print(f"  {C.GREEN}Files changed:{C.RESET}    {len(rd.get('changed_files', []))}")
    print(f"  {C.GREEN}Beliefs written:{C.RESET}  {rd.get('beliefs_written', 0)}")
    print(f"  {C.GREEN}Beliefs staled:{C.RESET}   {len(rd.get('refresh_result', {}).get('updated_entities', []))}")
    print(f"\n  {C.GREEN}Workspace synchronized.{C.RESET}\n")


def cmd_search(args):
    """entroly search -- full-text TF-IDF search across vault beliefs.

    Uses the Rust search engine with entity-name boosting (3x), title
    boosting (2x), and body TF-IDF scoring. Returns ranked results
    with excerpts — far cheaper than dumping all beliefs.
    """
    import os

    query = " ".join(args.query)
    top_k = args.top_k

    vault_base = os.environ.get(
        "ENTROLY_VAULT",
        os.path.join(os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")), "vault"),
    )

    print(f"\n  {C.CYAN}{C.BOLD}Vault Search{C.RESET}")
    print(f"  {C.GRAY}Query: {query}{C.RESET}")
    print(f"  {C.GRAY}Vault: {vault_base}{C.RESET}\n")

    try:
        from entroly_core import CogOpsEngine
        engine = CogOpsEngine(vault_base)
        results = engine.vault_search(query, top_k)
        engine_name = "rust"
    except ImportError:
        # Python fallback — simple substring
        from pathlib import Path

        from entroly.vault import VaultConfig, VaultManager, _parse_frontmatter
        vault = VaultManager(VaultConfig(base_path=vault_base))
        vault.ensure_structure()
        beliefs_dir = Path(vault_base) / "beliefs"
        query_lower = query.lower()
        results = []
        for md in sorted(beliefs_dir.rglob("*.md")):
            content = md.read_text(encoding="utf-8", errors="replace")
            if query_lower in content.lower():
                fm = _parse_frontmatter(content) or {}
                results.append({
                    "entity": fm.get("entity", md.stem),
                    "confidence": float(fm.get("confidence", 0)),
                    "status": fm.get("status", ""),
                    "excerpt": "",
                    "score": 1.0,
                })
        results = results[:top_k]
        engine_name = "python"

    if not results:
        print(f"  {C.YELLOW}No results found.{C.RESET}\n")
        return

    for i, r in enumerate(results, 1):
        entity = r.get("entity", r.get("entity", "?"))
        score = r.get("score", 0)
        conf = r.get("confidence", 0)
        status = r.get("status", "?")
        excerpt = r.get("excerpt", "")
        print(f"  {C.BOLD}{i}. {entity}{C.RESET}  (score={score:.2f}  conf={conf:.2f}  status={status})")
        if excerpt:
            for line in excerpt.split("\n")[:3]:
                print(f"     {C.GRAY}{line}{C.RESET}")
        print()

    print(f"  {C.GRAY}{len(results)} results ({engine_name} engine){C.RESET}\n")


def cmd_docs(args):
    """entroly docs -- compile markdown documentation into belief artifacts.

    Ingests README.md, ARCHITECTURE.md, docs/, CONTRIBUTING.md etc. into
    the vault as documentation beliefs with confidence 0.80 (human-authored
    context ranks higher than machine-inferred code beliefs).
    """
    import os

    target = args.directory or os.getcwd()
    max_files = args.max_files
    vault_base = os.environ.get(
        "ENTROLY_VAULT",
        os.path.join(os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")), "vault"),
    )

    print(f"\n  {C.CYAN}{C.BOLD}Compiling Documentation{C.RESET}")
    print(f"  {C.GRAY}Source: {target}{C.RESET}")
    print(f"  {C.GRAY}Vault:  {vault_base}{C.RESET}\n")

    try:
        from entroly_core import CogOpsEngine
        engine = CogOpsEngine(vault_base)
        result = engine.compile_docs(target, max_files)
    except ImportError:
        print(f"  {C.RED}entroly_core not installed — docs compilation requires the Rust engine.{C.RESET}")
        print(f"  {C.GRAY}Install with: pip install entroly-core{C.RESET}\n")
        return

    print(f"  {C.GREEN}Docs found:{C.RESET}      {result.get('docs_found', 0)}")
    print(f"  {C.GREEN}Docs compiled:{C.RESET}   {result.get('docs_compiled', 0)}")
    entities = result.get("entities", [])
    if entities:
        print(f"  {C.GREEN}Entities:{C.RESET}")
        for e in entities:
            print(f"    {C.GRAY}- {e}{C.RESET}")
    print(f"\n  {C.GREEN}Documentation beliefs persisted.{C.RESET}\n")


def cmd_finetune(args):
    """entroly finetune -- export vault beliefs as JSONL training data.

    Generates instruction-following Q&A pairs from compiled beliefs for
    LLM finetuning. Filters by PRISM 5D scoring: only beliefs with
    confidence >= 0.5 and non-stale status are included. Output is
    OpenAI-compatible JSONL format.

    After finetuning, the model "knows" the codebase in its weights —
    zero tokens needed for context. The ultimate compression.
    """
    import os

    output = args.output
    vault_base = os.environ.get(
        "ENTROLY_VAULT",
        os.path.join(os.environ.get("ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")), "vault"),
    )

    print(f"\n  {C.CYAN}{C.BOLD}Exporting Training Data{C.RESET}")
    print(f"  {C.GRAY}Vault:  {vault_base}{C.RESET}")
    print(f"  {C.GRAY}Output: {output}{C.RESET}")
    print(f"  {C.GRAY}Filter: PRISM 5D (confidence >= 0.5, non-stale){C.RESET}\n")

    try:
        from entroly_core import CogOpsEngine
        engine = CogOpsEngine(vault_base)
        result = engine.export_training_data(output, "jsonl")
    except ImportError:
        print(f"  {C.RED}entroly_core not installed — training export requires the Rust engine.{C.RESET}")
        print(f"  {C.GRAY}Install with: pip install entroly-core{C.RESET}\n")
        return

    print(f"  {C.GREEN}Beliefs used:{C.RESET}     {result.get('beliefs_used', 0)}")
    print(f"  {C.GREEN}Beliefs skipped:{C.RESET}  {result.get('beliefs_skipped', 0)} (low confidence / stale)")
    print(f"  {C.GREEN}Training pairs:{C.RESET}   {result.get('training_pairs', 0)}")
    print(f"  {C.GREEN}Approx tokens:{C.RESET}    {result.get('total_tokens_approx', 0):,}")
    if result.get("training_pairs", 0) == 0:
        print(f"\n  {C.YELLOW}No training pairs written.{C.RESET} Run {C.CYAN}entroly compile{C.RESET} first to populate the vault.\n")
        return
    print(f"\n  {C.GREEN}Training data exported.{C.RESET}")
    print(f"  {C.GRAY}Use with: openai api fine_tuning.jobs.create -t {output}{C.RESET}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="entroly",
        description="\u26a1 Entroly \u2014 Information-theoretic context optimization for AI coding agents",
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
        "--bypass", action="store_true",
        help="Start in bypass mode (forward requests unmodified, no optimization)",
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
        "--selector", type=str, choices=["knapsack", "dopt"], default="knapsack",
        help="Selection objective: knapsack (linear score) or dopt (Bayesian D-optimal log-det)",
    )
    optimize_parser.add_argument(
        "--exclude", type=str, action="append", default=[],
        help="Substring to exclude from the fragment source path (repeatable; dopt selector only)",
    )

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

    # entroly benchmark
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run competitive benchmark: Entroly vs Raw vs Top-K",
    )
    benchmark_parser.add_argument(
        "--budget", type=int, default=4096,
        help="Token budget per query (default: 4096)",
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
        "action", choices=["on", "off", "status"], nargs="?", default="status",
        help="Enable, disable, or check telemetry status (default: status)",
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

    # entroly demo (Gap #41)
    subparsers.add_parser(
        "demo",
        help="Quick-win demo: before/after comparison showing token savings",
    )

    # entroly wrap
    wrap_parser = subparsers.add_parser(
        "wrap",
        help="Start proxy + launch coding agent in one command (claude, codex, aider, cursor)",
    )
    wrap_parser.add_argument(
        "agent", type=str,
        help="Agent to wrap: claude, codex, aider, cursor",
    )
    wrap_parser.add_argument(
        "--port", type=int, default=None,
        help="Proxy port (default: 9377)",
    )
    wrap_parser.add_argument(
        "agent_args", nargs=argparse.REMAINDER,
        help="Additional arguments passed to the agent",
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

    # entroly verify
    subparsers.add_parser(
        "verify",
        help="Run verification pass on all beliefs (staleness, contradictions)",
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

    args = parser.parse_args()

    # First-run welcome + update check (non-blocking)
    _check_first_run()
    if args.command not in (None, "completions"):
        _check_for_update()

    _dispatch = {
        "optimize": cmd_optimize,
        "feedback": cmd_feedback,
        "init": cmd_init,
        "serve": cmd_serve,
        "go": cmd_go,
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
        "compile": cmd_compile,
        "verify": cmd_verify,
        "sync": cmd_sync,
        "search": cmd_search,
        "docs": cmd_docs,
        "finetune": cmd_finetune,
        "wrap": cmd_wrap,
        "learn": cmd_learn,
        "share": cmd_share,
    }

    handler = _dispatch.get(args.command)
    rc = 0
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print(f"\n  {C.GRAY}Interrupted.{C.RESET}")
            rc = 130
        except Exception as e:
            print(f"\n  {C.RED}Error:{C.RESET} {e}", file=sys.stderr)
            rc = 1
    else:
        parser.print_help()

    # Flush and terminate immediately.
    #
    # We use os._exit() instead of sys.exit() because:
    #   1. The Rust engine (entroly_core via PyO3) may hold OS-level threads
    #      that Python's threading.join() cannot interrupt, causing sys.exit()
    #      to block indefinitely during interpreter shutdown.
    #   2. Python's logging shutdown flushes handlers synchronously; if the
    #      Rust engine's log sink is slow or blocked, this hangs.
    #   3. On Windows/PowerShell, sys.exit() raises SystemExit which can be
    #      misinterpreted as a failure when stderr has prior output.
    #
    # This is the standard pattern used by pip, poetry, and other CLI tools
    # that wrap native libraries.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(rc)


if __name__ == "__main__":
    main()
