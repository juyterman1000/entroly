from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one source block, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "entroly/session_attach.py",
    'ATTACHMENT_CLIENTS = ("claude", "codex", "openclaw")',
    'ATTACHMENT_CLIENTS = ("claude", "codex", "copilot", "openclaw")',
)

replace_once(
    "entroly/session_attach.py",
    '''    if grant.client == "codex":
        return (("codex", "mcp", "add", name, "--env", source_env, "--", *server),)
    if grant.client == "openclaw":''',
    '''    if grant.client == "codex":
        return (("codex", "mcp", "add", name, "--env", source_env, "--", *server),)
    if grant.client == "copilot":
        return (("copilot", "mcp", "add", name, "--env", source_env, "--", *server),)
    if grant.client == "openclaw":''',
)

replace_once(
    "entroly/session_attach.py",
    '''    if grant.client == "codex":
        return (("codex", "mcp", "remove", name),)
    if grant.client == "openclaw":''',
    '''    if grant.client == "codex":
        return (("codex", "mcp", "remove", name),)
    if grant.client == "copilot":
        return (("copilot", "mcp", "remove", name),)
    if grant.client == "openclaw":''',
)

replace_once(
    "entroly/cli.py",
    '''    "aider": {
        "kind": "cli", "name": "Aider",
        "cmd": ["aider"],
        "env_key": "OPENAI_API_BASE",
        "env_val": "http://localhost:{port}/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "copilot": {
        "kind": "cli", "name": "GitHub Copilot CLI",
        "cmd": ["github-copilot-cli"],
        "env_key": "OPENAI_BASE_URL",
        "env_val": "http://localhost:{port}/v1",
    },''',
    '''    "aider": {
        "kind": "cli", "name": "Aider",
        "cmd": ["aider"],
        "env_key": "OPENAI_API_BASE",
        "env_val": "http://localhost:{port}/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "goose": {
        "kind": "cli", "name": "Goose",
        "cmd": ["goose"],
        "env_key": "OPENAI_HOST",
        "env_val": "http://localhost:{port}",
        "extra_env": {"OPENAI_BASE_PATH": "v1/chat/completions"},
        "api_key_env": "OPENAI_API_KEY",
    },
    "openhands": {
        "kind": "cli", "name": "OpenHands CLI",
        "cmd": ["openhands", "--override-with-envs"],
        "env_key": "LLM_BASE_URL",
        "env_val": "http://localhost:{port}/v1",
        "api_key_env": "LLM_API_KEY",
    },
    "copilot": {
        "kind": "cli", "name": "GitHub Copilot CLI (BYOK custom provider)",
        "cmd": ["copilot"],
        "env_key": "COPILOT_PROVIDER_BASE_URL",
        "env_val": "http://localhost:{port}/v1",
        "extra_env": {"COPILOT_PROVIDER_TYPE": "openai"},
        "api_key_env": "COPILOT_PROVIDER_API_KEY",
        "subscription_alt": "entroly attach create --client copilot --project . --ttl 4h --install",
    },''',
)

replace_once(
    "entroly/cli.py",
    '''    # ══════════════════════════════════════════════════════════════════
    "cline": {''',
    '''    # ══════════════════════════════════════════════════════════════════
    "grok": {
        "kind": "print", "name": "Grok CLI",
        "config_label": "~/.grok/config.toml",
        "key_path": "[model.entroly] custom model",
        "url": "http://localhost:{port}/v1",
        "snippet_toml": "[model.entroly]\\nmodel = \\\"grok-4.5\\\"\\nbase_url = \\\"http://localhost:{port}/v1\\\"\\nname = \\\"Grok via Entroly\\\"\\nenv_key = \\\"XAI_API_KEY\\\"\\napi_backend = \\\"chat_completions\\\"",
        "post_hint": "Start Entroly with ENTROLY_OPENAI_BASE set to the intended xAI/OpenAI-compatible upstream. Default signed-in Grok inference is not intercepted by this custom-model route.",
    },
    "vibe": {
        "kind": "print", "name": "Mistral Vibe",
        "config_label": "./.vibe/config.toml or ~/.vibe/config.toml",
        "key_path": "[[providers]] and [[models]]",
        "url": "http://localhost:{port}/v1",
        "snippet_toml": "active_model = \\\"entroly-model\\\"\\n\\n[[providers]]\\nname = \\\"entroly\\\"\\napi_base = \\\"http://localhost:{port}/v1\\\"\\napi_key_env_var = \\\"OPENAI_API_KEY\\\"\\napi_style = \\\"openai\\\"\\nbackend = \\\"generic\\\"\\n\\n[[models]]\\nname = \\\"gpt-4o\\\"\\nprovider = \\\"entroly\\\"\\nalias = \\\"entroly-model\\\"",
    },
    "omp": {
        "kind": "print", "name": "Oh My Pi",
        "config_label": "~/.omp/agent/models.yml",
        "key_path": "providers.entroly",
        "url": "http://localhost:{port}/v1",
        "snippet_yaml": "providers:\\n  entroly:\\n    baseUrl: http://localhost:{port}/v1\\n    api: openai-completions\\n    apiKey: OPENAI_API_KEY\\n    models:\\n      - id: gpt-4o\\n        name: GPT-4o via Entroly\\n        contextWindow: 128000\\n        maxTokens: 16384",
        "post_hint": "Choose the model/provider that matches the upstream configured behind Entroly. Do not assume an existing OAuth credential is valid for a custom proxy.",
    },
    "zcode": {
        "kind": "print", "name": "ZCode",
        "config_label": "ZCode provider settings",
        "key_path": "Custom OpenAI-compatible Base URL",
        "url": "http://localhost:{port}/v1",
    },
    "cline": {''',
)

replace_once(
    "entroly/cli.py",
    '''    print(
        f"\\n  {C.GRAY}If your installed {spec['name']} supports this custom endpoint, "
        f"requests sent through that endpoint should route through Entroly.{C.RESET}\\n"
    )


def _start_proxy_if_needed(port: int) -> bool:''',
    '''    print(
        f"\\n  {C.GRAY}If your installed {spec['name']} supports this custom endpoint, "
        f"requests sent through that endpoint should route through Entroly.{C.RESET}\\n"
    )
    post_hint = spec.get("post_hint")
    if post_hint:
        print(f"  {C.YELLOW}Boundary:{C.RESET} {post_hint.format(port=port)}\\n")


def _resolved_wrap_env(spec: dict, port: int) -> dict[str, str]:
    """Resolve the complete explicit environment contract for a CLI wrapper."""
    values = {spec["env_key"]: spec["env_val"].format(port=port)}
    for key, value in spec.get("extra_env", {}).items():
        values[str(key)] = str(value).format(port=port)
    return values


def _start_proxy_if_needed(port: int) -> bool:''',
)

replace_once(
    "entroly/cli.py",
    '_PREFLIGHT = {"codex": _codex_preflight}',
    '''def _copilot_preflight(port: int) -> list[str]:
    """Explain the BYOK boundary without claiming subscription interception."""
    del port
    if os.environ.get("COPILOT_MODEL"):
        return []
    return [
        "Copilot custom-provider mode needs COPILOT_MODEL to identify the model "
        "sent to the configured BYOK provider. Set it before launching, for "
        "example COPILOT_MODEL=gpt-4o. For a signed-in Copilot subscription, "
        "use the scoped MCP attachment instead; Entroly does not claim to "
        "intercept GitHub-hosted subscription inference."
    ]


_PREFLIGHT = {"codex": _codex_preflight, "copilot": _copilot_preflight}''',
)

replace_once(
    "entroly/cli.py",
    '''    if dry_run:
        launch = " ".join(spec["cmd"] + (args.agent_args or []))
        print(f"  {C.GRAY}[dry-run] would start the proxy on :{port}, set "
              f"{spec['env_key']}={spec['env_val'].format(port=port)}, and launch "
              f"`{launch}`. No changes made.{C.RESET}\\n")
        return 0
    if not _start_proxy_if_needed(port):
        return 1

    env = os.environ.copy()
    env[spec["env_key"]] = spec["env_val"].format(port=port)
    print(f"  {C.GRAY}Set {spec['env_key']}={spec['env_val'].format(port=port)}{C.RESET}")''',
    '''    if dry_run:
        resolved_env = _resolved_wrap_env(spec, port)
        env_text = ", ".join(f"{key}={value}" for key, value in resolved_env.items())
        launch = " ".join(spec["cmd"] + (args.agent_args or []))
        print(f"  {C.GRAY}[dry-run] would start the proxy on :{port}, set "
              f"{env_text}, and launch "
              f"`{launch}`. No changes made.{C.RESET}\\n")
        return 0
    if not _start_proxy_if_needed(port):
        return 1

    env = os.environ.copy()
    resolved_env = _resolved_wrap_env(spec, port)
    env.update(resolved_env)
    for key, value in resolved_env.items():
        print(f"  {C.GRAY}Set {key}={value}{C.RESET}")''',
)

replace_once(
    "README.md",
    '''`entroly wrap <agent>` picks the best integration for each tool — proxy env-wrap for CLIs, auto-merged `mcp.json` for MCP-aware IDEs, or a best-effort endpoint/config hint.

**Wrap in one command:** `claude` · `cursor` · `codex` · `aider` · `gemini` · `windsurf` · `vscode` · `zed` · `cline` · `continue` and **28 more**.

<details>
<summary><b>Full agent list (38 targets)</b></summary>

| Type | Agents |
|---|---|
| **CLI (env-wrap + exec)** | Claude Code, Codex CLI, Aider, Gemini CLI, Qwen Code, OpenCode, Charm CRUSH, Hermes, Pi, Ollama |
| **MCP IDEs (auto-merge `mcp.json`)** | Cursor, Windsurf, VS Code, Claude Desktop, Claude Code (MCP), Zed |
| **Copy-paste endpoint** | Cline, Roo Code, Continue, Cody, Amp, Kiro, Qoder, Trae, Antigravity, Amazon Q, Verdent, JetBrains AI, Helix, Tabby, Twinny, Sublime, Emacs, Neovim, Fitten, Tabnine, Supermaven |

Any tool that supports a custom `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` works via the proxy. Run `entroly wrap` (no agent) for the full grouped list. Use wrappers only with tools whose terms permit local proxies / custom endpoints.
</details>''',
    '''`entroly wrap <agent>` chooses the safest available integration: a session-scoped proxy launch, an MCP registration, or guided custom-endpoint setup when a third-party schema should not be mutated automatically.

### Agent compatibility

**Status describes integration depth—not a blanket quality or savings guarantee.** Provider-observed savings require requests to traverse an Entroly proxy route. MCP integrations add context, recovery, receipt, and verification tools but do not automatically intercept every model request.

| Agent or platform | Entroly path | Current status | Important boundary |
|---|---|---|---|
| **Claude Code** | Scoped MCP attachment; API-key proxy | **Native** | Claude Pro/Max subscription sessions use MCP; public-API proxying requires `ANTHROPIC_API_KEY`. |
| **Codex CLI** | Scoped MCP attachment; API-key proxy | **Native** | ChatGPT-account mode can bypass `OPENAI_BASE_URL`. |
| **GitHub Copilot CLI** | MCP for subscription sessions; BYOK custom-provider proxy | **Supported with mode boundary** | Entroly does not claim interception of GitHub-hosted subscription inference. |
| **OpenClaw** | ContextEngine plugin and scoped MCP attachment | **Native** | OpenClaw retains provider authentication; Entroly controls context assembly and receipts. |
| **Cursor** | Automatic project MCP config; optional custom endpoint | **Automatic MCP** | Proxy accounting exists only when the model route points through Entroly. |
| **Aider / OpenCode** | Session-scoped OpenAI-compatible proxy | **One command** | Requires a provider route that accepts a custom endpoint. |
| **Cline / Continue** | Printed endpoint or provider configuration | **Guided setup** | Entroly avoids silently mutating versioned extension schemas. |
| **Grok CLI** | Custom model pointed at Entroly | **Guided BYOK** | Default signed-in inference is not claimed as intercepted. |
| **Goose / OpenHands** | Documented custom endpoint | **Validation pending** | Added to the registry only with explicit auth boundaries and watchdog verification. |
| **Mistral Vibe / Oh My Pi / ZCode** | Generated custom-provider configuration | **Guided setup** | The user chooses the upstream model and credential contract. |
| **Kimi CLI** | Native MCP registration | **MCP-compatible** | OAuth inference passthrough is not claimed until independently tested. |
| **Cortex Code** | SDK/library boundary only | **Not validated as a wrap target** | No official tested endpoint contract is currently advertised by Entroly. |

[See the evidence-bounded compatibility guide](docs/agent-compatibility.md), including Copilot subscription vs BYOK behavior and the exact meaning of each status.

<details>
<summary><b>Code-backed setup registry</b></summary>

| Integration class | Current targets |
|---|---|
| **CLI proxy launch** | Claude Code, Codex CLI, Aider, GitHub Copilot CLI BYOK, Gemini CLI, Qwen Code, OpenCode, Charm CRUSH, Hermes, Pi, Ollama, Goose, OpenHands |
| **Automatic MCP config** | Cursor, Windsurf, VS Code MCP clients, Claude Desktop, Claude Code MCP mode, Zed |
| **Guided endpoint setup** | Grok CLI, Mistral Vibe, Oh My Pi, ZCode, Cline, Roo Code, Continue, Cody, Amp, Kiro, Qoder, Trae, Antigravity, Amazon Q, Verdent, JetBrains AI, Helix, Tabby, Twinny, Sublime, Emacs, Neovim, Fitten Code, Tabnine, Supermaven |

Any OpenAI-compatible client can use the proxy only when it supports a custom base URL, the upstream is configured correctly, and its authentication terms permit local routing. Entroly's post-session watchdog reports when a wrapped CLI sends the proxy zero requests.
</details>''',
)

Path("tests/test_agent_compatibility.py").write_text(
    '''from pathlib import Path
from types import SimpleNamespace

from entroly.cli import _WRAP_AGENTS, _resolved_wrap_env
from entroly.session_attach import (
    ATTACHMENT_CLIENTS,
    attachment_install_commands,
    attachment_remove_commands,
)


def test_copilot_attachment_uses_current_cli_contract(tmp_path):
    grant = SimpleNamespace(
        grant_id="att_0123456789abcdef",
        client="copilot",
        project_root=str(tmp_path),
    )
    install = attachment_install_commands(grant, tmp_path, tmp_path / "token")
    assert install[0][:4] == ("copilot", "mcp", "add", "entroly-att_0123456789abcdef")
    assert "--env" in install[0]
    assert attachment_remove_commands(grant) == (
        ("copilot", "mcp", "remove", "entroly-att_0123456789abcdef"),
    )
    assert "copilot" in ATTACHMENT_CLIENTS


def test_copilot_wrapper_uses_byok_provider_variables():
    spec = _WRAP_AGENTS["copilot"]
    assert spec["cmd"] == ["copilot"]
    assert spec["api_key_env"] == "COPILOT_PROVIDER_API_KEY"
    assert spec["subscription_alt"].startswith("entroly attach create --client copilot")
    assert _resolved_wrap_env(spec, 9377) == {
        "COPILOT_PROVIDER_BASE_URL": "http://localhost:9377/v1",
        "COPILOT_PROVIDER_TYPE": "openai",
    }


def test_new_wrapper_statuses_are_explicit():
    assert _resolved_wrap_env(_WRAP_AGENTS["goose"], 9377)["OPENAI_BASE_PATH"] == "v1/chat/completions"
    assert _WRAP_AGENTS["openhands"]["cmd"] == ["openhands", "--override-with-envs"]
    for name in ("grok", "vibe", "omp", "zcode"):
        assert _WRAP_AGENTS[name]["kind"] == "print"


def test_public_matrix_keeps_subscription_and_validation_boundaries():
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/agent-compatibility.md").read_text(encoding="utf-8")
    assert "does not claim interception of GitHub-hosted subscription inference" in readme
    assert "Cortex Code" in readme and "Not validated as a wrap target" in readme
    assert "Provider-bound token and cost measurements exist only" in guide
    assert "MCP-only integrations" in guide
''',
    encoding="utf-8",
)
