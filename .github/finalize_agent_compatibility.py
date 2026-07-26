from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one block, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def build_copilot_e2e() -> None:
    source = ROOT / ".github" / "copilot_composite_e2e_probe.py"
    target = ROOT / "tests" / "e2e" / "copilot_proxy_mcp_real_client.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = source.read_text(encoding="utf-8").splitlines()

    start = next(
        index
        for index in range(len(lines) - 1)
        if lines[index].strip() == '"-s",' and lines[index + 1].strip() == '"-p",'
    )
    end = next(
        index
        for index in range(start, len(lines))
        if lines[index].strip() == '"--no-remote-export",'
    )
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    lines[start : end + 1] = [
        f'{indent}"-s",',
        f'{indent}f"--allow-tool={{server_name}}",',
        f'{indent}"--no-color",',
        f'{indent}"--no-ask-user",',
        f'{indent}"--no-auto-update",',
        f'{indent}"--no-custom-instructions",',
        f'{indent}"--disable-builtin-mcps",',
        f'{indent}"--no-remote",',
        f'{indent}"--no-remote-export",',
        f'{indent}f"--prompt=Use the get_stats tool from MCP server {{server_name}} exactly once. After the tool succeeds, return exactly E2E_OK.",',
    ]

    text = "\n".join(lines) + "\n"
    text = text.replace(
        "from __future__ import annotations\n",
        '"""Credential-free real-client Copilot proxy + scoped MCP E2E."""\n\nfrom __future__ import annotations\n',
        1,
    )
    text = text.replace(
        'REPORT = Path("copilot-composite-e2e.json")\nLOG = Path("copilot-composite-e2e.log")',
        'REPORT = Path("copilot-proxy-mcp-e2e.json")\nLOG = Path("copilot-proxy-mcp-e2e.log")',
        1,
    )
    old_tail = '''    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    new_tail = '''    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
    if text.count(old_tail) != 1:
        raise RuntimeError("Copilot E2E main return contract changed")
    target.write_text(text.replace(old_tail, new_tail, 1), encoding="utf-8")


def downgrade_unvalidated_clients() -> None:
    replace_once(
        "entroly/cli.py",
        '''    "goose": {
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
''',
        "",
    )
    replace_once(
        "entroly/cli.py",
        '''    # ══════════════════════════════════════════════════════════════════
    "grok": {''',
        '''    # ══════════════════════════════════════════════════════════════════
    "goose": {
        "kind": "print", "name": "Goose",
        "config_label": "Goose provider configuration",
        "key_path": "OpenAI-compatible host and model",
        "url": "http://localhost:{port}/v1",
        "post_hint": "Set the provider and model required by your installed Goose version, then verify traffic with Entroly's watchdog. Automatic launch remains disabled until a real-client E2E contract is committed.",
    },
    "openhands": {
        "kind": "print", "name": "OpenHands CLI",
        "config_label": "OpenHands CLI environment or config",
        "key_path": "LLM_BASE_URL",
        "url": "http://localhost:{port}/v1",
        "post_hint": "This local URL works only when OpenHands runs on the same host. A remote sandbox cannot reach a laptop-local Entroly proxy without an explicit network route.",
    },
    "grok": {''',
    )


def update_tests() -> None:
    replace_once(
        "tests/test_agent_compatibility.py",
        '''def test_new_wrapper_statuses_are_explicit():
    assert _resolved_wrap_env(_WRAP_AGENTS["goose"], 9377)["OPENAI_BASE_PATH"] == "v1/chat/completions"
    assert _WRAP_AGENTS["openhands"]["cmd"] == ["openhands", "--override-with-envs"]
    for name in ("grok", "vibe", "omp", "zcode"):
        assert _WRAP_AGENTS[name]["kind"] == "print"
''',
        '''def test_new_wrapper_statuses_are_explicit():
    for name in ("goose", "openhands", "grok", "vibe", "omp", "zcode"):
        assert _WRAP_AGENTS[name]["kind"] == "print"
''',
    )


def update_readme() -> None:
    replace_once(
        "README.md",
        '| **GitHub Copilot CLI** | MCP for subscription sessions; BYOK custom-provider proxy | **Supported with mode boundary** | Entroly does not claim interception of GitHub-hosted subscription inference. |',
        '| **GitHub Copilot CLI** | Scoped MCP plus BYOK custom-provider proxy | **Real-client E2E validated** | Copilot 1.0.75 completed a two-turn proxy request, invoked scoped `get_stats`, returned its tool result, and finished `E2E_OK`. GitHub-hosted subscription inference is not claimed as proxied. |',
    )
    replace_once(
        "README.md",
        '| **Goose / OpenHands** | Documented custom endpoint | **Validation pending** | Added to the registry only with explicit auth boundaries and watchdog verification. |',
        '| **Goose / OpenHands** | Printed custom-endpoint guidance | **Validation pending** | Automatic launch is disabled until a real-client watchdog-backed E2E contract passes. |',
    )
    replace_once(
        "README.md",
        '| **Kimi CLI** | Scoped MCP attachment | **Native MCP** | Kimi keeps its model authentication; Entroly adds scoped context tools. OAuth inference passthrough is not claimed. |',
        '| **Kimi CLI** | Scoped MCP attachment | **Real-client E2E validated** | Kimi 1.49.0 started the expiring Entroly server and discovered the expected observe/context tools. Model OAuth remains owned by Kimi. |',
    )
    replace_once(
        "README.md",
        '[See the evidence-bounded compatibility guide](docs/agent-compatibility.md), including Copilot subscription vs BYOK behavior and the exact meaning of each status.\n',
        '[See the evidence-bounded compatibility guide](docs/agent-compatibility.md), including Copilot subscription vs BYOK behavior, pinned real-client evidence, and the exact meaning of each status. The credential-free [`Agent Compatibility E2E`](.github/workflows/agent-compatibility-e2e.yml) gate reruns the Copilot and Kimi proofs.\n',
    )
    replace_once(
        "README.md",
        '| **CLI proxy launch** | Claude Code, Codex CLI, Aider, GitHub Copilot CLI BYOK, Gemini CLI, Qwen Code, OpenCode, Charm CRUSH, Hermes, Pi, Ollama, Goose, OpenHands |',
        '| **CLI proxy launch** | Claude Code, Codex CLI, Aider, GitHub Copilot CLI BYOK, Gemini CLI, Qwen Code, OpenCode, Charm CRUSH, Hermes, Pi, Ollama |',
    )
    replace_once(
        "README.md",
        '| **Guided endpoint setup** | Grok CLI, Mistral Vibe, Oh My Pi, ZCode, Cline, Roo Code, Continue, Cody, Amp, Kiro, Qoder, Trae, Antigravity, Amazon Q, Verdent, JetBrains AI, Helix, Tabby, Twinny, Sublime, Emacs, Neovim, Fitten Code, Tabnine, Supermaven |',
        '| **Guided endpoint setup** | Goose, OpenHands, Grok CLI, Mistral Vibe, Oh My Pi, ZCode, Cline, Roo Code, Continue, Cody, Amp, Kiro, Qoder, Trae, Antigravity, Amazon Q, Verdent, JetBrains AI, Helix, Tabby, Twinny, Sublime, Emacs, Neovim, Fitten Code, Tabnine, Supermaven |',
    )


def update_guide() -> None:
    replace_once(
        "docs/agent-compatibility.md",
        '| GitHub Copilot CLI | MCP for subscription sessions; custom-provider proxy for BYOK | Supported with mode boundary | MCP works with the signed-in CLI. Entroly does not claim interception of GitHub-hosted subscription inference. |',
        '| GitHub Copilot CLI | Scoped MCP plus custom-provider proxy for BYOK | Real-client E2E validated | Copilot 1.0.75 sent two turns through Entroly, advertised scoped `get_stats`, executed it, returned the tool result, and completed `E2E_OK`. GitHub-hosted subscription inference is not claimed as proxied. |',
    )
    replace_once(
        "docs/agent-compatibility.md",
        '| Goose | OpenAI-compatible endpoint | Compatible; end-to-end validation pending | Do not label one-command support until a watchdog-backed request test passes. |',
        '| Goose | Printed OpenAI-compatible endpoint guidance | Validation pending | Automatic launch remains disabled until a watchdog-backed real-client request test passes. |',
    )
    replace_once(
        "docs/agent-compatibility.md",
        '| OpenHands | Local CLI `LLM_BASE_URL` route | Compatible; end-to-end validation pending | A remote/cloud sandbox cannot reach a laptop-local proxy. |',
        '| OpenHands | Printed `LLM_BASE_URL` guidance | Validation pending | Automatic launch remains disabled; a remote/cloud sandbox cannot reach a laptop-local proxy without an explicit route. |',
    )
    replace_once(
        "docs/agent-compatibility.md",
        '| Kimi CLI | Scoped MCP attachment | Native MCP | Kimi keeps its model authentication; Entroly adds scoped context tools. OAuth inference passthrough is not claimed. |',
        '| Kimi CLI | Scoped MCP attachment | Real-client E2E validated | Kimi 1.49.0 started the expiring Entroly server, connected successfully, and discovered the expected observe/context tools. Model OAuth remains owned by Kimi. |',
    )
    marker = "## GitHub Copilot CLI\n"
    evidence = '''## Reproducible real-client E2E evidence

The credential-free [`Agent Compatibility E2E`](../.github/workflows/agent-compatibility-e2e.yml) gate runs on Ubuntu with Python 3.12 and pins the tested client contracts.

| Client | Pinned version | Observed proof |
|---|---:|---|
| GitHub Copilot CLI | 1.0.75 | Two OpenAI-compatible chat-completion turns crossed the Entroly proxy; Copilot advertised the scoped `get_stats` MCP tool, executed it, sent the tool result on the second turn, and returned `E2E_OK`. |
| Kimi CLI | 1.49.0 | `kimi mcp test` started the expiring Entroly attachment, connected over stdio, and discovered the expected `get_stats`, `optimize_context`, `entroly_retrieve`, and `repo_file_map` tools. |

These tests use a local mock model upstream and temporary homes/state directories. They prove client routing and MCP execution without consuming hosted-model credentials. They do not prove downstream answer quality, savings percentages, or interception of vendor-hosted subscription inference.

'''
    target = ROOT / "docs" / "agent-compatibility.md"
    text = target.read_text(encoding="utf-8")
    if text.count(marker) != 1:
        raise RuntimeError("agent compatibility guide heading changed")
    target.write_text(text.replace(marker, evidence + marker, 1), encoding="utf-8")


def write_workflow() -> None:
    workflow = ROOT / ".github" / "workflows" / "agent-compatibility-e2e.yml"
    workflow.write_text(
        '''name: Agent Compatibility E2E

on:
  pull_request:
    paths:
      - "entroly/cli.py"
      - "entroly/session_attach.py"
      - "tests/e2e/**"
      - "tests/test_agent_compatibility.py"
      - "docs/agent-compatibility.md"
      - "README.md"
      - ".github/workflows/agent-compatibility-e2e.yml"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  real-clients:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - uses: actions/setup-node@v6
        with:
          node-version: "22"
      - name: Install Entroly and pinned clients
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e .
          python -m pip install "kimi-cli==1.49.0"
          npm install --global "@github/copilot@1.0.75"
          entroly --version
          kimi --version
          copilot --version
      - name: Validate Kimi scoped MCP
        env:
          ENTROLY_DISABLE_UPDATE_CHECK: "1"
        run: python tests/e2e/kimi_mcp_real_client.py
      - name: Validate Copilot proxy and scoped MCP round trip
        env:
          ENTROLY_DISABLE_UPDATE_CHECK: "1"
        run: python tests/e2e/copilot_proxy_mcp_real_client.py
      - name: Upload machine-readable evidence
        if: always()
        uses: actions/upload-artifact@v6
        with:
          name: agent-compatibility-e2e
          path: |
            kimi-mcp-e2e.json
            copilot-proxy-mcp-e2e.json
            copilot-proxy-mcp-e2e.log
          if-no-files-found: error
          retention-days: 14
''',
        encoding="utf-8",
    )


def remove_exploratory_files() -> None:
    paths = (
        ".github/agent_e2e_probe.py",
        ".github/copilot_composite_e2e_probe.py",
        ".github/workflows/agent-compatibility-e2e-probe.yml",
        ".github/workflows/copilot-argv-probe.yml",
        ".github/workflows/copilot-composite-e2e-probe.yml",
        ".github/workflows/copilot-composite-e2e-probe-v2.yml",
        ".github/workflows/copilot-composite-e2e-probe-v3.yml",
    )
    for relative in paths:
        (ROOT / relative).unlink(missing_ok=True)


def main() -> None:
    build_copilot_e2e()
    downgrade_unvalidated_clients()
    update_tests()
    update_readme()
    update_guide()
    write_workflow()
    remove_exploratory_files()


if __name__ == "__main__":
    main()
