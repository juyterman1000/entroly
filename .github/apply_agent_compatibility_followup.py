from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one follow-up block, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "entroly/session_attach.py",
    'ATTACHMENT_CLIENTS = ("claude", "codex", "copilot", "openclaw")',
    'ATTACHMENT_CLIENTS = ("claude", "codex", "copilot", "kimi", "openclaw")',
)

replace_once(
    "entroly/session_attach.py",
    '''    if grant.client == "copilot":
        return (("copilot", "mcp", "add", name, "--env", source_env, "--", *server),)
    if grant.client == "openclaw":''',
    '''    if grant.client == "copilot":
        return (("copilot", "mcp", "add", name, "--env", source_env, "--", *server),)
    if grant.client == "kimi":
        return (("kimi", "mcp", "add", "--transport", "stdio", "--env", source_env, name, "--", *server),)
    if grant.client == "openclaw":''',
)

replace_once(
    "entroly/session_attach.py",
    '''    if grant.client == "copilot":
        return (("copilot", "mcp", "remove", name),)
    if grant.client == "openclaw":''',
    '''    if grant.client == "copilot":
        return (("copilot", "mcp", "remove", name),)
    if grant.client == "kimi":
        return (("kimi", "mcp", "remove", name),)
    if grant.client == "openclaw":''',
)

replace_once(
    "entroly/cli.py",
    '''    if key_env and not os.environ.get(key_env) and not force_flag:
        print(f"  {C.GRAY}No {key_env} set — looks like a {spec['name']} subscription login.{C.RESET}\\n")
        alt = spec.get("subscription_alt")
        if alt:
            print(f"  {C.BOLD}Best setup for a subscription — the MCP integration:{C.RESET}")
            print(f"    {C.CYAN}{alt}{C.RESET}")
            print(f"  {C.GRAY}(Claude Code stays your client; Entroly adds its tools.){C.RESET}\\n")''',
    '''    if key_env and not os.environ.get(key_env) and not force_flag:
        print(f"  {C.GRAY}No {key_env} set — this proxy path has no explicit provider credential.{C.RESET}\\n")
        alt = spec.get("subscription_alt")
        if alt:
            print(f"  {C.BOLD}Best setup for a signed-in or subscription session — the MCP integration:{C.RESET}")
            print(f"    {C.CYAN}{alt}{C.RESET}")
            print(f"  {C.GRAY}(Your agent stays the client; Entroly adds scoped context tools.){C.RESET}\\n")''',
)

replace_once(
    "README.md",
    '| **Kimi CLI** | Native MCP registration | **MCP-compatible** | OAuth inference passthrough is not claimed until independently tested. |',
    '| **Kimi CLI** | Scoped MCP attachment | **Native MCP** | Kimi keeps its model authentication; Entroly adds scoped context tools. OAuth inference passthrough is not claimed. |',
)

replace_once(
    "docs/agent-compatibility.md",
    '| Kimi CLI | Native MCP registration; custom provider where configured | MCP-compatible | OAuth inference passthrough is not claimed until independently tested. |',
    '| Kimi CLI | Scoped MCP attachment | Native MCP | Kimi keeps its model authentication; Entroly adds scoped context tools. OAuth inference passthrough is not claimed. |',
)

path = Path("tests/test_agent_compatibility.py")
text = path.read_text(encoding="utf-8")
text += '''


def test_kimi_attachment_uses_official_stdio_mcp_shape(tmp_path):
    grant = SimpleNamespace(
        grant_id="att_fedcba9876543210",
        client="kimi",
        project_root=str(tmp_path),
    )
    install = attachment_install_commands(grant, tmp_path, tmp_path / "token")
    command = install[0]
    assert command[:5] == ("kimi", "mcp", "add", "--transport", "stdio")
    assert "--env" in command and "--" in command
    assert attachment_remove_commands(grant) == (
        ("kimi", "mcp", "remove", "entroly-att_fedcba9876543210"),
    )
    assert "kimi" in ATTACHMENT_CLIENTS
'''
path.write_text(text, encoding="utf-8")
