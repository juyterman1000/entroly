from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "entroly/session_attach.py",
    'ATTACHMENT_CLIENTS = ("claude", "codex", "copilot", "kimi", "openclaw")',
    'ATTACHMENT_CLIENTS = ("claude", "codex", "copilot", "openclaw")',
)
replace_once(
    "entroly/session_attach.py",
    '''    if grant.client == "kimi":
        return (("kimi", "mcp", "add", "--transport", "stdio", "--env", source_env, name, "--", *server),)
''',
    "",
)
replace_once(
    "entroly/session_attach.py",
    '''    if grant.client == "kimi":
        return (("kimi", "mcp", "remove", name),)
''',
    "",
)

for relative in ("README.md", "docs/agent-compatibility.md"):
    path = ROOT / relative
    lines = path.read_text(encoding="utf-8").splitlines()
    filtered = [line for line in lines if "Kimi" not in line and "kimi" not in line]
    if len(filtered) == len(lines):
        raise RuntimeError(f"{relative}: no Kimi line found")
    path.write_text("\n".join(filtered) + "\n", encoding="utf-8")

path = ROOT / "tests/test_agent_compatibility.py"
text = path.read_text(encoding="utf-8")
pattern = re.compile(
    r"\n\ndef test_kimi_attachment_uses_official_stdio_mcp_shape\(tmp_path\):.*?"
    r"(?=\n\ndef test_wrap_splits_wrapper_options_from_client_arguments\()",
    re.DOTALL,
)
text, count = pattern.subn("", text, count=1)
if count != 1:
    raise RuntimeError(f"tests/test_agent_compatibility.py: expected one Kimi test, found {count}")
path.write_text(text, encoding="utf-8")

for relative in (
    ".github/agent_e2e_probe.py",
    ".github/finalize_agent_compatibility.py",
    ".github/finalize_agent_compatibility_lintfix.py",
    ".github/workflows/agent-compatibility-e2e-probe.yml",
    ".github/workflows/finalize-agent-compatibility-e2e-v2.yml",
    ".github/workflows/finalize-agent-compatibility-e2e.yml",
    "tests/e2e/kimi_mcp_real_client.py",
    ".github/remove_kimi.py",
    ".github/workflows/remove-kimi.yml",
):
    (ROOT / relative).unlink(missing_ok=True)
