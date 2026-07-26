from __future__ import annotations

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
    "README.md",
    '''<h1 align="center">Entroly — The Open-Source Context OS for AI Agents</h1>

<p align="center"><b>Keep your agent. Give it a Context OS.</b><br>
The observability, governance, and decision layer for AI context.<br>
Entroly brings together what agents read, remember, trust, recover, spend, and learn—without replacing the model or agent runtime you already use.</p>

<p align="center">
  <sub>Integrates with Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider, MCP clients, and custom provider applications. Choose the supported setup path for your client.</sub>
</p>
''',
    '''<h1 align="center">Entroly — Drop-In Context Assurance to Lower AI Operational Cost</h1>

<p align="center"><b>Reduce unnecessary context without losing control of critical evidence.</b><br>
Entroly uses budgeted context selection, content-addressed recovery, and auditable receipts to lower provider-bound inference expenditure—without rewriting your codebase or agent architecture.</p>

<p align="center">
  <sub>Works through supported proxy, MCP, plugin, wrapper, and SDK paths with Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider, local models, and OpenAI/Anthropic-compatible applications.</sub>
</p>
''',
)

replace_once(
    "entroly/proxy.py",
    '''_CERT_ENV_VARS = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS")

# ── Privacy utilities ───────────────────────────────────────────────────
''',
    '''_CERT_ENV_VARS = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS")
_AUTH_SCHEMES_REQUIRING_CREDENTIALS = frozenset({"bearer", "basic", "token"})


def _has_meaningful_auth_value(name: str, value: str) -> bool:
    """Return whether an inbound auth header contains usable credentials.

    Some custom-provider clients emit an empty value such as ``Bearer `` when
    authentication is optional. Forwarding that malformed value causes HTTP
    clients to reject the request before it reaches an unauthenticated local
    upstream. Empty credentials are dropped; valid credentials are preserved.
    """
    normalized = str(value).strip()
    if not normalized:
        return False
    if name.lower() != "authorization":
        return True
    scheme, separator, credentials = normalized.partition(" ")
    if scheme.lower() in _AUTH_SCHEMES_REQUIRING_CREDENTIALS:
        return bool(separator and credentials.strip())
    return True


# ── Privacy utilities ───────────────────────────────────────────────────
''',
)

replace_once(
    "entroly/proxy.py",
    '''            if lower in capability.auth_headers:
                forward[name] = value
                continue
''',
    '''            if lower in capability.auth_headers:
                if _has_meaningful_auth_value(lower, value):
                    forward[name] = value
                continue
''',
)

replace_once(
    "tests/test_proxy_providers.py",
    '''        assert "content-length" not in out

    def test_openai_compatible_prefix_models_use_registry_context_window(self):
''',
    '''        assert "content-length" not in out

    @pytest.mark.parametrize(
        "authorization",
        ["", "   ", "Bearer", "Bearer ", "bearer\\t", "Basic"],
    )
    def test_empty_authorization_credentials_are_not_forwarded(self, authorization):
        proxy = PromptCompilerProxy(object(), ProxyConfig())

        out = proxy._build_headers(
            {"authorization": authorization, "x-api-key": ""},
            "openai",
        )

        assert "authorization" not in out
        assert "x-api-key" not in out
        assert out["Content-Type"] == "application/json"

    def test_valid_authorization_credentials_are_preserved(self):
        proxy = PromptCompilerProxy(object(), ProxyConfig())

        out = proxy._build_headers(
            {
                "authorization": "Bearer local-provider-token",
                "x-api-key": "provider-key",
            },
            "openai",
        )

        assert out["authorization"] == "Bearer local-provider-token"
        assert out["x-api-key"] == "provider-key"

    def test_openai_compatible_prefix_models_use_registry_context_window(self):
''',
)
