"""`entroly go` must not tell Anthropic users to use a /v1 base URL.

It printed one proxy URL, http://localhost:9377/v1, and said "Point your AI
tool's API base URL to the proxy URL above". The Anthropic SDK and Claude
Code append /v1/messages to ANTHROPIC_BASE_URL, so the proxy received
/v1/v1/messages, forwarded it upstream, and every request came back 404 --
reproduced through the proxy app with a mocked provider. The dashboard's
first-run hint said the same thing.
"""

from __future__ import annotations

import re

import entroly.cli as cli
import entroly.dashboard as dashboard

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_go_names_the_right_base_url_per_client_family() -> None:
    text = ANSI.sub("", cli._proxy_base_url_guide(9377))

    assert "ANTHROPIC_BASE_URL=http://localhost:9377\n" in text + "\n"
    assert "ANTHROPIC_BASE_URL=http://localhost:9377/v1" not in text
    assert "OPENAI_BASE_URL=http://localhost:9377/v1" in text


def test_go_guide_follows_the_chosen_port() -> None:
    text = ANSI.sub("", cli._proxy_base_url_guide(9500))

    assert "localhost:9500" in text and "9377" not in text


def test_go_guide_matches_what_wrap_sets_for_claude_and_codex() -> None:
    text = ANSI.sub("", cli._proxy_base_url_guide(9377))
    for agent in ("claude", "codex"):
        spec = cli._WRAP_AGENTS[agent]
        expected = f"{spec['env_key']}={spec['env_val'].format(port=9377)}"
        assert expected in text, (agent, expected)


def test_dashboard_hint_does_not_send_anthropic_users_to_v1() -> None:
    with open(dashboard.__file__, encoding="utf-8") as handle:
        html = handle.read()

    assert "Point your AI tool's API base URL to <code>http://localhost:9377/v1</code>" not in html
    assert "ANTHROPIC_BASE_URL=http://localhost:9377</code>" in html
