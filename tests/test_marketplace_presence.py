from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from marketplace_presence import (  # noqa: E402
    Presence,
    probe_mcp_registry,
)

CANONICAL_REPOSITORY = "https://github.com/juyterman1000/entroly"


def _server(version: str, *, repository: str = CANONICAL_REPOSITORY, packages=None):
    return {
        "servers": [
            {
                "server": {
                    "name": "io.github.juyterman1000/entroly",
                    "version": version,
                    "repository": {"url": repository, "source": "github"},
                    "packages": packages
                    if packages is not None
                    else [
                        {"registryType": "pypi", "identifier": "entroly"},
                        {"registryType": "npm", "identifier": "entroly-mcp"},
                    ],
                }
            }
        ]
    }


def test_present_when_the_released_version_is_listed() -> None:
    probe = probe_mcp_registry("1.0.84", fetch=lambda url: _server("1.0.84"))

    assert probe.presence is Presence.PRESENT
    assert probe.channel == "mcp-registry"


def test_absent_when_only_an_older_version_is_listed() -> None:
    # The real skew this guards: the registry sat at 1.0.81 while the
    # repository was at 1.0.83.
    probe = probe_mcp_registry("1.0.84", fetch=lambda url: _server("1.0.81"))

    assert probe.presence is Presence.ABSENT


def test_absent_when_the_registry_returns_nothing() -> None:
    probe = probe_mcp_registry("1.0.84", fetch=lambda url: {"servers": []})

    assert probe.presence is Presence.ABSENT


def test_unknown_when_the_payload_shape_is_unrecognised() -> None:
    # Two API paths answered on 2026-09-06 (/v0 and /v0.1), so the schema is
    # moving. A shape change must not redden a healthy release.
    probe = probe_mcp_registry("1.0.84", fetch=lambda url: {"unexpected": True})

    assert probe.presence is Presence.UNKNOWN


def test_unknown_when_the_network_fails() -> None:
    def explode(url: str) -> dict:
        raise OSError("connection reset")

    probe = probe_mcp_registry("1.0.84", fetch=explode)

    assert probe.presence is Presence.UNKNOWN
    assert "connection reset" in probe.detail


def test_absent_when_the_listing_points_at_a_non_canonical_repository() -> None:
    # Reproduces the workflow's ownership assertion: a listing under our name
    # pointing somewhere else is a hijack, not a success.
    probe = probe_mcp_registry(
        "1.0.84",
        fetch=lambda url: _server("1.0.84", repository="https://github.com/someone/else"),
    )

    assert probe.presence is Presence.ABSENT
    assert "repository" in probe.detail


def test_absent_when_the_package_set_is_unexpected() -> None:
    # Reproduces the workflow's package-set assertion.
    probe = probe_mcp_registry(
        "1.0.84",
        fetch=lambda url: _server(
            "1.0.84",
            packages=[{"registryType": "pypi", "identifier": "entroly"}],
        ),
    )

    assert probe.presence is Presence.ABSENT
    assert "package" in probe.detail


from marketplace_presence import (  # noqa: E402
    BLOCKING,
    probe_claude_marketplace,
    probe_smithery,
)


def _marketplace(version: str) -> dict:
    return {
        "name": "entroly",
        "owner": {"name": "juyterman1000"},
        "plugins": [{"name": "entroly", "source": "./", "version": version}],
    }


def test_claude_marketplace_present_when_raw_manifest_lists_the_version() -> None:
    probe = probe_claude_marketplace("1.0.84", fetch=lambda url: _marketplace("1.0.84"))

    assert probe.presence is Presence.PRESENT
    assert probe.channel == "claude-marketplace"


def test_claude_marketplace_absent_when_the_manifest_lags() -> None:
    probe = probe_claude_marketplace("1.0.84", fetch=lambda url: _marketplace("1.0.83"))

    assert probe.presence is Presence.ABSENT


def test_claude_marketplace_unknown_when_the_manifest_is_unreachable() -> None:
    def explode(url: str) -> dict:
        raise OSError("404 not found")

    probe = probe_claude_marketplace("1.0.84", fetch=explode)

    assert probe.presence is Presence.UNKNOWN


def test_smithery_absent_when_the_server_page_is_missing() -> None:
    # The audited state on 2026-09-06: correct smithery.yaml, no server page.
    def missing(url: str) -> dict:
        raise OSError("HTTP Error 404: Not Found")

    probe = probe_smithery("1.0.84", fetch=missing)

    assert probe.presence is Presence.ABSENT


def test_smithery_is_advisory_so_it_cannot_block_a_release() -> None:
    # Smithery indexes from GitHub on its own schedule, so ABSENT there can
    # be nobody's fault and must never stop a release.
    assert "smithery" not in BLOCKING
    assert "claude-marketplace" in BLOCKING
