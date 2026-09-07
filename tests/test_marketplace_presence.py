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
