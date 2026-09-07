"""Is Entroly actually listed on the channels it claims to ship through?

Smithery held a correct `smithery.yaml` for months while its server page did
not exist, and nothing reported it. The one channel that stayed live is the
one a workflow verified. This module is that verification, generalised.

Three result states, not two. `/v0/servers` and `/v0.1/servers` both answered
on 2026-09-06, so the registry schema is moving; a checker that treats an
unrecognised payload as failure turns a green release red for a reason that
has nothing to do with Entroly. UNKNOWN is recorded and warned about. It
never silently passes and it never blocks.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Callable

CANONICAL_NAME = "io.github.juyterman1000/entroly"
CANONICAL_REPOSITORY = "https://github.com/juyterman1000/entroly"
EXPECTED_PACKAGES = {("pypi", "entroly"), ("npm", "entroly-mcp")}
USER_AGENT = "entroly-canonical-mcp-publisher"

Fetch = Callable[[str], dict]


class Presence(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Probe:
    channel: str
    presence: Presence
    detail: str


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def probe_mcp_registry(version: str, *, fetch: Fetch = _fetch_json) -> Probe:
    channel = "mcp-registry"
    url = (
        "https://registry.modelcontextprotocol.io/v0.1/servers?search="
        + urllib.parse.quote(CANONICAL_NAME, safe="")
    )

    try:
        payload = fetch(url)
    except Exception as error:  # noqa: BLE001 - any failure to reach is UNKNOWN
        return Probe(channel, Presence.UNKNOWN, f"could not reach registry: {error}")

    servers = payload.get("servers")
    if not isinstance(servers, list):
        return Probe(channel, Presence.UNKNOWN, "unrecognised payload: no server list")

    for item in servers:
        server = item.get("server", item)
        if server.get("name") != CANONICAL_NAME:
            continue
        if server.get("version") != version:
            continue

        repository = (server.get("repository") or {}).get("url")
        if repository != CANONICAL_REPOSITORY:
            return Probe(
                channel,
                Presence.ABSENT,
                f"listing claims a non-canonical repository: {repository!r}",
            )

        packages = {
            (package.get("registryType"), package.get("identifier"))
            for package in server.get("packages", [])
        }
        if packages != EXPECTED_PACKAGES:
            return Probe(
                channel,
                Presence.ABSENT,
                f"unexpected package set: {sorted(packages)!r}",
            )

        return Probe(channel, Presence.PRESENT, f"listed at {version}")

    return Probe(channel, Presence.ABSENT, f"not listed at {version}")


RAW_MARKETPLACE_URL = (
    "https://raw.githubusercontent.com/juyterman1000/entroly/main/"
    ".claude-plugin/marketplace.json"
)
SMITHERY_URL = "https://smithery.ai/server/@juyterman1000/entroly"


def probe_claude_marketplace(version: str, *, fetch: Fetch = _fetch_json) -> Probe:
    channel = "claude-marketplace"

    try:
        payload = fetch(RAW_MARKETPLACE_URL)
    except Exception as error:  # noqa: BLE001
        return Probe(channel, Presence.UNKNOWN, f"could not read manifest: {error}")

    plugins = payload.get("plugins")
    if not isinstance(plugins, list):
        return Probe(channel, Presence.UNKNOWN, "unrecognised manifest: no plugin list")

    for plugin in plugins:
        if plugin.get("name") != "entroly":
            continue
        if plugin.get("version") == version:
            return Probe(channel, Presence.PRESENT, f"listed at {version}")
        return Probe(
            channel,
            Presence.ABSENT,
            f"manifest on main declares {plugin.get('version')!r}, not {version!r}",
        )

    return Probe(channel, Presence.ABSENT, "manifest does not list the entroly plugin")


def _fetch_text(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return {"status": response.status}


def probe_smithery(version: str, *, fetch: Fetch = _fetch_text) -> Probe:
    channel = "smithery"

    try:
        payload = fetch(SMITHERY_URL)
    except Exception as error:  # noqa: BLE001
        # A 404 here is a real answer: the server page does not exist. That
        # was the audited state while smithery.yaml sat correct in the repo.
        if "404" in str(error):
            return Probe(channel, Presence.ABSENT, "server page does not exist")
        return Probe(channel, Presence.UNKNOWN, f"could not reach smithery: {error}")

    if payload.get("status") == 200:
        return Probe(channel, Presence.PRESENT, "server page exists")
    return Probe(channel, Presence.UNKNOWN, f"unexpected status: {payload.get('status')}")


PROBES: dict[str, Callable[..., Probe]] = {
    "mcp-registry": probe_mcp_registry,
    "claude-marketplace": probe_claude_marketplace,
    "smithery": probe_smithery,
}

# Channels Entroly publishes to block a release when absent. Channels that
# index on their own schedule cannot: Smithery pulls from GitHub whenever it
# chooses, so ABSENT there can be nobody's fault.
BLOCKING = {"mcp-registry", "claude-marketplace"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", default="all", choices=["all", *PROBES])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    names = list(PROBES) if args.channel == "all" else [args.channel]
    probes = [PROBES[name](args.version) for name in names]

    if args.json:
        print(json.dumps([probe.__dict__ for probe in probes], indent=2, default=str))
    else:
        for probe in probes:
            policy = "blocking" if probe.channel in BLOCKING else "advisory"
            print(f"{probe.channel} [{policy}]: {probe.presence.value} - {probe.detail}")

    failed = [
        probe
        for probe in probes
        if probe.presence is Presence.ABSENT and probe.channel in BLOCKING
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
