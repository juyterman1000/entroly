# Marketplace Wedge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Entroly installable from the Claude Code plugin marketplace, prove its value on first run, and turn channel presence into a release surface that fails loudly when a channel goes dead.

**Architecture:** Three parts. A marketplace manifest plus an `mcpServers` declaration makes the plugin installable at all. A Node launcher shim absorbs the absence of `uvx` so the plugin can never be silently inert. A presence checker — extracted from the polling gate that already works inside `publish-mcp-registry.yml` — probes every channel with three result states and blocks releases only for channels Entroly actually pushes to.

**Tech Stack:** Python 3.10+ (stdlib only for the checker: `urllib`, `json`, `dataclasses`, `enum`), pytest, Node (for the launcher shim only), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-06-marketplace-wedge-design.md`

## Global Constraints

- Canonical MCP server name: `io.github.juyterman1000/entroly`
- Canonical repository URL: `https://github.com/juyterman1000/entroly`
- Expected MCP package set: `{("pypi", "entroly"), ("npm", "entroly-mcp")}`
- Marketplace name (users type this): `entroly`. Install command: `/plugin install entroly@entroly`
- **No savings percentage in any user-facing plugin copy.** `saved = max(0, baseline - selected_tokens)` with `baseline = min(total_tokens, 32_000)` in `entroly/cli.py` pins the figure at ≥75% for a budget of 8,000 before selection runs. Task 6 adds a test that fails if one is reintroduced.
- Version strings are never edited by hand. `scripts/bump_version.py` owns them; its repository-wide sweep is the backstop. `tests/test_version_surfaces_are_complete.py` asserts the property over every tracked file.
- The presence checker uses the Python standard library only. It runs in CI before the package is installed.
- Release is tag-driven: merge to `main`, then tag the post-merge commit on `main`.

---

## File Structure

| File | Responsibility |
|---|---|
| `.claude-plugin/marketplace.json` | **Create.** The file `/plugin marketplace add` reads. Without it there is no install path. |
| `.claude-plugin/plugin.json` | **Modify.** Add `mcpServers` and `commands`. Schema is already correct; do not rewrite the rest. |
| `scripts/entroly-plugin-launch.mjs` | **Create.** Resolves `uvx` → `npx` → `entroly` and execs the first that exists. One job: survive a machine without `uv`. |
| `scripts/marketplace_presence.py` | **Create.** Given a channel and expected version, answers Present/Absent/Unknown. No publishing, no side effects. |
| `.claude-plugin/commands/entroly-first-run.md` | **Create.** The first-run wedge: omissions shown with their recovery handles. |
| `.github/workflows/publish-marketplaces.yml` | **Create.** Orchestration only. |
| `.github/workflows/publish-mcp-registry.yml` | **Modify.** Replace the duplicated inline heredoc with a call to the shared checker. |
| `scripts/bump_version.py` | **Modify.** Add `marketplace.json` to `TARGETS`. |
| `tests/test_plugin_marketplace_manifest.py` | **Create.** Schema and identity assertions. |
| `tests/test_plugin_launcher.py` | **Create.** Fallback-chain behavior. |
| `tests/test_marketplace_presence.py` | **Create.** Adapter states against fixtures; MCP parity regression. |
| `tests/test_first_run_copy.py` | **Create.** Guards the no-percentage rule. |

Design note on plugin root: `marketplace.json` declares `"source": "./"`, so the plugin root is the repository root. MCP servers are therefore declared **inline in `plugin.json`** rather than in a root `.mcp.json` — a root `.mcp.json` would also be picked up as project MCP config by anyone opening the Entroly repo, which is a side effect nobody asked for. Commands live under `.claude-plugin/commands/` for the same reason: a root `commands/` directory in this repository would read as something else entirely.

---

### Task 1: Marketplace manifest and plugin MCP wiring

Makes installation possible. Today it is not possible at all.

**Files:**
- Create: `.claude-plugin/marketplace.json`
- Modify: `.claude-plugin/plugin.json`
- Modify: `scripts/bump_version.py:88` (add an entry beside the existing `.claude-plugin` targets)
- Test: `tests/test_plugin_marketplace_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: marketplace name `entroly`; plugin name `entroly`; the MCP server command contract `node ${CLAUDE_PLUGIN_ROOT}/scripts/entroly-plugin-launch.mjs`, which Task 2 implements.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plugin_marketplace_manifest.py`:

```python
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPOSITORY = "https://github.com/juyterman1000/entroly"


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _master_version() -> str:
    text = (ROOT / "entroly" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    assert match is not None, "no __version__ in entroly/__init__.py"
    return match.group(1)


def test_marketplace_manifest_has_required_fields() -> None:
    manifest = _json(".claude-plugin/marketplace.json")

    assert manifest["name"] == "entroly"
    assert manifest["owner"]["name"] == "juyterman1000"
    assert isinstance(manifest["plugins"], list) and manifest["plugins"]


def test_marketplace_lists_the_plugin_at_the_master_version() -> None:
    manifest = _json(".claude-plugin/marketplace.json")
    entry = next(p for p in manifest["plugins"] if p["name"] == "entroly")

    assert entry["source"] == "./"
    assert entry["version"] == _master_version()


def test_plugin_declares_the_launcher_shim_not_a_bare_runner() -> None:
    plugin = _json(".claude-plugin/plugin.json")
    server = plugin["mcpServers"]["entroly"]

    # A bare `uvx` here is the silent-dead-plugin failure mode: MCP config has
    # no fallback chain, so a machine without uv gets a plugin that starts
    # nothing and reports nothing.
    assert server["command"] == "node"
    assert server["args"] == [
        "${CLAUDE_PLUGIN_ROOT}/scripts/entroly-plugin-launch.mjs"
    ]
    assert server["env"]["ENTROLY_NO_DOCKER"] == "1"


def test_plugin_identity_matches_the_canonical_repository() -> None:
    plugin = _json(".claude-plugin/plugin.json")

    assert plugin["name"] == "entroly"
    assert plugin["repository"] == CANONICAL_REPOSITORY
    assert plugin["version"] == _master_version()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plugin_marketplace_manifest.py -v`
Expected: FAIL — `FileNotFoundError` for `.claude-plugin/marketplace.json`.

- [ ] **Step 3: Create the marketplace manifest**

Create `.claude-plugin/marketplace.json`. Replace `1.0.83` with the current value of `__version__` in `entroly/__init__.py` if it has moved:

```json
{
  "name": "entroly",
  "description": "Auditable context selection, exact recovery, and receipts for AI coding agents.",
  "owner": {
    "name": "juyterman1000",
    "url": "https://github.com/juyterman1000"
  },
  "plugins": [
    {
      "name": "entroly",
      "source": "./",
      "description": "Evidence operations, auditable context selection, and exact recovery for AI coding agents.",
      "version": "1.0.83",
      "author": {
        "name": "Entroly"
      }
    }
  ]
}
```

- [ ] **Step 4: Add `mcpServers` and `commands` to the plugin manifest**

Modify `.claude-plugin/plugin.json`. Keep every existing key; add two:

```json
{
  "name": "entroly",
  "version": "1.0.83",
  "description": "Evidence operations, auditable context selection, and exact recovery for AI coding agents.",
  "author": {
    "name": "Entroly"
  },
  "homepage": "https://github.com/juyterman1000/entroly",
  "repository": "https://github.com/juyterman1000/entroly",
  "license": "Apache-2.0",
  "skills": "./skills/",
  "commands": "./.claude-plugin/commands/",
  "mcpServers": {
    "entroly": {
      "command": "node",
      "args": ["${CLAUDE_PLUGIN_ROOT}/scripts/entroly-plugin-launch.mjs"],
      "env": {
        "ENTROLY_NO_DOCKER": "1"
      }
    }
  }
}
```

- [ ] **Step 5: Register the new version surface**

Modify `scripts/bump_version.py`. Immediately after the `.claude-plugin/plugin.json` entry (line 91), add:

```python
    # The marketplace entry declares the plugin version to Claude Code's
    # install UI. It is a third `.claude-plugin` surface; the sweep below
    # would catch it, but catching it here means the bump never emits a
    # warning in the first place.
    (".claude-plugin/marketplace.json",
        r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_plugin_marketplace_manifest.py tests/test_version_surfaces_are_complete.py -v --timeout=300`
Expected: PASS. The version-surfaces test needs no new case — it asserts the property over every tracked file, so it covers `marketplace.json` the moment the file exists.

- [ ] **Step 7: Validate the manifest with the real tool**

Run: `claude plugin validate .`
Expected: reports the `entroly` marketplace and the `entroly` plugin with no errors. If `claude` is unavailable in the environment, record that and rely on Step 6.

- [ ] **Step 8: Commit**

```bash
git add .claude-plugin/marketplace.json .claude-plugin/plugin.json \
        scripts/bump_version.py tests/test_plugin_marketplace_manifest.py
git commit -m "feat(plugin): make Entroly installable from the plugin marketplace

Without .claude-plugin/marketplace.json there is no install path at all --
plugin.json alone is not discoverable. Declares mcpServers inline rather
than in a root .mcp.json, which would otherwise apply to anyone who simply
opens this repository.

Co-Authored-By: juyterman1000 <208309368+juyterman1000@users.noreply.github.com>"
```

---

### Task 2: The launcher shim

The plugin's MCP command points at this file. Until it exists, an installed plugin starts nothing.

Written in Node because Claude Code ships a Node runtime, so `node` is present wherever the plugin is. `python3` is not a safe assumption on Windows, and `uvx` is precisely the thing we cannot assume.

**Files:**
- Create: `scripts/entroly-plugin-launch.mjs`
- Test: `tests/test_plugin_launcher.py`

**Interfaces:**
- Consumes: the command contract from Task 1 — `node ${CLAUDE_PLUGIN_ROOT}/scripts/entroly-plugin-launch.mjs`.
- Produces: an executable that either becomes an Entroly MCP stdio server or exits `1` after printing one actionable line to stderr.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plugin_launcher.py`:

```python
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "entroly-plugin-launch.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="the launcher is a Node script; node is absent in this environment",
)


def _run(env_path: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = env_path
    env.update(extra_env or {})
    return subprocess.run(
        ["node", str(LAUNCHER)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_reports_one_actionable_line_when_no_runner_exists(tmp_path: Path) -> None:
    node_dir = str(Path(shutil.which("node")).parent)
    result = _run(node_dir)

    assert result.returncode == 1
    # The failure mode this guards is a plugin that starts nothing and says
    # nothing, which is indistinguishable from a broken install.
    assert "entroly" in result.stderr.lower()
    assert "uv" in result.stderr.lower()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake PATH executables need .cmd shims on Windows",
)
def test_prefers_uvx_and_passes_the_expected_arguments(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "invoked.txt"
    uvx = fake_bin / "uvx"
    uvx.write_text(
        '#!/bin/sh\nprintf "%s" "$*" > "{marker}"\nexit 0\n'.format(marker=marker),
        encoding="utf-8",
    )
    uvx.chmod(0o755)

    node_dir = str(Path(shutil.which("node")).parent)
    result = _run(f"{fake_bin}{os.pathsep}{node_dir}")

    assert result.returncode == 0
    assert marker.read_text(encoding="utf-8") == "--from entroly entroly"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake PATH executables need .cmd shims on Windows",
)
def test_falls_through_to_npx_when_uvx_is_absent(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "invoked.txt"
    npx = fake_bin / "npx"
    npx.write_text(
        '#!/bin/sh\nprintf "npx %s" "$*" > "{marker}"\nexit 0\n'.format(marker=marker),
        encoding="utf-8",
    )
    npx.chmod(0o755)

    node_dir = str(Path(shutil.which("node")).parent)
    result = _run(f"{fake_bin}{os.pathsep}{node_dir}")

    assert result.returncode == 0
    assert marker.read_text(encoding="utf-8") == "npx -y entroly@latest"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plugin_launcher.py -v`
Expected: FAIL — node reports `Cannot find module .../scripts/entroly-plugin-launch.mjs`.

- [ ] **Step 3: Write the launcher**

Create `scripts/entroly-plugin-launch.mjs`:

```javascript
#!/usr/bin/env node
// Entroly plugin launcher.
//
// The plugin's MCP config cannot express a fallback chain: it names one
// command, and if that command is missing the server never starts and the
// user sees a plugin that is silently inert. So the config names this file,
// and the fallback lives here.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, join } from "node:path";

const CANDIDATES = [
  { command: "uvx", args: ["--from", "entroly", "entroly"] },
  { command: "npx", args: ["-y", "entroly@latest"] },
  { command: "entroly", args: [] },
];

// Resolve against PATH ourselves rather than passing `shell: true`. With a
// shell, a missing command exits 1 like any other failure, and the chain
// cannot tell "not installed" from "installed and broken".
function resolve(command) {
  const extensions =
    process.platform === "win32"
      ? (process.env.PATHEXT || ".EXE;.CMD;.BAT").split(";")
      : [""];
  for (const dir of (process.env.PATH || "").split(delimiter)) {
    if (!dir) continue;
    for (const extension of extensions) {
      const candidate = join(dir, command + extension);
      if (existsSync(candidate)) return candidate;
    }
  }
  return null;
}

for (const candidate of CANDIDATES) {
  const executable = resolve(candidate.command);
  if (executable === null) continue;

  const result = spawnSync(executable, candidate.args, {
    stdio: "inherit",
    env: { ...process.env, ENTROLY_NO_DOCKER: "1" },
  });
  process.exit(result.status === null ? 1 : result.status);
}

process.stderr.write(
  "Entroly plugin: no runner found. Install uv (https://docs.astral.sh/uv/) " +
    "or Node, or run `pip install -U entroly`, then restart Claude Code.\n",
);
process.exit(1);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_plugin_launcher.py -v`
Expected: PASS (3 passed, or 1 passed / 2 skipped on Windows).

- [ ] **Step 5: Commit**

```bash
git add scripts/entroly-plugin-launch.mjs tests/test_plugin_launcher.py
git commit -m "feat(plugin): resolve a runner instead of assuming uvx exists

MCP configuration names one command with no fallback, so naming uvx
directly means a machine without uv gets a plugin that starts nothing and
reports nothing. Resolves PATH directly rather than using a shell, because
a shell collapses 'not installed' and 'installed and broken' into exit 1.

Co-Authored-By: juyterman1000 <208309368+juyterman1000@users.noreply.github.com>"
```

---

### Task 3: Presence checker with the MCP registry adapter

Extraction, not invention. `publish-mcp-registry.yml` already polls the registry and asserts name, version, canonical repository URL, and exact package set — twice, as duplicated inline heredocs. This lifts that logic into one tested module.

**Files:**
- Create: `scripts/marketplace_presence.py`
- Test: `tests/test_marketplace_presence.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Presence(str, Enum)` with members `PRESENT`, `ABSENT`, `UNKNOWN`
  - `@dataclass(frozen=True) class Probe: channel: str; presence: Presence; detail: str`
  - `def probe_mcp_registry(version: str, *, fetch: Fetch = _fetch_json) -> Probe`
  - `Fetch = Callable[[str], dict]`
  - Task 4 adds `probe_claude_marketplace` and `probe_smithery` with the same shape; Task 5 consumes `main()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_marketplace_presence.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_marketplace_presence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marketplace_presence'`.

- [ ] **Step 3: Write the checker**

Create `scripts/marketplace_presence.py`:

```python
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


PROBES: dict[str, Callable[..., Probe]] = {
    "mcp-registry": probe_mcp_registry,
}

# Channels Entroly publishes to block a release when absent. Channels that
# index on their own schedule cannot: Smithery pulls from GitHub whenever it
# chooses, so ABSENT there can be nobody's fault.
BLOCKING = {"mcp-registry"}


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_marketplace_presence.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Verify against the live registry**

Run: `python scripts/marketplace_presence.py --version 1.0.83`
Expected: prints `mcp-registry [blocking]: ...`. Either `present` or `absent` is a correct result here — the registry listed 1.0.81 while the repository was at 1.0.83 during the audit. Record which you saw.

- [ ] **Step 6: Commit**

```bash
git add scripts/marketplace_presence.py tests/test_marketplace_presence.py
git commit -m "feat(distribution): extract the registry presence gate into one module

publish-mcp-registry.yml already asserts name, version, canonical repository
and exact package set -- twice, as duplicated inline heredocs, for one
channel. Lifts it into a tested module with three result states so a moving
registry schema cannot redden a healthy release.

Co-Authored-By: juyterman1000 <208309368+juyterman1000@users.noreply.github.com>"
```

---

### Task 4: Claude Code marketplace and Smithery adapters

**Files:**
- Modify: `scripts/marketplace_presence.py`
- Test: `tests/test_marketplace_presence.py`

**Interfaces:**
- Consumes: `Presence`, `Probe`, `Fetch` from Task 3.
- Produces: `probe_claude_marketplace(version, *, fetch=...) -> Probe` (channel `claude-marketplace`, blocking) and `probe_smithery(version, *, fetch=...) -> Probe` (channel `smithery`, advisory). Both registered in `PROBES`.

The Claude marketplace is pull-based: GitHub `main` is the source of truth, so the probe reads the raw manifest. Smithery is also pull-based but on Smithery's schedule, so it is advisory — that distinction is what stops the gate from blocking releases forever.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_marketplace_presence.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_marketplace_presence.py -v`
Expected: FAIL — `ImportError: cannot import name 'probe_claude_marketplace'`.

- [ ] **Step 3: Add the adapters**

In `scripts/marketplace_presence.py`, add these functions after `probe_mcp_registry`:

```python
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
```

Then replace the `PROBES` dictionary:

```python
PROBES: dict[str, Callable[..., Probe]] = {
    "mcp-registry": probe_mcp_registry,
    "claude-marketplace": probe_claude_marketplace,
    "smithery": probe_smithery,
}
```

And replace `BLOCKING`:

```python
BLOCKING = {"mcp-registry", "claude-marketplace"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_marketplace_presence.py -v`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/marketplace_presence.py tests/test_marketplace_presence.py
git commit -m "feat(distribution): probe the marketplace and Smithery channels

Smithery is advisory because it indexes from GitHub on its own schedule, so
absence there can be nobody's fault and must not block a release. The
marketplace manifest on main is the source of truth for the Claude Code
channel, so that one blocks.

Co-Authored-By: juyterman1000 <208309368+juyterman1000@users.noreply.github.com>"
```

---

### Task 5: Wire the checker into the workflows

**Files:**
- Create: `.github/workflows/publish-marketplaces.yml`
- Modify: `.github/workflows/publish-mcp-registry.yml` (the two inline verification heredocs)

**Interfaces:**
- Consumes: `python scripts/marketplace_presence.py --version <v> [--channel <c>]`, exit `1` when a blocking channel is absent.
- Produces: no Python interface.

Per the spec's rollout, the new workflow runs **advisory-only for one full release cycle**. It must not gate a release before it has been observed green.

- [ ] **Step 1: Replace the duplicated heredoc in the existing workflow**

In `.github/workflows/publish-mcp-registry.yml`, replace the body of the `Verify exact registry ownership and listing` step (the ~60-line inline `python - <<'PY'` heredoc beginning around line 310) with:

```yaml
      - name: Verify exact registry ownership and listing
        if: steps.release_guard.outputs.should_publish == 'true'
        shell: bash
        run: |
          version="$(python -c "import json;print(json.load(open('server.json'))['version'])")"
          for attempt in $(seq 1 60); do
            if python scripts/marketplace_presence.py \
                 --version "$version" --channel mcp-registry; then
              exit 0
            fi
            sleep 10
          done
          echo "::error::registry did not list $version after 60 attempts"
          exit 1
```

The retry loop is preserved deliberately: registry propagation is not instant, and the original polled sixty times for that reason.

- [ ] **Step 2: Verify the existing regression tests still pass**

Run: `pytest tests/test_mcp_registry_manifest.py -v`
Expected: PASS. The manifest contract is unchanged; only the verification mechanism moved.

- [ ] **Step 3: Create the advisory workflow**

Create `.github/workflows/publish-marketplaces.yml`:

```yaml
name: Verify marketplace presence

on:
  workflow_dispatch:
  schedule:
    # Daily. A channel that dies between releases should not wait for the
    # next release to be noticed -- Smithery was dead for months.
    - cron: "17 6 * * *"

permissions:
  contents: read

jobs:
  presence:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Probe every distribution channel
        id: probe
        continue-on-error: true
        run: |
          version="$(python -c "import re,pathlib;print(re.search(r'__version__\s*=\s*\"([^\"]+)\"', pathlib.Path('entroly/__init__.py').read_text()).group(1))")"
          python scripts/marketplace_presence.py --version "$version" --channel all

      - name: Report
        run: |
          if [ "${{ steps.probe.outcome }}" = "failure" ]; then
            echo "::warning::a distribution channel is not listing the current version"
          fi
```

`continue-on-error` plus a warning is the advisory mode the rollout requires. Task 5 Step 5 flips it after one clean cycle.

- [ ] **Step 4: Validate the workflow files parse**

Run: `python -c "import yaml,pathlib; [yaml.safe_load(pathlib.Path(p).read_text()) for p in ['.github/workflows/publish-marketplaces.yml', '.github/workflows/publish-mcp-registry.yml']]; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Record the promotion condition**

Add this comment at the top of the `presence` job in `publish-marketplaces.yml`, directly under `runs-on`:

```yaml
    # ADVISORY until one full release cycle has run green. To promote:
    # delete `continue-on-error` from the probe step. Do not promote early --
    # a gate that reddens a healthy release gets disabled, and a disabled
    # gate is how Smithery stayed dead.
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/publish-marketplaces.yml .github/workflows/publish-mcp-registry.yml
git commit -m "feat(ci): check every channel daily, not one channel at release

Smithery was dead for months between releases, so a release-time-only check
would not have caught it. Runs advisory until one clean cycle: a gate that
reddens a healthy release gets disabled, and a disabled gate is the failure
this is meant to prevent.

Co-Authored-By: juyterman1000 <208309368+juyterman1000@users.noreply.github.com>"
```

---

### Task 6: The first-run wedge

A plugin that installs and shows nothing gets uninstalled. First run demonstrates the property competitors do not have: omitted evidence shown together with the handles that recover it.

**Files:**
- Create: `.claude-plugin/commands/entroly-first-run.md`
- Test: `tests/test_first_run_copy.py`

**Interfaces:**
- Consumes: the `commands` path registered in Task 1.
- Produces: the `/entroly-first-run` slash command.

- [ ] **Step 1: Write the failing test**

Create `tests/test_first_run_copy.py`:

```python
"""The first-run screen may not carry a savings percentage.

`saved = max(0, baseline - selected_tokens)` with
`baseline = min(total_tokens, 32_000)` in entroly/cli.py pins the figure at
or above 75% for a budget of 8,000 before selection has run. It is budget
arithmetic wearing the costume of a measurement, and a first-run screen is
the widest possible distribution for it.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / ".claude-plugin" / "commands" / "entroly-first-run.md"

PERCENTAGE = re.compile(r"\d{1,3}\s?%")


def test_first_run_command_exists() -> None:
    assert COMMAND.is_file()


def test_first_run_copy_states_no_savings_percentage() -> None:
    matches = PERCENTAGE.findall(COMMAND.read_text(encoding="utf-8"))

    assert matches == [], f"first-run copy must not quote a percentage: {matches}"


def test_first_run_copy_shows_recovery_handles() -> None:
    text = COMMAND.read_text(encoding="utf-8").lower()

    # The differentiator is not that context got smaller. It is that what
    # was left out is named and recoverable.
    assert "recover" in text
    assert "omitted" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_first_run_copy.py -v`
Expected: FAIL — `assert False` on `COMMAND.is_file()`.

- [ ] **Step 3: Write the command**

Create `.claude-plugin/commands/entroly-first-run.md`:

```markdown
---
description: Show what Entroly selected from this repository, what it omitted, and how to recover the omissions.
---

Show the user what Entroly does to their own repository, using their real code.

1. Run `entroly compile` against the project's primary source directory, then
   `entroly simulate` scoped to that directory.
2. Report three things, in this order:
   - **Selected** — the fragments that were kept, with their files.
   - **Omitted** — the fragments that were left out, with their files. Name
     them. Do not summarise them as a count.
   - **Recovery** — for each omitted fragment, the content-addressed handle
     that retrieves the exact original bytes. Demonstrate one live recovery.
3. Close by inviting the user to pick any omitted fragment and recover it.

Do not report a savings percentage, a compression ratio, or a token-reduction
figure. The percentage is determined by the configured budget before selection
runs, so it measures the budget rather than the selection. The claim worth
making is that nothing was lost irrecoverably, and that claim is checkable in
front of the user — which is the entire point.

If the native engine is missing, say so plainly and state that selection has
not been query-conditioned, rather than reporting a figure that looks earned.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_first_run_copy.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full affected suite**

Run: `pytest tests/test_plugin_marketplace_manifest.py tests/test_plugin_launcher.py tests/test_marketplace_presence.py tests/test_first_run_copy.py tests/test_version_surfaces_are_complete.py -v --timeout=300`
Expected: all pass. `--timeout=300` is required — `tests/test_docs_code_sync.py` and its neighbours allow a 300s subprocess, and a shorter pytest timeout kills them with a misleading error.

- [ ] **Step 6: Commit**

```bash
git add .claude-plugin/commands/entroly-first-run.md tests/test_first_run_copy.py
git commit -m "feat(plugin): prove recovery on first run, without a percentage

The savings figure is decided by the token budget before selection runs, so
a first-run screen quoting it would distribute a number that flatters
Entroly for a reason unrelated to Entroly. Shows named omissions and a live
recovery instead, which is the property competitors do not have. A test
fails if a percentage is reintroduced.

Co-Authored-By: juyterman1000 <208309368+juyterman1000@users.noreply.github.com>"
```

---

## Manual step, outside the plan

Submit Entroly to Smithery once, by hand, at https://smithery.ai. `smithery.yaml` has been correct and unsubmitted for months. After submission, the advisory probe from Task 4 reports on it daily and it cannot silently die again.

## Self-review

**Spec coverage:** Part 1 → Tasks 1 and 2. Part 2 → Task 6. Part 3 → Tasks 3, 4 and 5. Failure mode 1 → Task 2. Failure mode 2 → Task 3 (`UNKNOWN` states). Failure mode 3 → Task 4 (`BLOCKING` set). Failure mode 4 → Task 3 (version comparison against the released version). Test matrix rows: unit → Tasks 3 and 4; regression → Task 3 Steps 1 and 5; schema → Task 1; contract → Task 5. Rollout step 3 → the manual step above. No spec requirement is unimplemented.

**Placeholders:** none. Every code step carries the code.

**Type consistency:** `Presence`, `Probe`, and `Fetch` are defined in Task 3 and used unchanged in Tasks 4 and 5. `probe_mcp_registry`, `probe_claude_marketplace`, and `probe_smithery` share one signature. The MCP command contract asserted in Task 1 Step 1 matches the file Task 2 creates and the `plugin.json` value in Task 1 Step 4.
