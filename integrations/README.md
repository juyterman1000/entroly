# Agent bundles

Entroly ships narrow bundles for Codex, Claude Code, Gemini CLI, Cursor, and
Kiro. Codex, Claude Code, and Gemini bundle prompt hooks with their plugin or
extension. Cursor and Kiro use reversible project installers because their
current hook distribution contracts differ. None alter provider credentials or
enable remote telemetry.

An MCP server alone does not guarantee use: the model may never choose its
tools. The prompt-hook paths run before agent planning. Confirm execution with
`entroly activation status --json`; an `unobserved` state is not evidence that
the hook is installed or working.

## Install, inspect, and reverse

Public Codex install (Node 16+ and npm required):

```console
codex plugin marketplace add juyterman1000/entroly --ref main
codex plugin add entroly@entroly-public
```

The public marketplace entry installs the bundled portable plugin from the
signed Git checkout. The separately published `entroly` npm package declares
the exact `entroly-wasm` runtime dependency for npm-based installs. Codex asks
the user to review and trust bundled hooks.
Restart Codex after installation, run one task, and confirm a recent receipt
with `entroly activation status --json`.

Public Gemini CLI install (Git required):

```console
gemini extensions install https://github.com/juyterman1000/entroly --ref main --consent
```

The repository root is a complete Gemini extension. Restart Gemini CLI after
installing; extension management changes take effect on restart.

VS Code and Kiro users can install the versioned `entroly-vscode-*.vsix` file
from the [latest GitHub release](https://github.com/juyterman1000/entroly/releases/latest)
using **Extensions: Install from VSIX** or `code --install-extension`.

JetBrains AI Assistant users can add Entroly globally from **Settings | Tools |
AI Assistant | Model Context Protocol (MCP)** with:

```json
{
  "mcpServers": {
    "entroly": {
      "command": "npx",
      "args": ["-y", "entroly-mcp@1.0.85", "serve"],
      "env": {
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_MCP_PASSIVE": "1",
        "ENTROLY_MCP_PROFILE": "public",
        "ENTROLY_MAX_FILES": "200"
      }
    }
  }
}
```

Marketplace and plugin manifests select the compact public MCP profile. It
keeps the first run focused on context, receipts, continuity, exact recovery,
and verification. A direct `entroly serve` invocation remains backwards
compatible and exposes the full surface; choose explicitly with
`ENTROLY_MCP_PROFILE=public` or `ENTROLY_MCP_PROFILE=full`.

Windows PowerShell:

```powershell
./scripts/install-agent-bundles.ps1 status
./scripts/install-agent-bundles.ps1 install -Agent all
./scripts/install-agent-bundles.ps1 uninstall -Agent gemini
```

macOS/Linux:

```bash
./scripts/install-agent-bundles.sh status
./scripts/install-agent-bundles.sh install --agent all
./scripts/install-agent-bundles.sh uninstall --agent gemini
```

Install refuses existing destinations unless `-Force`/`--force` is explicit.
Forced installs create timestamped backups. Uninstall moves only directories
carrying an Entroly bundle marker to a recoverable disabled path; it does not
delete them.

The portable Codex plugin source is in `integrations/codex/entroly`; the
published npm copy is assembled in `entroly/npm-alias`. Contract tests require
their manifests, hook, launcher, and skill to remain byte-for-byte aligned.

Cursor's native `beforeSubmitPrompt` output cannot inject context, so Entroly
uses Cursor's documented Claude-hook compatibility mode:

```console
entroly activation install --host cursor --project .
```

For MCP-only use, Cursor can add the published server through this one-click
[MCP install link](cursor://anysphere.cursor-deeplink/mcp/install?name=entroly&config=eyJjb21tYW5kIjoibnB4IiwiYXJncyI6WyIteSIsImVudHJvbHktbWNwQDEuMC44NCIsInNlcnZlIl0sImVudiI6eyJFTlRST0xZX05PX0RPQ0tFUiI6IjEiLCJFTlRST0xZX01DUF9QQVNTSVZFIjoiMSIsIkVOVFJPTFlfTUFYX0ZJTEVTIjoiMjAwIn19).

Kiro IDE 1.x and CLI 3.x accept context from successful `PromptSubmit` command
stdout:

```console
entroly activation install --host kiro --project .
```

Both project installers preserve existing configuration, create backups before
rewriting an existing file, and support reversible `activation uninstall`.
