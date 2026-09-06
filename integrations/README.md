# Agent bundles

Entroly ships narrow, native bundles for Codex, Claude Code, and Gemini CLI.
They expose evidence operations and the local MCP server; they do not alter
provider credentials or enable remote telemetry.

## Install, inspect, and reverse

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

The Codex plugin archive is in `integrations/codex/entroly`. The installer
places its skill directly in the local Codex skill directory because repository
distribution and marketplace publication are separate release operations.
