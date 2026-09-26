# Windows artifact verification

Status: prepared, not submitted.

Evidence that the released Windows CLI artifact is a stable, self-contained
binary — the precondition the Scoop Main entry in
[`targets.json`](targets.json) was blocked on.

Everything below was executed against the **published release asset**, not a
local build. Commands and raw output are recorded so the result is reproducible
rather than asserted.

- Release: `entroly-v1.0.85`
- Asset: `entroly-rs-x86_64-pc-windows-msvc.zip` (1,293,179 bytes)
- Host: Windows 11, x86_64
- Date: 2026-09-24

## 1. Download and checksum

```
gh release download entroly-v1.0.85 \
  --pattern "entroly-rs-x86_64-pc-windows-msvc.zip*"
```

Published sidecar `entroly-rs-x86_64-pc-windows-msvc.zip.sha256`:

```
1f5301c6c043915566e90856fa79b51296c9004dc4958c02e86a670024d5781d *entroly-rs-x86_64-pc-windows-msvc.zip
```

Recomputed locally:

```
computed: 1f5301c6c043915566e90856fa79b51296c9004dc4958c02e86a670024d5781d
size    : 1293179 bytes
```

**Match.**

## 2. Archive contents

```
   2892800  entroly-rs.exe
```

One file, no installer, no side directories. Extract-and-run.

## 3. The binary runs

```
> entroly-rs.exe --version
entroly-rs 1.0.85

> entroly-rs.exe --help
Single-binary context compressor — no Python runtime required.
exit code: 0
```

The reported version matches the release tag and the repository version.

## 4. It does the thing it claims

```
> entroly-rs.exe compress --budget 120 entroly/cli.py
input estimate: ~85303 tokens
output estimate: ~181 tokens (726 UTF-8 bytes)
exit code: 0
```

Output decoded as UTF-8 and contained no NUL bytes. This smoke test establishes
that the published binary executes its compression path; it does not claim that
the compressed selection preserves every behavior of the input module.

## 5. Silent install contract

Scoop installs a `bin` entry by extracting the archive and shimming the
executable. This artifact satisfies that contract without special handling:

- no installer, no MSI, no elevation prompt, no interactive step;
- a single self-contained `.exe`; nothing is written outside the extraction
  directory at install time;
- no Python runtime, no PATH mutation performed by the artifact itself;
- uninstall is removal of the extracted directory — the binary installs nothing
  elsewhere. Runtime state is created only when a command is run, under
  `ENTROLY_DIR`, and is user data rather than install state.

## 6. Not verified here

Recorded so the gap is visible instead of implied:

- **Autoupdate execution is unverified.** The 1.0.85 sidecar now names the bare
  archive basename expected by the manifest's `$url.sha256` source, and the
  digest was independently recomputed. Scoop itself was not installed on this
  verification host, so its updater was not executed.
- **`scoop install` end-to-end** — not run, same reason.
- **32-bit and ARM64 Windows** — no such asset is published; the manifest
  declares `64bit` only.

## Manifest

[`packaging/scoop/entroly.json`](../../packaging/scoop/entroly.json), pinned to
the verified URL and hash above, with `checkver` on the `entroly-v<version>`
tag format this project uses.
