# Windows artifact verification

Status: prepared, not submitted.

Evidence that the released Windows CLI artifact is a stable, self-contained
binary — the precondition the Scoop Main entry in
[`targets.json`](targets.json) was blocked on.

Everything below was executed against the **published release asset**, not a
local build. Commands and raw output are recorded so the result is reproducible
rather than asserted.

- Release: `entroly-v1.0.81`
- Asset: `entroly-rs-x86_64-pc-windows-msvc.zip` (1,285,828 bytes)
- Host: Windows 11, x86_64
- Date: 2026-09-05

## 1. Download and checksum

```
gh release download entroly-v1.0.81 \
  --pattern "entroly-rs-x86_64-pc-windows-msvc.zip*"
```

Published sidecar `entroly-rs-x86_64-pc-windows-msvc.zip.sha256`:

```
d7d0a92eb90318f2316071c2bf8356d2cadd2c28feffc793591079475aa97894 *dist/entroly-rs-x86_64-pc-windows-msvc.zip
```

Recomputed locally:

```
computed: d7d0a92eb90318f2316071c2bf8356d2cadd2c28feffc793591079475aa97894
size    : 1285828 bytes
```

**Match.**

## 2. Archive contents

```
   2885120  entroly-rs.exe
```

One file, no installer, no side directories. Extract-and-run.

## 3. The binary runs

```
> entroly-rs.exe --version
entroly-rs 1.0.81

> entroly-rs.exe --help
Single-binary context compressor — no Python runtime required.
exit code: 0
```

The reported version matches the release tag and the repository version.

## 4. It does the thing it claims

```
> entroly-rs.exe compress --budget 120 sample.py
input bytes: 7786
output bytes: 445
exit code: 0
```

Output is well-formed source text, not a truncation artifact.

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

- **Autoupdate hash extraction is unverified.** The manifest resolves the hash
  from `$url.sha256`, and the published sidecar names the file as
  `*dist/entroly-rs-x86_64-pc-windows-msvc.zip`. Scoop matches a hash line
  against the *basename*, so the `dist/` prefix may defeat extraction and force
  a fallback. Scoop is not installed on the verification host, so this was not
  exercised. Resolve before submitting: either confirm Scoop's fallback handles
  the prefix, or publish the sidecar with a bare basename.
- **`scoop install` end-to-end** — not run, same reason.
- **32-bit and ARM64 Windows** — no such asset is published; the manifest
  declares `64bit` only.

## Manifest

[`packaging/scoop/entroly.json`](../../packaging/scoop/entroly.json), pinned to
the verified URL and hash above, with `checkver` on the `entroly-v<version>`
tag format this project uses.
