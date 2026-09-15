# Recovery data: storage and access boundaries

Recoverability requires retaining source material. Context Receipt indexes,
Context Commits, and compression-recovery records can contain exact source code,
prompts, paths, and secrets present in their inputs. A digest is an integrity
check, not encryption, identity verification, or proof of answer correctness.

## Locate the data before sharing or deleting it

| Surface | Location and behavior |
| --- | --- |
| Context Receipts | `.entroly/receipts/` at the repository root discovered by upward search, bounded by `.git`; explicit output paths may differ. |
| Context Commits | `.entroly/context-commits/` under the selected workspace; inspect the path reported by the command. |
| `entroly compress` / `recover` | Explicit `--store-path`, otherwise `ENTROLY_DIR/recovery.json`, otherwise `~/.entroly/checkpoints/<cwd-hash>/recovery.json`. If the home checkpoint directory is unwritable, the checkpoint resolver can fall back under the OS temporary directory. |
| SDK / proxy recovery | The caller's configured store and scope. In-memory stores do not survive process exit. |
| Other local state | Vaults, checkpoints, ledgers, logs, telemetry queues, and backups have separate paths and policies. Deleting a project's `.entroly/` does not erase all of these. |

Use a private, operator-controlled directory. Keep recovery artifacts out of Git,
public issue attachments, shared build outputs, and unreviewed diagnostic bundles.
Do not ingest credentials merely because a secret scanner exists: exact recovery
intentionally preserves inputs and cannot also promise to redact every original.

## What the store enforces

Context Receipt JSON, text reports, latest pointers, and Context Commit JSON written
through `context_receipts.store` use a temporary file in the destination directory,
flush and fsync it, and atomically replace the destination. A failed replacement
preserves the previous artifact. This is per-file atomicity, not a transaction
across all receipt artifacts or a guarantee against every power-loss scenario.

On POSIX, new artifact files use mode `0600`, and directly created store directories
use `0700`. Existing directories and intermediate parents retain their permissions.
On Windows, files inherit the parent directory's ACL; Entroly does not provision a
new Windows access-control policy. Inspect that ACL before storing sensitive data.
Final-component symlink destinations are rejected; parent paths must still be
trusted. This is not a sandbox against malicious filesystem changes by another
process with write access to the parent directory.

The compression-retrieval API separately validates scope and content integrity.
Scope filtering prevents accidental cross-workspace API retrieval; a scope string
is not an OS identity or a secret capability. Anyone who can read the backing
files can read their contents. Use separate service identities and private stores
for different tenants. These changes do not encrypt the data at rest or provide
protection from a compromised account running Entroly.

## Retention and deletion

Context Receipts and Context Commits have no automatic expiry policy. Set retention
according to the source data's sensitivity and your need to reproduce a result.
The compression store has a separate byte cap (`ENTROLY_RECOVERY_STORE_MAX_BYTES`);
a byte cap is not time-based deletion. Telemetry's retention limits do not apply to
source recovery records.

To retire recovery material: stop writers, resolve and inspect the actual store
paths, retain any explicitly required evidence in a protected backup, then remove
the intended artifacts using your filesystem's tools. Include configured paths,
home checkpoint stores, temporary fallbacks, backups, and synced copies in that
review. Do not delete a shared store for one project without checking its other
users. Deletion invalidates affected recovery handles; missing recovery material
must be reported as unavailable rather than reconstructed from a guess.

Filesystem deletion does not guarantee forensic erasure on SSDs, snapshots, or
remote backups. Use your disk-encryption and backup-retention controls where needed.
See [Privacy](../PRIVACY.md) for network recipients and telemetry withdrawal.
