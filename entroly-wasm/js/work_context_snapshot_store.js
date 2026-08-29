'use strict';

// Host persistence mechanics only. The v1 verified-context commitment rules
// live in entroly-engine and are reached here through the generated WASM
// binding. This file owns repository isolation, bounds, atomic persistence and
// byte preservation; it does not define what makes a context snapshot valid.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { verifiedContextSnapshotVerifyBytes } = require('../pkg/entroly_wasm');
const { WorkGraphStateError, WorkGraphStore } = require('./work_graph_store');

const CONTEXT_SNAPSHOT_TOKEN_PREFIX = 'wctx1.';
const DEFAULT_MAX_CONTEXT_BYTES = 512 * 1024;
const DEFAULT_MAX_SNAPSHOTS = 8192;
const DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024;
const DIGEST_RE = /^[0-9a-f]{64}$/;

class WorkContextSnapshotError extends WorkGraphStateError {}

function positiveInteger(value, name, minimum = 1) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new WorkContextSnapshotError(`${name} must be a safe integer >= ${minimum}`);
  }
  return value;
}

function secureDirectory(target) {
  fs.mkdirSync(target, { recursive: true, mode: 0o700 });
  const info = fs.lstatSync(target);
  if (info.isSymbolicLink() || !info.isDirectory()) {
    throw new WorkContextSnapshotError(`unsafe context snapshot directory: ${target}`);
  }
  if (process.platform !== 'win32') fs.chmodSync(target, 0o700);
}

function fsyncDirectory(target) {
  if (process.platform === 'win32') return;
  let fd;
  try {
    fd = fs.openSync(target, fs.constants.O_RDONLY | (fs.constants.O_DIRECTORY || 0));
    fs.fsyncSync(fd);
  } catch (_) {
    // Directory fsync is not supported by every filesystem/runtime. The Python
    // store has the same best-effort boundary; file data itself is fsynced.
  } finally {
    if (fd !== undefined) {
      try { fs.closeSync(fd); } catch (_) {}
    }
  }
}

function asBuffer(value) {
  if (Buffer.isBuffer(value)) return Buffer.from(value);
  if (value instanceof Uint8Array) return Buffer.from(value);
  if (typeof value === 'string') return Buffer.from(value, 'utf8');
  throw new WorkContextSnapshotError('context snapshot must be bytes or UTF-8 text');
}

/**
 * Verify exact snapshot bytes through the Rust semantic kernel.
 *
 * The byte stream is intentionally not normalized or reserialized before
 * verification. Python v1 commits its compact ensure-ASCII representation and
 * valid number lexemes such as `1.0` must survive a Python -> Node -> Python
 * handoff byte-for-byte.
 */
function verifyCanonicalSnapshotBytes(
  value,
  expectedCommitment,
  maxBytes = DEFAULT_MAX_CONTEXT_BYTES,
) {
  const bytes = asBuffer(value);
  positiveInteger(maxBytes, 'maxBytes', 1024);
  if (bytes.length > maxBytes) {
    throw new WorkContextSnapshotError(`context snapshot exceeds ${maxBytes} bytes`);
  }
  const expected = String(expectedCommitment || '');
  try {
    const commitment = verifiedContextSnapshotVerifyBytes(bytes, expected);
    // Parsing is presentation only and happens after Rust has accepted the exact
    // bytes. The parsed object is never reserialized for commitment verification.
    const payload = JSON.parse(bytes.toString('ascii'));
    return { payload, commitment, bytes };
  } catch (error) {
    if (error instanceof WorkContextSnapshotError) throw error;
    const detail = error && error.message ? error.message : String(error);
    throw new WorkContextSnapshotError(detail);
  }
}

class WorkContextSnapshotStore {
  constructor(graphStore, options = {}) {
    if (!(graphStore instanceof WorkGraphStore)) {
      throw new TypeError('graphStore must be a WorkGraphStore');
    }
    this.graphStore = graphStore;
    this.maxContextBytes = positiveInteger(
      options.maxContextBytes ?? DEFAULT_MAX_CONTEXT_BYTES,
      'maxContextBytes',
      1024,
    );
    this.maxSnapshots = positiveInteger(
      options.maxSnapshots ?? DEFAULT_MAX_SNAPSHOTS,
      'maxSnapshots',
    );
    if (this.maxSnapshots > 100000) {
      throw new WorkContextSnapshotError('maxSnapshots must not exceed 100000');
    }
    this.maxTotalBytes = positiveInteger(
      options.maxTotalBytes ?? DEFAULT_MAX_TOTAL_BYTES,
      'maxTotalBytes',
      this.maxContextBytes,
    );
    this.contextDir = path.join(graphStore.repoDir, 'context-snapshots');
    secureDirectory(this.contextDir);
  }

  static tokenForCommitment(commitment) {
    const digest = String(commitment || '');
    if (!DIGEST_RE.test(digest)) {
      throw new WorkContextSnapshotError('context commitment is not a sha256 digest');
    }
    return CONTEXT_SNAPSHOT_TOKEN_PREFIX + digest;
  }

  static digestFromToken(token) {
    const value = String(token || '');
    if (!value.startsWith(CONTEXT_SNAPSHOT_TOKEN_PREFIX)) {
      throw new WorkContextSnapshotError('unsupported context snapshot token');
    }
    const digest = value.slice(CONTEXT_SNAPSHOT_TOKEN_PREFIX.length);
    if (!DIGEST_RE.test(digest)) {
      throw new WorkContextSnapshotError('context snapshot token has an invalid digest');
    }
    return digest;
  }

  _snapshotPath(digest) {
    return path.join(this.contextDir, `${digest}.json`);
  }

  _readUnlocked(target, expectedDigest) {
    let info;
    try { info = fs.lstatSync(target); }
    catch (error) {
      if (error && error.code === 'ENOENT') {
        throw new WorkContextSnapshotError('context snapshot is unavailable');
      }
      throw new WorkContextSnapshotError(`cannot inspect context snapshot: ${error.message}`);
    }
    if (info.isSymbolicLink() || !info.isFile()) {
      throw new WorkContextSnapshotError('unsafe context snapshot path');
    }
    if (info.size > this.maxContextBytes) {
      throw new WorkContextSnapshotError('context snapshot exceeds its byte bound');
    }

    const flags = fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0);
    let fd;
    try {
      fd = fs.openSync(target, flags);
      const current = fs.fstatSync(fd);
      if (!current.isFile() || current.size > this.maxContextBytes) {
        throw new WorkContextSnapshotError('unsafe or oversized context snapshot');
      }
      const bytes = fs.readFileSync(fd);
      const verified = verifyCanonicalSnapshotBytes(bytes, expectedDigest, this.maxContextBytes);
      return verified.bytes;
    } catch (error) {
      if (error instanceof WorkContextSnapshotError) throw error;
      throw new WorkContextSnapshotError(`cannot open context snapshot safely: ${error.message}`);
    } finally {
      if (fd !== undefined) {
        try { fs.closeSync(fd); } catch (_) {}
      }
    }
  }

  _usageUnlocked() {
    let count = 0;
    let totalBytes = 0;
    for (const name of fs.readdirSync(this.contextDir)) {
      if (name.startsWith('.context-') && name.endsWith('.tmp')) continue;
      if (!name.endsWith('.json')) continue;
      const target = path.join(this.contextDir, name);
      const info = fs.lstatSync(target);
      if (info.isSymbolicLink() || !info.isFile()) {
        throw new WorkContextSnapshotError('unsafe context snapshot entry');
      }
      count += 1;
      totalBytes += info.size;
      if (count > this.maxSnapshots || totalBytes > this.maxTotalBytes) break;
    }
    return { count, totalBytes };
  }

  putCanonicalBytes(value, expectedCommitment) {
    const expected = String(expectedCommitment || '');
    if (!DIGEST_RE.test(expected)) {
      throw new WorkContextSnapshotError('expectedCommitment must be a sha256 digest');
    }
    const verified = verifyCanonicalSnapshotBytes(value, expected, this.maxContextBytes);
    const token = WorkContextSnapshotStore.tokenForCommitment(expected);
    const target = this._snapshotPath(expected);

    return this.graphStore.withLock(() => {
      secureDirectory(this.contextDir);
      if (fs.existsSync(target)) {
        const existing = this._readUnlocked(target, expected);
        if (!existing.equals(verified.bytes)) {
          throw new WorkContextSnapshotError(
            'context commitment maps to conflicting stable snapshot bytes',
          );
        }
        return token;
      }
      const usage = this._usageUnlocked();
      if (usage.count >= this.maxSnapshots) {
        throw new WorkContextSnapshotError('context snapshot store reached its bounded entry limit');
      }
      if (usage.totalBytes + verified.bytes.length > this.maxTotalBytes) {
        throw new WorkContextSnapshotError('context snapshot store reached its bounded byte limit');
      }

      const temp = path.join(
        this.contextDir,
        `.context-${process.pid}-${crypto.randomUUID().replace(/-/g, '')}.tmp`,
      );
      let fd;
      try {
        fd = fs.openSync(temp, 'wx', 0o600);
        fs.writeFileSync(fd, verified.bytes);
        fs.fsyncSync(fd);
        fs.closeSync(fd);
        fd = undefined;
        if (fs.existsSync(target)) {
          const existing = this._readUnlocked(target, expected);
          if (!existing.equals(verified.bytes)) {
            throw new WorkContextSnapshotError(
              'context commitment maps to conflicting stable snapshot bytes',
            );
          }
        } else {
          fs.renameSync(temp, target);
          if (process.platform !== 'win32') fs.chmodSync(target, 0o600);
          fsyncDirectory(this.contextDir);
        }
      } finally {
        if (fd !== undefined) {
          try { fs.closeSync(fd); } catch (_) {}
        }
        try { fs.unlinkSync(temp); } catch (_) {}
      }
      return token;
    });
  }

  getCanonicalBytes(token) {
    const digest = WorkContextSnapshotStore.digestFromToken(token);
    return this.graphStore.withLock(() => {
      secureDirectory(this.contextDir);
      return this._readUnlocked(this._snapshotPath(digest), digest);
    });
  }

  getJSON(token) {
    return JSON.parse(this.getCanonicalBytes(token).toString('ascii'));
  }
}

module.exports = {
  CONTEXT_SNAPSHOT_TOKEN_PREFIX,
  DEFAULT_MAX_CONTEXT_BYTES,
  DEFAULT_MAX_SNAPSHOTS,
  DEFAULT_MAX_TOTAL_BYTES,
  WorkContextSnapshotError,
  WorkContextSnapshotStore,
  verifyCanonicalSnapshotBytes,
};
