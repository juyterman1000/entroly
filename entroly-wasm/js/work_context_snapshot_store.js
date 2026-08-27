'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { WorkGraphStateError, WorkGraphStore } = require('./work_graph_store');

const CONTEXT_SNAPSHOT_TOKEN_PREFIX = 'wctx1.';
const CONTEXT_SCHEMA_VERSION = 'entroly.verified-code-context.v1';
const CONTEXT_COMMITMENT_SCOPE = 'payload-excluding-generation-command-and-context-sha256';
const DEFAULT_MAX_CONTEXT_BYTES = 512 * 1024;
const DEFAULT_MAX_SNAPSHOTS = 8192;
const DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024;
const DIGEST_RE = /^[0-9a-f]{64}$/;
const VOLATILE_FIELDS = new Set(['generation', 'command']);

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

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function asBuffer(value) {
  if (Buffer.isBuffer(value)) return Buffer.from(value);
  if (value instanceof Uint8Array) return Buffer.from(value);
  if (typeof value === 'string') return Buffer.from(value, 'utf8');
  throw new WorkContextSnapshotError('context snapshot must be bytes or UTF-8 text');
}

function countContextShaKeys(root) {
  let count = 0;
  const stack = [root];
  while (stack.length) {
    const value = stack.pop();
    if (!value || typeof value !== 'object') continue;
    if (Array.isArray(value)) {
      for (const child of value) stack.push(child);
      continue;
    }
    if (Object.prototype.hasOwnProperty.call(value, 'context_sha256')) count += 1;
    for (const child of Object.values(value)) stack.push(child);
  }
  return count;
}

/**
 * Verify a Python v1 verified-code-context snapshot without reserialising it.
 *
 * Python seals the context by hashing compact, sorted, ensure_ascii JSON after
 * removing `generation`, `command` and `receipt.context_sha256`. The snapshot
 * store has already removed the two volatile top-level fields, so the exact
 * seal preimage is the stored byte stream with that one receipt field removed.
 * Working on bytes is intentional: JSON.parse/stringify would turn `1.0` into
 * `1`, which would make a legitimate Python commitment impossible to verify.
 */
function verifyCanonicalSnapshotBytes(value, expectedCommitment = null, maxBytes = DEFAULT_MAX_CONTEXT_BYTES) {
  const bytes = asBuffer(value);
  positiveInteger(maxBytes, 'maxBytes', 1024);
  if (bytes.length > maxBytes) {
    throw new WorkContextSnapshotError(`context snapshot exceeds ${maxBytes} bytes`);
  }
  for (const byte of bytes) {
    if (byte >= 0x80) {
      throw new WorkContextSnapshotError('context snapshot is not canonical ASCII JSON');
    }
  }

  let payload;
  try { payload = JSON.parse(bytes.toString('ascii')); }
  catch (error) {
    throw new WorkContextSnapshotError(`context snapshot is not valid JSON: ${error.message}`);
  }
  if (!payload || Array.isArray(payload) || typeof payload !== 'object') {
    throw new WorkContextSnapshotError('context snapshot root must be an object');
  }
  if (payload.schema_version !== CONTEXT_SCHEMA_VERSION) {
    throw new WorkContextSnapshotError('unsupported context snapshot schema');
  }
  for (const field of VOLATILE_FIELDS) {
    if (Object.prototype.hasOwnProperty.call(payload, field)) {
      throw new WorkContextSnapshotError('context snapshot contains volatile host metadata');
    }
  }
  if (!payload.receipt || typeof payload.receipt !== 'object' || Array.isArray(payload.receipt)) {
    throw new WorkContextSnapshotError('context snapshot is missing its receipt');
  }
  if (payload.receipt.commitment_scope !== CONTEXT_COMMITMENT_SCOPE) {
    throw new WorkContextSnapshotError('unsupported context snapshot commitment scope');
  }
  const digest = payload.receipt.context_sha256;
  if (typeof digest !== 'string' || !DIGEST_RE.test(digest)) {
    throw new WorkContextSnapshotError('context snapshot is missing a valid context commitment');
  }
  if (expectedCommitment != null && digest !== String(expectedCommitment)) {
    throw new WorkContextSnapshotError('context snapshot does not match the expected commitment');
  }
  if (countContextShaKeys(payload) !== 1) {
    throw new WorkContextSnapshotError('context snapshot has an ambiguous context_sha256 field');
  }

  const field = Buffer.from(`\"context_sha256\":\"${digest}\"`, 'ascii');
  const first = bytes.indexOf(field);
  if (first < 0 || bytes.indexOf(field, first + 1) >= 0) {
    throw new WorkContextSnapshotError('context snapshot commitment field is not canonical');
  }
  let removeStart = first;
  let removeEnd = first + field.length;
  if (bytes[removeEnd] === 0x2c) { // comma after property
    removeEnd += 1;
  } else if (removeStart > 0 && bytes[removeStart - 1] === 0x2c) {
    removeStart -= 1;
  } else {
    throw new WorkContextSnapshotError('context snapshot commitment field is not a JSON property');
  }
  const preimage = Buffer.concat([bytes.subarray(0, removeStart), bytes.subarray(removeEnd)]);
  if (sha256(preimage) !== digest) {
    throw new WorkContextSnapshotError('context snapshot commitment is invalid');
  }
  return { payload, commitment: digest, bytes };
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
  CONTEXT_SCHEMA_VERSION,
  CONTEXT_COMMITMENT_SCOPE,
  DEFAULT_MAX_CONTEXT_BYTES,
  DEFAULT_MAX_SNAPSHOTS,
  DEFAULT_MAX_TOTAL_BYTES,
  WorkContextSnapshotError,
  WorkContextSnapshotStore,
  verifyCanonicalSnapshotBytes,
};
