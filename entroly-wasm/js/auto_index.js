// Entroly Auto-Index — JS port of auto_index.py
// Git-aware codebase discovery and ingestion.

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const SUPPORTED_EXTENSIONS = new Set([
  // Systems
  '.rs', '.c', '.cpp', '.h', '.hpp', '.cc', '.hxx', '.zig',
  // Web / JS / TS
  '.js', '.ts', '.jsx', '.tsx', '.mjs', '.mts', '.cjs', '.cts', '.vue', '.svelte',
  // Python
  '.py', '.pyi',
  // JVM
  '.java', '.kt', '.scala',
  // .NET
  '.cs', '.csx', '.fs',
  // Go
  '.go',
  // Swift
  '.swift',
  // Ruby
  '.rb',
  // PHP
  '.php',
  // Dart
  '.dart',
  // Elixir
  '.ex', '.exs',
  // Lua
  '.lua',
  // R
  '.r',
  // Shell / Config
  '.sh', '.bash', '.zsh', '.toml', '.yaml', '.yml', '.json',
  // Terraform
  '.tf', '.hcl',
  // Docs
  '.md', '.rst',
  // SQL
  '.sql',
  // Docker
  '.dockerfile',
]);

const SKIP_PATTERNS = new Set([
  'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'Cargo.lock',
  'poetry.lock', 'Pipfile.lock', 'composer.lock', 'Gemfile.lock',
  '.DS_Store', 'thumbs.db',
]);

const BINARY_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp', '.tiff',
  '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac', '.ogg', '.webm',
  '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz',
  '.wasm', '.so', '.dll', '.dylib', '.a', '.o', '.obj', '.exe', '.bin',
  '.pyc', '.pyo', '.class', '.jar',
  '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
  '.ttf', '.otf', '.woff', '.woff2', '.eot',
  '.db', '.sqlite', '.sqlite3',
  '.dat', '.pak', '.map',
]);

const MAX_FILE_BYTES = 50 * 1024;
const ABSOLUTE_MAX_BYTES = 500 * 1024;
const MAX_FILES = parseInt(process.env.ENTROLY_MAX_FILES || '5000', 10);

let ignorePatterns = [];

function gitLsFiles(projectDir) {
  try {
    const result = execSync('git ls-files --cached --others --exclude-standard', {
      cwd: projectDir, encoding: 'utf-8', timeout: 10000, stdio: ['pipe', 'pipe', 'pipe'],
    });
    return result.split('\n').map(f => f.trim()).filter(Boolean);
  } catch {
    return [];
  }
}

function walkFallback(projectDir) {
  const files = [];
  const skipDirs = new Set(['node_modules', '__pycache__', 'target', 'dist', 'build', '.git', 'venv', '.venv', 'env']);

  function walk(dir) {
    if (files.length >= MAX_FILES) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      if (files.length >= MAX_FILES) return;
      if (entry.name.startsWith('.') || skipDirs.has(entry.name)) {
        if (entry.isDirectory()) continue;
      }
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) { walk(full); }
      else { files.push(path.relative(projectDir, full)); }
    }
  }
  walk(projectDir);
  return files;
}

function loadEntrolyIgnore(projectDir) {
  const ignorePath = path.join(projectDir, '.entrolyignore');
  try {
    const content = fs.readFileSync(ignorePath, 'utf-8');
    return content.split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
  } catch { return []; }
}

function matchesIgnore(relPath) {
  const { minimatch } = (() => {
    try { return require('minimatch'); } catch { return { minimatch: null }; }
  })();
  for (const pattern of ignorePatterns) {
    if (minimatch) {
      if (minimatch(relPath, pattern)) return true;
      if (minimatch(path.basename(relPath), pattern)) return true;
    } else {
      // Simple glob fallback
      if (relPath.includes(pattern.replace('*', '')) || path.basename(relPath).includes(pattern.replace('*', ''))) return true;
    }
  }
  return false;
}

function shouldIndex(relPath) {
  const basename = path.basename(relPath);
  if (SKIP_PATTERNS.has(basename)) return false;
  const ext = path.extname(basename).toLowerCase();
  if (BINARY_EXTENSIONS.has(ext)) return false;
  if (ignorePatterns.length && matchesIgnore(relPath)) return false;
  if (basename.startsWith('Dockerfile')) return true;
  return SUPPORTED_EXTENSIONS.has(ext);
}

function estimateTokens(content) {
  return Math.max(1, Math.floor(content.length / 4));
}

/**
 * Auto-index a project's codebase into the Entroly wasm engine.
 * @param {WasmEntrolyEngine} engine
 * @param {string} [projectDir]
 * @param {boolean} [force=false]
 * @returns {object} Summary with indexed file count, tokens, and duration.
 */
function autoIndex(engine, projectDir, force = false) {
  projectDir = projectDir || process.cwd();
  projectDir = path.resolve(projectDir);

  ignorePatterns = loadEntrolyIgnore(projectDir);

  if (!force && engine.fragment_count() > 0) {
    return {
      status: 'skipped', reason: 'persistent_index_loaded',
      existing_fragments: engine.fragment_count(),
    };
  }

  const t0 = Date.now();

  let files = gitLsFiles(projectDir);
  let discovery = 'git';
  if (!files.length) { files = walkFallback(projectDir); discovery = 'walk'; }

  const allIndexable = files.filter(shouldIndex);
  const indexable = allIndexable.slice(0, MAX_FILES);

  let indexed = 0, totalTokens = 0, skippedSize = 0, skippedRead = 0;

  for (const relPath of indexable) {
    const absPath = path.join(projectDir, relPath);
    let stat;
    try { stat = fs.statSync(absPath); } catch { continue; }

    if (stat.size > ABSOLUTE_MAX_BYTES) { skippedSize++; continue; }
    if (stat.size > MAX_FILE_BYTES || stat.size === 0) { skippedSize++; continue; }

    // Binary detection
    try {
      const fd = fs.openSync(absPath, 'r');
      const buf = Buffer.alloc(Math.min(8192, stat.size));
      fs.readSync(fd, buf, 0, buf.length, 0);
      fs.closeSync(fd);
      if (buf.includes(0)) { skippedRead++; continue; }
    } catch { skippedRead++; continue; }

    let content;
    try { content = fs.readFileSync(absPath, 'utf-8'); } catch { skippedRead++; continue; }
    if (!content.trim()) continue;

    const tokens = estimateTokens(content);
    engine.ingest(content, `file:${relPath}`, tokens, false);
    indexed++;
    totalTokens += tokens;
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(2);

  // Trigger dep graph build
  if (indexed > 0) {
    try { engine.optimize(1, ''); } catch {}
  }

  return {
    status: 'indexed', files_indexed: indexed, total_tokens: totalTokens,
    duration_s: parseFloat(elapsed), discovery_method: discovery,
    skipped_too_large: skippedSize, skipped_unreadable: skippedRead,
    project_dir: projectDir,
  };
}

/**
 * Start incremental file watcher (background interval).
 * @param {WasmEntrolyEngine} engine
 * @param {string} [projectDir]
 * @param {number} [intervalMs=120000]
 * @returns {NodeJS.Timeout}
 */
function startIncrementalWatcher(engine, projectDir, intervalMs = 120000) {
  projectDir = projectDir || process.cwd();
  projectDir = path.resolve(projectDir);
  const indexedMtimes = {};

  // Initial snapshot
  const files = gitLsFiles(projectDir);
  for (const rel of files) {
    try { indexedMtimes[rel] = fs.statSync(path.join(projectDir, rel)).mtimeMs; } catch {}
  }

  return setInterval(() => {
    try {
      const files = gitLsFiles(projectDir);
      let count = 0;
      for (const rel of files) {
        if (!shouldIndex(rel)) continue;
        const abs = path.join(projectDir, rel);
        let mtime;
        try { mtime = fs.statSync(abs).mtimeMs; } catch { continue; }
        if (indexedMtimes[rel] === undefined || mtime > indexedMtimes[rel]) {
          indexedMtimes[rel] = mtime;
          try {
            const stat = fs.statSync(abs);
            if (stat.size > MAX_FILE_BYTES || stat.size === 0) continue;
            const content = fs.readFileSync(abs, 'utf-8');
            if (!content.trim()) continue;
            engine.ingest(content, `file:${rel}`, estimateTokens(content), false);
            count++;
          } catch { continue; }
        }
        if (count >= 100) break;
      }
      if (count > 0) console.error(`[entroly] Incremental re-index: ${count} new/modified files`);
    } catch (e) {
      console.error(`[entroly] Incremental re-index error: ${e.message}`);
    }
  }, intervalMs);
}

module.exports = { autoIndex, startIncrementalWatcher, estimateTokens };
