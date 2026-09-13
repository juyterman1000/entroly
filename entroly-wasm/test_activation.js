const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, execFileSync } = require('child_process');

const {
  ACTIVE_FRESHNESS_SECONDS,
  activationStatus,
  parseHookInput,
  runHook,
} = require('./js/activation');

function run(command, args, options = {}) {
  return execFileSync(command, args, {
    encoding: 'utf8',
    windowsHide: true,
    ...options,
  });
}

function invokeCli(cliPath, payload, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [cliPath, 'activation', 'hook', '--host', 'codex'],
      { env, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] },
    );
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.stderr.on('data', chunk => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', code => resolve({ code, stdout, stderr }));
    child.stdin.end(JSON.stringify(payload));
  });
}

async function main() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-activation-test-'));
  const project = path.join(root, 'project');
  const state = path.join(root, 'state');
  const staleState = path.join(root, 'stale-state');
  const concurrentState = path.join(root, 'concurrent-state');
  fs.mkdirSync(path.join(project, 'src'), { recursive: true });
  fs.writeFileSync(
    path.join(project, 'src', 'auth.js'),
    'export function verifyToken(token) { return token.length > 20; }\n',
  );
  fs.writeFileSync(
    path.join(project, 'README.md'),
    'Authentication lives in src/auth.js and validates session tokens.\n',
  );
  run('git', ['init', '-q', project]);
  run('git', ['-C', project, 'add', '.']);
  run('git', [
    '-C', project, '-c', 'user.name=Entroly',
    '-c', 'user.email=entroly@example.invalid', 'commit', '-qm', 'init',
  ]);

  assert.deepStrictEqual(parseHookInput('{}'), {});
  assert.throws(() => parseHookInput('[]'), /JSON object/);
  assert.throws(
    () => parseHookInput(JSON.stringify({ prompt: 'x'.repeat(1024 * 1024) })),
    /exceeds 1 MiB/,
  );

  const prompt = `fix-auth-${crypto.randomBytes(12).toString('hex')}`;
  const session = `session-${crypto.randomBytes(12).toString('hex')}`;
  const payload = {
    hook_event_name: 'UserPromptSubmit',
    prompt,
    session_id: session,
    cwd: project,
  };

  const fake = runHook(payload, {
    host: 'codex',
    stateDir: state,
    selector: () => ({
      status: 'activated',
      sources: ['src/auth.js'],
      context: '<entroly-evidence source="src/auth.js">safe</entroly-evidence>',
      selectedTokens: 4,
      nativeEngine: true,
      detail: '',
    }),
  });
  assert.match(fake.hookSpecificOutput.additionalContext, /status: activated/);
  assert.match(fake.hookSpecificOutput.additionalContext, /src\/auth\.js/);

  // Read every receipt without relying on a platform-specific recursive entry type.
  const receiptFiles = [];
  const collect = directory => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) collect(absolute);
      else if (entry.name.endsWith('.json')) receiptFiles.push(absolute);
    }
  };
  receiptFiles.length = 0;
  collect(state);
  const persisted = receiptFiles.map(file => fs.readFileSync(file, 'utf8')).join('\n');
  assert(!persisted.includes(prompt), 'raw prompt must not be persisted');
  assert(!persisted.includes(session), 'raw session id must not be persisted');
  assert.match(persisted, /"prompt_persisted": false/);

  const stale = runHook(payload, {
    host: 'codex',
    stateDir: staleState,
    selector: () => ({
      status: 'activated', sources: ['src/auth.js'], context: 'safe',
      selectedTokens: 1, nativeEngine: true, detail: '',
    }),
  });
  assert(stale.hookSpecificOutput.additionalContext.includes('status: activated'));
  const staleReceipts = [];
  const collectStale = directory => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) collectStale(absolute);
      else if (entry.name.endsWith('.json')) staleReceipts.push(absolute);
    }
  };
  collectStale(staleState);
  const staleReceipt = JSON.parse(fs.readFileSync(staleReceipts[0], 'utf8'));
  staleReceipt.recorded_at_unix -= ACTIVE_FRESHNESS_SECONDS + 60;
  fs.writeFileSync(staleReceipts[0], JSON.stringify(staleReceipt));
  assert.strictEqual(activationStatus(project, staleState).state, 'stale');

  process.env.ENTROLY_DIR = state;
  const real = runHook({ ...payload, prompt: 'Fix token authentication in src auth' }, {
    host: 'codex',
    stateDir: state,
  });
  assert.match(real.hookSpecificOutput.additionalContext, /status: activated/);
  assert.match(real.hookSpecificOutput.additionalContext, /<entroly-evidence/);
  assert.match(real.hookSpecificOutput.additionalContext, /src\/auth\.js/);

  fs.appendFileSync(path.join(project, 'src', 'auth.js'), '// invalidate cache\n');
  const refreshed = runHook({ ...payload, prompt: 'Review token authentication' }, {
    host: 'codex',
    stateDir: state,
  });
  assert.match(refreshed.hookSpecificOutput.additionalContext, /status: activated/);

  const cliPath = path.join(__dirname, 'js', 'cli.js');
  const concurrentPayload = { ...payload, prompt: 'Locate token authentication' };
  const concurrentEnv = { ...process.env, ENTROLY_DIR: concurrentState };
  const concurrent = await Promise.all(
    Array.from({ length: 4 }, () => invokeCli(cliPath, concurrentPayload, concurrentEnv)),
  );
  for (const result of concurrent) {
    assert.strictEqual(result.code, 0, result.stderr);
    const parsed = JSON.parse(result.stdout);
    assert.strictEqual(parsed.hookSpecificOutput.hookEventName, 'UserPromptSubmit');
  }
  const concurrentStatus = activationStatus(project, path.join(concurrentState, 'activation'));
  assert.strictEqual(concurrentStatus.activation_events, 4);
  assert(concurrentStatus.effective_activation_events >= 1);

  const invalid = await invokeCli(cliPath, '{not-json', concurrentEnv);
  assert.strictEqual(invalid.code, 0);
  assert.match(JSON.parse(invalid.stdout).hookSpecificOutput.additionalContext, /failed before parsing/);

  fs.rmSync(root, { recursive: true, force: true });
  console.log('Activation hook tests passed');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
