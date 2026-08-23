// Entroly — Information-theoretic context optimization for JavaScript/TypeScript.
// 
// Usage:
//   const { EntrolyEngine, autoIndex, EntrolyConfig } = require('entroly-wasm');
//
//   const engine = new EntrolyEngine();
//   autoIndex(engine);   // auto-discover and ingest codebase
//   engine.optimize(8000, "fix the auth bug");
//
// Or as MCP server:
//   npx entroly-wasm serve

let WasmEntrolyEngine;
let WasmWorkGraph;
let WasmTrustEngine;
let classifyQueryTransitionRust;
let rewardWeightedOptimizeRust;
let optimizeTaskProfilesRust;
let classifyLearningQueryRust;
let wasmExports = {};
function bindWasmExports(mod) {
  wasmExports = mod;
  ({
    WasmEntrolyEngine,
    WasmWorkGraph,
    WasmTrustEngine,
    classify_query_transition: classifyQueryTransitionRust,
    reward_weighted_optimize: rewardWeightedOptimizeRust,
    optimize_task_profiles: optimizeTaskProfilesRust,
    classify_learning_query: classifyLearningQueryRust,
  } = mod);
}

function buildAndBindWasm() {
  const { execFileSync } = require('child_process');
  execFileSync('wasm-pack', ['build', '--target', 'nodejs', '--out-dir', 'pkg'], {
    cwd: __dirname,
    stdio: 'inherit',
  });
  const pkgPath = require.resolve('./pkg/entroly_wasm');
  delete require.cache[pkgPath];
  bindWasmExports(require('./pkg/entroly_wasm'));
}

try {
  bindWasmExports(require('./pkg/entroly_wasm'));
  if (
    !WasmWorkGraph ||
    !WasmTrustEngine ||
    !classifyQueryTransitionRust ||
    !rewardWeightedOptimizeRust ||
    !optimizeTaskProfilesRust ||
    !classifyLearningQueryRust ||
    !wasmExports.contextReceiptBuildJSON ||
    !wasmExports.recoveryHandleBuildJSON ||
    !wasmExports.memoryRecordBuildJSON
  ) {
    // A source checkout can contain an older generated pkg/ directory. Treat
    // a missing Work Graph export exactly like any other stale native surface
    // and rebuild before loading the high-level JS wrapper.
    buildAndBindWasm();
  }
} catch (err) {
  if (err && err.code === 'MODULE_NOT_FOUND') {
    // Source checkouts do not commit wasm-pack output. Build it lazily so
    // `node -e "require('./entroly-wasm')"` and local smoke tests exercise the
    // same module shape that the published npm tarball contains.
    buildAndBindWasm();
  } else {
    throw err;
  }
}
const { EntrolyConfig } = require('./js/config');
const { autoIndex, startIncrementalWatcher, estimateTokens } = require('./js/auto_index');
const { CheckpointManager, persistIndex, loadIndex } = require('./js/checkpoint');
const { EntrolyMCPServer } = require('./js/server');
const { runAutotune, startAutotuneDaemon, TaskProfileOptimizer, FeedbackJournal } = require('./js/autotune');
const { ValueTracker, EVOLUTION_TAX_RATE, estimateCost } = require('./js/value_tracker');
const { exportPromoted: exportAgentSkills } = require('./js/agentskills_export');
const { TelegramGateway, DiscordGateway, SlackGateway } = require('./js/gateways');
const { VaultObserver } = require('./js/vault_observer');
const { TrustEngine } = require('./js/trust_engine');
const {
  WorkGraph,
  RepositoryDiscoveryError,
  discoverRepositoryIdentity,
  discoverRepositoryObservation,
} = require('./js/work_graph');
const {
  WorkGraphLockTimeout,
  WorkGraphStateError,
  WorkGraphStore,
  WorkGraphStoreError,
} = require('./js/work_graph_store');
const {
  MAX_RESUME_EVIDENCE,
  handoffRepository,
  resumeRepository,
} = require('./js/work_graph_continuity');
const {
  createContextReceipt,
  explainReceiptOmission,
  ingestReceiptDocuments,
  renderContextReceipt,
  selectReceiptContext,
} = require('./js/context_receipts');
const {
  createEntrolyMiddleware,
  optimizeAnthropicParams,
  optimizeGeminiParams,
  optimizeMessages,
  optimizeOpenAIParams,
  optimizeRequestParams,
  stableStringify,
  wrapAnthropic,
  wrapGemini,
  wrapOpenAI,
} = require('./js/app_sdk');

function classifyQueryTransition(...args) {
  if (!classifyQueryTransitionRust) {
    throw new Error('Rust classify_query_transition is unavailable; rebuild entroly-wasm');
  }
  return classifyQueryTransitionRust(...args);
}

function rewardWeightedOptimize(...args) {
  if (!rewardWeightedOptimizeRust) {
    throw new Error('Rust reward_weighted_optimize is unavailable; rebuild entroly-wasm');
  }
  return rewardWeightedOptimizeRust(...args);
}

function optimizeTaskProfiles(...args) {
  if (!optimizeTaskProfilesRust) {
    throw new Error('Rust optimize_task_profiles is unavailable; rebuild entroly-wasm');
  }
  return optimizeTaskProfilesRust(...args);
}

function classifyLearningQuery(...args) {
  if (!classifyLearningQueryRust) {
    throw new Error('Rust classify_learning_query is unavailable; rebuild entroly-wasm');
  }
  return classifyLearningQueryRust(...args);
}

module.exports = {
  // The TypeScript surface re-exports the generated bindings. Mirror that at
  // runtime so package-root imports and `index.d.ts` describe the same API.
  ...wasmExports,

  // Core engine (wasm)
  EntrolyEngine: WasmEntrolyEngine,
  WasmEntrolyEngine,
  classifyQueryTransition,
  rewardWeightedOptimize,
  optimizeTaskProfiles,
  classifyLearningQuery,

  // Shared Rust evidence-bounded Trust Engine.
  TrustEngine,

  // Shared Rust AI Work Graph with Node-only repository/persistence glue.
  WorkGraph,
  RepositoryDiscoveryError,
  discoverRepositoryIdentity,
  discoverRepositoryObservation,
  WorkGraphStore,
  WorkGraphStoreError,
  WorkGraphLockTimeout,
  WorkGraphStateError,
  MAX_RESUME_EVIDENCE,
  handoffRepository,
  resumeRepository,

  // Configuration
  EntrolyConfig,

  // Codebase scanning
  autoIndex,
  startIncrementalWatcher,
  estimateTokens,

  // State persistence
  CheckpointManager,
  persistIndex,
  loadIndex,

  // MCP Server
  EntrolyMCPServer,

  // Autotune + Task-Conditioned Profiles
  runAutotune,
  startAutotuneDaemon,
  TaskProfileOptimizer,
  FeedbackJournal,

  // Provider-classified evolution budget (C_spent ≤ τ·S_provider(t))
  ValueTracker,
  EVOLUTION_TAX_RATE,
  estimateCost,

  // agentskills.io portable export
  exportAgentSkills,

  // Chat gateways (live self-evolution event stream)
  TelegramGateway,
  DiscordGateway,
  SlackGateway,

  // Observe the shared vault — works with skills promoted by the
  // Python daemon OR any node-side orchestrator. No daemon required.
  VaultObserver,

  // App SDK: middleware shape plus native provider wrappers.
  createEntrolyMiddleware,
  optimizeAnthropicParams,
  optimizeGeminiParams,
  optimizeMessages,
  optimizeOpenAIParams,
  optimizeRequestParams,
  stableStringify,
  wrapAnthropic,
  wrapGemini,
  wrapOpenAI,

  // Context Receipts: auditable context selection receipts for JS users.
  ingestReceiptDocuments,
  selectReceiptContext,
  createContextReceipt,
  renderContextReceipt,
  explainReceiptOmission,
};
