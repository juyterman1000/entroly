#!/usr/bin/env node
// Entroly MCP Server — JS port of server.py
// Pure-wasm MCP server — no Python dependency.
//
// Architecture: MCP Client → JSON-RPC (stdio) → Node.js → Wasm Engine → Results

const { WasmEntrolyEngine } = require('../pkg/entroly_wasm');
const { EntrolyConfig } = require('./config');
const { CheckpointManager, persistIndex, loadIndex } = require('./checkpoint');
const { autoIndex, startIncrementalWatcher } = require('./auto_index');
const { startAutotuneDaemon, FeedbackJournal, TaskProfileOptimizer } = require('./autotune');
const { VaultManager } = require('./vault');
const { getTracker } = require('./value_tracker');
const { EpistemicRouter, BeliefCompiler, VerificationEngine, ChangePipeline, FlowOrchestrator, compileDocs, exportTrainingData } = require('./cogops');
const { WorkspaceChangeListener } = require('./workspace');
const { SkillEngine } = require('./skills');
const { buildRepoMap, renderRepoMapMarkdown } = require('./repo_map');
const { ingestDiff, ingestDiagram, ingestVoice } = require('./multimodal');
const path = require('path');
const fs = require('fs');

function resolveProjectDirectory(root, candidate = '.') {
  try {
    const realRoot = fs.realpathSync(root);
    const resolved = fs.realpathSync(path.isAbsolute(candidate) ? candidate : path.join(realRoot, candidate));
    const relative = path.relative(realRoot, resolved);
    if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) return null;
    return fs.statSync(resolved).isDirectory() ? resolved : null;
  } catch {
    return null;
  }
}

function resolveProjectOutput(root, candidate) {
  try {
    const realRoot = fs.realpathSync(root);
    const output = path.isAbsolute(candidate) ? candidate : path.join(realRoot, candidate);
    const parent = fs.realpathSync(path.dirname(output));
    const relative = path.relative(realRoot, parent);
    if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) return null;
    if (!fs.existsSync(output)) return output;
    const resolved = fs.realpathSync(output);
    const outputRelative = path.relative(realRoot, resolved);
    if (outputRelative === '..' || outputRelative.startsWith(`..${path.sep}`) || path.isAbsolute(outputRelative)) return null;
    return fs.statSync(resolved).isFile() ? resolved : null;
  } catch {
    return null;
  }
}

// ── MCP Protocol Implementation (stdio JSON-RPC 2.0) ──

class EntrolyMCPServer {
  constructor(config) {
    this.config = config || new EntrolyConfig();
    this.engine = new WasmEntrolyEngine();
    this.turnCounter = 0;

    // Shared cross-runtime telemetry sink — SAME file/schema/dir as the
    // Python tracker, so npm users' value shows up on the same dashboard
    // as pip/MCP/SDK users. Fail-open: never let telemetry break a tool.
    try { this.valueTracker = getTracker(); } catch (_) { this.valueTracker = null; }

    // Feedback journal — persists episodes for cross-session autotune
    this.feedbackJournal = new FeedbackJournal(this.config.checkpointDir);
    // Task-conditioned weight profiles (novel: per-task-type optimization)
    this.taskProfiles = new TaskProfileOptimizer(this.feedbackJournal);
    this.taskProfiles.optimizeAll(); // warm from existing journal
    // Track last optimization context for feedback attribution
    this._lastOptCtx = null;

    // Checkpoint manager
    this.checkpointMgr = new CheckpointManager(
      this.config.checkpointDir, this.config.autoCheckpointInterval
    );

    // Index path for persistent repo-level indexing
    this.indexPath = path.join(this.config.checkpointDir, 'index.json.gz');

    // CogOps layer — epistemic engine, vault, compiler, verifier, etc.
    const vaultBase = process.env.ENTROLY_VAULT || path.join(process.env.ENTROLY_DIR || path.join(process.cwd(), '.entroly'), 'vault');
    this.vault = new VaultManager(vaultBase);
    this.epistemicRouter = new EpistemicRouter(this.vault, { missThreshold: 3, freshnessHours: 24, minConfidence: 0.6 });
    this.beliefCompiler = new BeliefCompiler(this.vault);
    this.verifier = new VerificationEngine(this.vault, { freshnessHours: 24, minConfidence: 0.5 });
    this.changePipe = new ChangePipeline(this.vault, this.verifier);
    this.skillEngine = new SkillEngine(this.vault);
    this.sourceDir = fs.realpathSync(process.env.ENTROLY_SOURCE || process.cwd());
    this.flowOrchestrator = new FlowOrchestrator(this.vault, this.epistemicRouter, this.beliefCompiler, this.verifier, this.changePipe, this.sourceDir);
    this.workspaceListener = new WorkspaceChangeListener(this.vault, this.beliefCompiler, this.verifier, this.changePipe, this.sourceDir);

    // Try loading persistent index
    if (loadIndex(this.engine, this.indexPath)) {
      const n = this.engine.fragment_count();
      this._log(`Loaded persistent index: ${n} fragments`);
    }

    this._buffer = '';
    this._initialized = false;
  }

  _log(msg) { process.stderr.write(`${new Date().toISOString()} [entroly] ${msg}\n`); }

  _projectDir(candidate = '') {
    const resolved = resolveProjectDirectory(this.sourceDir, candidate || '.');
    if (!resolved) throw new Error(`Path must remain within project root: ${candidate}`);
    return resolved;
  }

  _projectOutput(candidate) {
    const resolved = resolveProjectOutput(this.sourceDir, candidate);
    if (!resolved) throw new Error(`Path must remain within project root: ${candidate}`);
    return resolved;
  }

  // ── MCP Tool Definitions (36 tools — full parity with pip) ──
  get tools() {
    return [
      // ── Core Engine Tools (12) ──
      { name: 'remember_fragment', description: 'Store a context fragment with automatic dedup and entropy scoring.', inputSchema: { type: 'object', properties: { content: { type: 'string', description: 'Text content to store' }, source: { type: 'string', description: 'Origin label (e.g. file:utils.py)', default: '' }, token_count: { type: 'integer', description: 'Token count (auto if 0)', default: 0 }, is_pinned: { type: 'boolean', description: 'Always include in optimized context', default: false } }, required: ['content'] } },
      { name: 'optimize_context', description: 'Select the optimal context subset for a token budget. Uses IOS + PRISM RL + Channel Coding.', inputSchema: { type: 'object', properties: { token_budget: { type: 'integer', description: 'Max tokens (default 128K)', default: 128000 }, query: { type: 'string', description: 'Current task/query for semantic scoring', default: '' } } } },
      { name: 'recall_relevant', description: 'Semantic recall of most relevant stored fragments.', inputSchema: { type: 'object', properties: { query: { type: 'string', description: 'Search query' }, top_k: { type: 'integer', description: 'Number of results', default: 5 } }, required: ['query'] } },
      { name: 'record_outcome', description: 'Record whether selected fragments led to success/failure (RL feedback).', inputSchema: { type: 'object', properties: { fragment_ids: { type: 'string', description: 'Comma-separated fragment IDs' }, success: { type: 'boolean', description: 'True if output was good', default: true } }, required: ['fragment_ids'] } },
      { name: 'explain_context', description: 'Explain why each fragment was included/excluded in last optimization.', inputSchema: { type: 'object', properties: {} } },
      { name: 'get_stats', description: 'Get comprehensive session statistics.', inputSchema: { type: 'object', properties: {} } },
      { name: 'checkpoint_state', description: 'Save current state for crash recovery.', inputSchema: { type: 'object', properties: { task_description: { type: 'string', default: '' } } } },
      { name: 'resume_state', description: 'Resume from the latest checkpoint.', inputSchema: { type: 'object', properties: {} } },
      { name: 'analyze_codebase_health', description: 'Analyze codebase health (grade A-F).', inputSchema: { type: 'object', properties: {} } },
      { name: 'security_report', description: 'Session-wide security audit across all ingested fragments.', inputSchema: { type: 'object', properties: {} } },
      { name: 'entroly_dashboard', description: 'Show live value metrics: money saved, performance, bloat prevention, quality.', inputSchema: { type: 'object', properties: {} } },
      { name: 'scan_for_vulnerabilities', description: 'Scan code for security vulnerabilities (SAST).', inputSchema: { type: 'object', properties: { content: { type: 'string', description: 'Source code to scan' }, source: { type: 'string', description: 'File path', default: 'unknown' } }, required: ['content'] } },
      // ── CogOps Epistemic Tools (9) ──
      { name: 'epistemic_route', description: 'Route a query through the CogOps Epistemic Ingress Controller. Inspects 4 signals (intent, belief coverage, freshness, risk) and selects one of 5 canonical flows.', inputSchema: { type: 'object', properties: { query: { type: 'string', description: 'The user query or event description' }, is_event: { type: 'boolean', description: 'True if this is a change-driven event', default: false }, event_type: { type: 'string', description: 'Type of event (pr, commit, release, incident, scheduled)', default: '' } }, required: ['query'] } },
      { name: 'vault_status', description: 'Show the current state of the CogOps Knowledge Vault: total beliefs, verification status, confidence distribution, and routing statistics.', inputSchema: { type: 'object', properties: {} } },
      { name: 'vault_write_belief', description: 'Write a belief artifact to the CogOps Knowledge Vault with machine-auditable frontmatter.', inputSchema: { type: 'object', properties: { entity: { type: 'string', description: 'System entity this belief is about (e.g. auth::token_rotation)' }, title: { type: 'string', description: 'Human-readable title' }, body: { type: 'string', description: 'Belief content (markdown)' }, confidence: { type: 'number', description: 'Machine-assigned confidence 0.0-1.0', default: 0.7 }, status: { type: 'string', description: 'observed|inferred|verified|stale|hypothesis', default: 'inferred' }, sources: { type: 'string', description: 'Comma-separated source paths', default: '' }, derived_from: { type: 'string', description: 'Comma-separated component names', default: '' } }, required: ['entity', 'title', 'body'] } },
      { name: 'vault_query', description: 'Query the CogOps Knowledge Vault for existing beliefs. Supports lookup by entity name or listing all.', inputSchema: { type: 'object', properties: { entity: { type: 'string', description: 'Entity name to look up (fuzzy match)', default: '' }, list_all: { type: 'boolean', description: 'If true, return all beliefs with frontmatter summary', default: false } } } },
      { name: 'vault_write_action', description: 'Write a task output or report to the CogOps Knowledge Vault (actions/).', inputSchema: { type: 'object', properties: { title: { type: 'string', description: 'Title of the output' }, content: { type: 'string', description: 'Full markdown content' }, action_type: { type: 'string', description: 'Type tag (report, pr_brief, answer, diagram, context_pack)', default: 'report' } }, required: ['title', 'content'] } },
      { name: 'vault_search', description: 'Full-text search across all belief artifacts in the vault. Uses keyword matching with entity-name boosting.', inputSchema: { type: 'object', properties: { query: { type: 'string', description: 'Natural language search query' }, top_k: { type: 'integer', description: 'Maximum results', default: 5 } }, required: ['query'] } },
      { name: 'compile_beliefs', description: 'Compile source code into belief artifacts (Truth → Belief pipeline). Scans directory for source files, extracts entities, resolves dependencies, writes beliefs.', inputSchema: { type: 'object', properties: { directory: { type: 'string', description: 'Path to scan. Defaults to project root.', default: '' }, max_files: { type: 'integer', description: 'Maximum files to process', default: 200 } } } },
      { name: 'verify_beliefs', description: 'Run a full verification pass on all beliefs. Checks staleness, contradictions, confidence divergence, low confidence scores.', inputSchema: { type: 'object', properties: {} } },
      // ── CogOps Analysis & Change Tools (7) ──
      { name: 'blast_radius', description: 'Analyze the blast radius of file changes on existing beliefs. Returns affected beliefs, risk level, and recommendations.', inputSchema: { type: 'object', properties: { changed_files: { type: 'string', description: 'Comma-separated list of changed file paths' } }, required: ['changed_files'] } },
      { name: 'coverage_gaps', description: 'Find source files with no corresponding belief in the vault. Identifies blind spots before running compile_beliefs.', inputSchema: { type: 'object', properties: { directory: { type: 'string', description: 'Path to scan. Defaults to project root.', default: '' } } } },
      { name: 'process_change', description: 'Process a code change through the Change-Driven pipeline (Flow ④). Diff → ChangeSet → Review → Blast Radius → Vault.', inputSchema: { type: 'object', properties: { diff_text: { type: 'string', description: 'Raw unified diff text' }, commit_message: { type: 'string', description: 'Optional commit message', default: '' }, pr_title: { type: 'string', description: 'Optional PR title', default: '' } }, required: ['diff_text'] } },
      { name: 'execute_flow', description: 'Execute a full canonical epistemic flow end-to-end. Routes query through the Ingress Controller then chains the appropriate pipeline.', inputSchema: { type: 'object', properties: { query: { type: 'string', description: 'The user query or event description' }, diff_text: { type: 'string', description: 'Raw diff for change-driven flows', default: '' }, is_event: { type: 'boolean', description: 'True if change-driven event', default: false }, event_type: { type: 'string', description: 'Type of event', default: '' } }, required: ['query'] } },
      { name: 'refresh_beliefs', description: 'Mark beliefs as stale after file changes so the next verify pass will flag them for re-compilation.', inputSchema: { type: 'object', properties: { changed_files: { type: 'string', description: 'Comma-separated list of changed file paths' } }, required: ['changed_files'] } },
      { name: 'compile_docs', description: 'Compile markdown documentation files into belief artifacts with confidence 0.80 (human-authored > machine-inferred).', inputSchema: { type: 'object', properties: { directory: { type: 'string', description: 'Project root to scan. Defaults to project root.', default: '' }, max_files: { type: 'integer', description: 'Maximum doc files to process', default: 50 } } } },
      { name: 'export_training_data', description: 'Export vault beliefs as JSONL training data for LLM finetuning. Filters out stale and low-confidence beliefs.', inputSchema: { type: 'object', properties: { output_path: { type: 'string', description: 'Path to write JSONL file', default: 'training_data.jsonl' }, format: { type: 'string', description: 'Output format (jsonl)', default: 'jsonl' } } } },
      // ── Ingestion Tools (3) ──
      { name: 'ingest_diff', description: 'Ingest a code diff/patch into context memory. Converts unified diff into structured change summary with intent classification.', inputSchema: { type: 'object', properties: { diff_text: { type: 'string', description: 'Raw unified diff text' }, source: { type: 'string', description: 'Identifier (e.g. pr_42_auth_refactor.diff)' }, commit_message: { type: 'string', description: 'Optional commit message', default: '' } }, required: ['diff_text', 'source'] } },
      { name: 'ingest_diagram', description: 'Ingest an image/diagram into context memory (requires PIL/OCR — not available in WASM runtime, use pip install entroly).', inputSchema: { type: 'object', properties: { image_path: { type: 'string', description: 'Path to image file' } }, required: ['image_path'] } },
      { name: 'ingest_voice', description: 'Ingest audio into context memory (requires whisper — not available in WASM runtime, use pip install entroly).', inputSchema: { type: 'object', properties: { audio_path: { type: 'string', description: 'Path to audio file' } }, required: ['audio_path'] } },
      // ── Workspace Tools (2) ──
      { name: 'sync_workspace_changes', description: 'Synchronize workspace file changes into the belief and verification layers. Detects new/modified/deleted files, marks beliefs stale, recompiles.', inputSchema: { type: 'object', properties: { directory: { type: 'string', description: 'Project directory', default: '' }, force: { type: 'boolean', description: 'Force re-scan all files', default: false }, max_files: { type: 'integer', description: 'Maximum files to process', default: 100 } } } },
      { name: 'start_workspace_listener', description: 'Start a background workspace listener that continuously feeds repo changes into CogOps belief CI.', inputSchema: { type: 'object', properties: { directory: { type: 'string', description: 'Project directory', default: '' }, interval_s: { type: 'integer', description: 'Scan interval in seconds', default: 120 }, force_initial: { type: 'boolean', description: 'Force initial full scan', default: false }, max_files: { type: 'integer', description: 'Max files per scan', default: 100 } } } },
      // ── Skills Tools (2) ──
      { name: 'create_skill', description: 'Create a new skill from a capability gap (Evolution layer). Generates SKILL.md, tool.py, metrics.json, and test cases.', inputSchema: { type: 'object', properties: { entity_key: { type: 'string', description: 'Entity this skill handles (e.g. protobuf_analysis)' }, failing_queries: { type: 'string', description: 'Pipe-separated list of failing queries' }, intent: { type: 'string', description: 'Intent class for this skill', default: '' } }, required: ['entity_key', 'failing_queries'] } },
      { name: 'manage_skills', description: 'Manage CogOps skill lifecycle (Evolution layer). Actions: list, benchmark, promote.', inputSchema: { type: 'object', properties: { action: { type: 'string', description: 'list | benchmark | promote', default: 'list' }, skill_id: { type: 'string', description: 'Required for benchmark/promote', default: '' } } } },
      // ── Analysis Tools (2) ──
      { name: 'repo_file_map', description: 'Return the canonical Entroly file map across Python, Rust core, and WASM repos with ownership roles.', inputSchema: { type: 'object', properties: { format: { type: 'string', description: 'Output format: markdown or json', default: 'markdown' } } } },
      { name: 'prefetch_related', description: 'Predict which files the LLM will need next based on co-access patterns and dependency graph.', inputSchema: { type: 'object', properties: { file_path: { type: 'string', description: 'Current file being worked on' }, source_content: { type: 'string', description: 'Content of current file for import analysis', default: '' }, language: { type: 'string', description: 'Programming language', default: '' } }, required: ['file_path'] } },
      // ── Hallucination Detection Tool ──
      { name: 'verify_response', description: 'Verify an AI-generated response for hallucination. Computes entity coverage gap, hedging curvature, and entropy consistency. Returns fused_risk [0.0=safe, 1.0=hallucinated], verdict (pass/warn/flag), and flagged claims. All local — zero LLM calls.', inputSchema: { type: 'object', properties: { response: { type: 'string', description: 'The AI-generated text to verify' }, context: { type: 'string', description: 'Source context provided to the AI', default: '' }, prompt: { type: 'string', description: 'Original user prompt', default: '' } }, required: ['response'] } },
    ];
  }

  // ── Tool Handlers (36 tools) ──
  handleTool(name, args) {
    switch (name) {
      // ── Core Engine Tools ──
      case 'remember_fragment':
        return this.engine.ingest(args.content, args.source || '', args.token_count || 0, args.is_pinned || false);

      case 'optimize_context': {
        this.turnCounter++;
        this.engine.advance_turn();
        const budget = args.token_budget || this.config.defaultTokenBudget;
        const query = args.query || '';
        const profile = this.taskProfiles.applyToEngine(this.engine, query);
        const result = this.engine.optimize(budget, query);
        result._taskProfile = { taskType: profile.taskType, confidence: profile.confidence };
        // Record value to the shared sink (mirrors entroly/server.py).
        try {
          const ts = (result && result.tokens_saved) || 0;
          if (this.valueTracker && ts > 0) {
            this.valueTracker.record({
              tokensSaved: ts,
              model: (result && result.model) || '',
              duplicates: (result && result.duplicates_caught) || 0,
              optimized: true,
            });
          }
        } catch (_) { /* never fail optimization for telemetry */ }
        const state = this.engine.export_state();
        this._lastOptCtx = { weights: { w_r: state.w_recency, w_f: state.w_frequency, w_s: state.w_semantic, w_e: state.w_entropy }, selectedSources: (result.selected || []).map(s => s.source).filter(Boolean), selectedCount: result.selected_count || 0, tokenBudget: budget, query, turn: this.turnCounter };
        if (this.checkpointMgr.shouldAutoCheckpoint()) { try { persistIndex(this.engine, this.indexPath); this.checkpointMgr.save({ engine_state: state, turn: this.turnCounter }); } catch {} }
        return result;
      }

      case 'recall_relevant':
        // Use BM25-backed relevance recall, not SimHash near-duplicate recall.
        // This mirrors the Python MCP server's recall_relevant path.
        return this.engine.recall_auto(args.query, args.top_k || 5);

      case 'record_outcome': {
        const ids = (args.fragment_ids || '').split(',').map(s => s.trim()).filter(Boolean);
        const idsJson = JSON.stringify(ids);
        const success = args.success !== false;
        if (success) this.engine.record_success(idsJson); else this.engine.record_failure(idsJson);
        if (this._lastOptCtx) { this.feedbackJournal.log({ ...this._lastOptCtx, reward: success ? 1.0 : -1.0 }); }
        return { status: 'recorded', fragment_ids: ids, outcome: success ? 'success' : 'failure' };
      }

      case 'explain_context':
        return this.engine.explain_selection();

      case 'get_stats': {
        const stats = this.engine.stats();
        stats.checkpoint = this.checkpointMgr.stats();
        return stats;
      }

      case 'checkpoint_state': {
        try { persistIndex(this.engine, this.indexPath); const p = this.checkpointMgr.save({ engine_state: this.engine.export_state(), turn: this.turnCounter, task: args.task_description || '' }); return { status: 'checkpoint_saved', path: p }; }
        catch (e) { return { status: 'error', error: e.message }; }
      }

      case 'resume_state': {
        const ckpt = this.checkpointMgr.loadLatest();
        if (!ckpt) return { status: 'no_checkpoint_found' };
        if (ckpt.engine_state) this.engine.import_state(JSON.stringify(ckpt.engine_state));
        return { status: 'resumed', checkpoint_id: ckpt.checkpoint_id, turn: ckpt.turn || 0 };
      }

      case 'analyze_codebase_health':
        return this.engine.analyze_health();

      case 'security_report':
        return this.engine.security_report();

      case 'scan_for_vulnerabilities':
        return this.engine.scan_fragment(args.source || 'unknown');

      case 'entroly_dashboard': {
        const stats = this.engine.stats();
        const explanation = this.engine.explain_selection();
        // npm users live in their MCP client (no Python dashboard), so
        // surface the persisted cross-session value here directly.
        let value = null;
        try {
          if (this.valueTracker) {
            const tr = this.valueTracker.getTrends();
            value = {
              lifetime: tr.lifetime,
              today: (tr.daily.slice(-1)[0]) || null,
              recent_activity: tr.activity.slice(0, 10),
              data_source: 'shared telemetry file (cross-runtime)',
            };
          }
        } catch (_) { /* fail-open */ }
        return { stats, explanation, value, turn: this.turnCounter };
      }

      // ── CogOps Epistemic Tools ──
      case 'epistemic_route':
        return this.epistemicRouter.route(args.query, args.is_event || false, args.event_type || '');

      case 'vault_status': {
        const init = this.vault.ensureStructure();
        const coverage = this.vault.coverageIndex();
        const routing = this.epistemicRouter.stats();
        return { vault: init, coverage, routing };
      }

      case 'vault_write_belief': {
        const { randomUUID } = require('crypto');
        const artifact = {
          claim_id: randomUUID(),
          entity: args.entity,
          title: args.title,
          body: args.body,
          confidence: args.confidence ?? 0.7,
          status: args.status || 'inferred',
          sources: args.sources ? args.sources.split(',').map(s => s.trim()).filter(Boolean) : [],
          last_checked: new Date().toISOString(),
          derived_from: args.derived_from ? args.derived_from.split(',').map(s => s.trim()).filter(Boolean) : [],
        };
        const result = this.vault.writeBelief(artifact);
        result.artifact = { claim_id: artifact.claim_id, entity: artifact.entity, status: artifact.status, confidence: artifact.confidence };
        return result;
      }

      case 'vault_query': {
        if (args.list_all) { const beliefs = this.vault.listBeliefs(); return { beliefs, total: beliefs.length }; }
        if (args.entity) { const r = this.vault.readBelief(args.entity); return r || { status: 'not_found', entity: args.entity }; }
        return this.vault.coverageIndex();
      }

      case 'vault_write_action':
        return this.vault.writeAction(args.title, args.content, args.action_type || 'report');

      case 'vault_search': {
        const beliefs = this.vault.listBeliefs();
        const q = (args.query || '').toLowerCase();
        const topK = args.top_k || 5;
        const matches = [];
        for (const b of beliefs) {
          const full = this.vault.readBelief(b.entity);
          if (full && (b.entity.toLowerCase().includes(q) || (full.body || '').toLowerCase().includes(q))) {
            matches.push({ entity: b.entity, confidence: b.confidence, status: b.status, score: b.entity.toLowerCase().includes(q) ? 3.0 : 1.0 });
          }
        }
        matches.sort((a, b) => b.score - a.score);
        return { query: args.query, results: matches.slice(0, topK), total: matches.length, engine: 'javascript' };
      }

      case 'compile_beliefs':
        return this.beliefCompiler.compileDirectory(this._projectDir(args.directory), args.max_files || 200);

      case 'verify_beliefs':
        return this.verifier.fullVerificationPass();

      // ── CogOps Analysis & Change Tools ──
      case 'blast_radius': {
        const files = (args.changed_files || '').split(',').map(s => s.trim()).filter(Boolean);
        return this.verifier.blastRadius(files);
      }

      case 'coverage_gaps':
        return this.verifier.coverageGaps(this._projectDir(args.directory));

      case 'process_change':
        return this.changePipe.processDiff(args.diff_text, args.commit_message || '', args.pr_title || '');

      case 'execute_flow':
        return this.flowOrchestrator.execute(args.query, args.diff_text || '', args.is_event || false, args.event_type || '');

      case 'refresh_beliefs': {
        const files = (args.changed_files || '').split(',').map(s => s.trim()).filter(Boolean);
        return this.changePipe.refreshDocs(files);
      }

      case 'compile_docs':
        return compileDocs(this.vault, this._projectDir(args.directory), args.max_files || 50);

      case 'export_training_data': {
        const format = args.format || 'jsonl';
        if (format !== 'jsonl') return { error: `Unsupported export format: ${format}` };
        return exportTrainingData(this.vault, this._projectOutput(args.output_path || 'training_data.jsonl'));
      }

      // ── Ingestion Tools ──
      case 'ingest_diff': {
        const modal = ingestDiff(args.diff_text, args.source, args.commit_message || '');
        const data = this.engine.ingest(modal.text, args.source, modal.token_estimate, false);
        return { ...data, modal_source_type: 'diff', intent: modal.metadata.intent, files_changed: modal.metadata.files_changed, added_lines: modal.metadata.added_lines, removed_lines: modal.metadata.removed_lines, symbols_changed: modal.metadata.symbols_changed };
      }

      case 'ingest_diagram':
        return ingestDiagram();

      case 'ingest_voice':
        return ingestVoice();

      // ── Workspace Tools ──
      case 'sync_workspace_changes': {
        const target = this._projectDir(args.directory);
        const listener = target === this.sourceDir ? this.workspaceListener : new WorkspaceChangeListener(this.vault, this.beliefCompiler, this.verifier, this.changePipe, target);
        return listener.scanOnce(args.force || false, args.max_files || 100);
      }

      case 'start_workspace_listener': {
        const target = this._projectDir(args.directory);
        const listener = target === this.sourceDir ? this.workspaceListener : new WorkspaceChangeListener(this.vault, this.beliefCompiler, this.verifier, this.changePipe, target);
        return listener.start(args.interval_s || 120, args.max_files || 100, args.force_initial || false);
      }

      // ── Skills Tools ──
      case 'create_skill': {
        const queries = (args.failing_queries || '').split('|').map(s => s.trim()).filter(Boolean);
        return this.skillEngine.createSkill(args.entity_key, queries, args.intent || '');
      }

      case 'manage_skills': {
        const action = args.action || 'list';
        if (action === 'list') { const skills = this.skillEngine.listSkills(); return { skills, total: skills.length, engine: 'javascript' }; }
        if (!args.skill_id) return { error: `skill_id required for '${action}'` };
        if (action === 'benchmark') return this.skillEngine.benchmarkSkill(args.skill_id);
        if (action === 'promote') return this.skillEngine.promoteOrPrune(args.skill_id);
        return { error: `Unknown action '${action}'. Use: list, benchmark, promote` };
      }

      // ── Analysis Tools ──
      case 'repo_file_map': {
        const grouped = buildRepoMap(this.sourceDir);
        if ((args.format || '').toLowerCase() === 'json') return grouped;
        return renderRepoMapMarkdown(grouped);
      }

      case 'prefetch_related': {
        // Use dep_graph_stats + import analysis for prediction
        const depStats = this.engine.dep_graph_stats();
        const imports = [];
        const content = args.source_content || '';
        const importRe = /(?:import|require|from|use|include)\s+['"]?([^\s'";\)]+)/g;
        let m;
        while ((m = importRe.exec(content)) !== null) imports.push(m[1]);
        return { file: args.file_path, language: args.language || 'unknown', predicted_files: imports.slice(0, 10), dep_graph: depStats, engine: 'javascript' };
      }

      // ── Hallucination Detection ──
      case 'verify_response': {
        const resp = args.response || '';
        const ctx = args.context || '';
        const prm = args.prompt || '';

        // Entity coverage gap: extract identifiers from context vs response
        const ctxEntities = new Set((ctx + ' ' + prm).match(/[a-zA-Z_][a-zA-Z0-9_.]{2,}/g) || []);
        const respEntities = (resp.match(/[a-zA-Z_][a-zA-Z0-9_.]{2,}/g) || []);
        const respUnique = [...new Set(respEntities)];
        let grounded = 0;
        const flaggedClaims = [];
        for (const ent of respUnique) {
          if (ctxEntities.has(ent) || ent.length <= 3) {
            grounded++;
          } else {
            flaggedClaims.push({ claim: ent, score: 1.0 });
          }
        }
        const entityGap = respUnique.length > 0 ? 1.0 - (grounded / respUnique.length) : 0;

        // ECE-like: hedging language curvature
        const hedgeWords = ['might', 'could', 'possibly', 'perhaps', 'maybe', 'uncertain', 'likely', 'probably', 'appears to', 'seems'];
        const wordCount = resp.split(/\s+/).length;
        let hedgeCount = 0;
        for (const h of hedgeWords) { hedgeCount += (resp.toLowerCase().match(new RegExp(h, 'g')) || []).length; }
        const hedgeCurvature = Math.min(1.0, hedgeCount / Math.max(wordCount, 1) * 20);

        // EPR-like: entropy consistency (char-bigram)
        const bigrams = {};
        for (let i = 0; i < resp.length - 1; i++) {
          const bg = resp.slice(i, i + 2);
          bigrams[bg] = (bigrams[bg] || 0) + 1;
        }
        const total = Object.values(bigrams).reduce((a, b) => a + b, 0) || 1;
        let entropy = 0;
        for (const c of Object.values(bigrams)) {
          const p = c / total;
          if (p > 0) entropy -= p * Math.log2(p);
        }
        // Normalize: high entropy is normal for text, low entropy is suspicious
        const maxEntropy = Math.log2(Math.min(Object.keys(bigrams).length, 500)) || 1;
        const eprRisk = Math.max(0, 1.0 - entropy / maxEntropy);

        // Fuse (same weights as proxy)
        const fused = Math.max(0, Math.min(1.0,
          0.80 * entityGap + 0.08 * hedgeCurvature + 0.07 * eprRisk + 0.05 * 0
        ));

        let verdict, recommendation;
        if (fused < 0.15) { verdict = 'pass'; recommendation = 'Accept — response appears well-grounded'; }
        else if (fused < 0.40) { verdict = 'warn'; recommendation = 'Review — some claims may not be grounded'; }
        else { verdict = 'flag'; recommendation = 'Reject or rephrase — high hallucination risk'; }

        return {
          fused_risk: Math.round(fused * 10000) / 10000,
          verdict,
          recommendation,
          witness: { entity_coverage_gap: Math.round(entityGap * 10000) / 10000, total_entities: respUnique.length, grounded_count: grounded },
          ece: { hedging_curvature: Math.round(hedgeCurvature * 10000) / 10000, hedge_count: hedgeCount },
          epr: { entropy_risk: Math.round(eprRisk * 10000) / 10000, bigram_entropy: Math.round(entropy * 100) / 100 },
          flagged_claims: flaggedClaims.slice(0, 10),
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  // ── JSON-RPC stdio transport ──
  async run() {
    this._log(`Starting Entroly MCP server v${this.config.serverVersion} (Wasm engine)`);

    // Auto-index on startup
    try {
      const result = autoIndex(this.engine);
      if (result.status === 'indexed') {
        this._log(`Auto-indexed ${result.files_indexed} files (${result.total_tokens.toLocaleString()} tokens) in ${result.duration_s}s`);
      }
      startIncrementalWatcher(this.engine);
    } catch (e) {
      this._log(`Auto-index failed (non-fatal): ${e.message}`);
    }

    // Start autotune daemon — reads tuning_config.json, hot-reloads weights
    try {
      const tid = startAutotuneDaemon(this.engine);
      if (tid) this._log('Autotune daemon started (hot-reload every 30s)');
    } catch (e) {
      this._log(`Autotune: failed to start daemon: ${e.message}`);
    }

    // Graceful shutdown
    process.on('SIGTERM', () => {
      this._log('Shutdown — persisting state...');
      try { persistIndex(this.engine, this.indexPath); } catch {}
      process.exit(0);
    });

    // Read JSONRPC from stdin
    process.stdin.setEncoding('utf-8');
    process.stdin.on('data', (chunk) => {
      this._buffer += chunk;
      this._processBuffer();
    });
    process.stdin.on('end', () => {
      this._log('stdin closed — shutting down');
      try { persistIndex(this.engine, this.indexPath); } catch {}
    });
  }

  _processBuffer() {
    while (true) {
      const headerEnd = this._buffer.indexOf('\r\n\r\n');

      if (headerEnd !== -1) {
        const header = this._buffer.slice(0, headerEnd);
        const match = header.match(/Content-Length:\s*(\d+)/i);
        if (match) {
          const contentLength = parseInt(match[1], 10);
          const bodyStart = headerEnd + 4;
          if (this._buffer.length < bodyStart + contentLength) return;
          const body = this._buffer.slice(bodyStart, bodyStart + contentLength);
          this._buffer = this._buffer.slice(bodyStart + contentLength);
          try {
            this._handleMessage(JSON.parse(body));
          } catch (e) {
            this._log(`JSON parse error (LSP frame): ${e.message}`);
          }
          continue;
        }
      }

      const nlIdx = this._buffer.indexOf('\n');
      if (nlIdx === -1) return;
      const line = this._buffer.slice(0, nlIdx).trim();
      this._buffer = this._buffer.slice(nlIdx + 1);
      if (!line) continue;
      try {
        this._handleMessage(JSON.parse(line));
      } catch (e) {
        this._log(`JSON parse error (NDJSON): ${e.message}`);
      }
    }
  }

  _handleMessage(msg) {
    if (msg.method === 'initialize') {
      this._initialized = true;
      this._respond(msg.id, {
        protocolVersion: '2024-11-05',
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: this.config.serverName, version: this.config.serverVersion },
      });
    } else if (msg.method === 'notifications/initialized') {
      // No response needed for notifications
    } else if (msg.method === 'tools/list') {
      this._respond(msg.id, { tools: this.tools });
    } else if (msg.method === 'tools/call') {
      const { name, arguments: toolArgs } = msg.params;
      try {
        const result = this.handleTool(name, toolArgs || {});
        const text = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
        this._respond(msg.id, {
          content: [{ type: 'text', text }], isError: false,
        });
      } catch (e) {
        this._respond(msg.id, {
          content: [{ type: 'text', text: JSON.stringify({ error: e.message }) }], isError: true,
        });
      }
    } else if (msg.method === 'ping') {
      this._respond(msg.id, {});
    } else if (msg.id !== undefined) {
      // Unknown method with ID — respond with error
      this._respondError(msg.id, -32601, `Method not found: ${msg.method}`);
    }
  }

  _respond(id, result) {
    if (id === undefined) return;
    const response = JSON.stringify({ jsonrpc: '2.0', id, result });
    const header = `Content-Length: ${Buffer.byteLength(response)}\r\n\r\n`;
    process.stdout.write(header + response);
  }

  _respondError(id, code, message) {
    const response = JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } });
    const header = `Content-Length: ${Buffer.byteLength(response)}\r\n\r\n`;
    process.stdout.write(header + response);
  }
}

// ── Entry Point ──
if (require.main === module) {
  const server = new EntrolyMCPServer();
  server.run();
}

module.exports = { EntrolyMCPServer, resolveProjectDirectory, resolveProjectOutput };
