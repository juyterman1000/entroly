/**
 * CogOps Engine — pure-JS port of entroly's epistemic layer
 *
 * Ports: epistemic_router.py, belief_compiler.py, verification_engine.py,
 *        change_pipeline.py, flow_orchestrator.py, evolution_logger.py
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');
const { VaultManager, parseFrontmatter, extractSources, extractBody, safeFilename, nowISO, SOURCE_EXTS } = require('./vault');

// ── Intent Classification ───────────────────────────────────────────

const INTENTS = {
  INCIDENT: 'incident', AUDIT: 'audit', REPAIR: 'repair', TEST_GAP: 'test_gap',
  RELEASE: 'release', PR_BRIEF: 'pr_brief', CODE_GENERATION: 'code_generation',
  REPORT: 'report', RESEARCH: 'research', REUSE: 'reuse',
  ARCHITECTURE: 'architecture', ONBOARDING: 'onboarding', GENERAL: 'general',
};

const FLOWS = {
  FAST_ANSWER: 'fast_answer', VERIFY_BEFORE: 'verify_before',
  COMPILE_ON_DEMAND: 'compile_on_demand', CHANGE_DRIVEN: 'change_driven',
  SELF_IMPROVEMENT: 'self_improvement',
};

const INTENT_PATTERNS = [
  [INTENTS.INCIDENT, /\b(incident|outage|spike|latency|down|crash|alert|page|5\d\d|timeout|spiking|degraded|error.rate|p99|on.?call)\b/i],
  [INTENTS.AUDIT, /\b(audit|compliance|pii|gdpr|hipaa|security.review|vulnerability|cve|leak|injection|xss|csrf|secrets?)\b/i],
  [INTENTS.REPAIR, /\b(fail|broke|broken|regression|bisect|root.?cause|fix|debug|stack.?trace|exception|error|traceback|started.failing)\b/i],
  [INTENTS.TEST_GAP, /\b(test.gap|missing.test|uncovered|coverage|untested|edge.case|what.tests)\b/i],
  [INTENTS.RELEASE, /\b(release.?ready|ship|deploy|rollout|canary|staging|production.ready|go.?live|launch)\b/i],
  [INTENTS.PR_BRIEF, /\b(pr|pull.request|diff|code.review|review.this|patch|changeset|commit|merge)\b/i],
  [INTENTS.CODE_GENERATION, /\b(generate|implement|create|write|scaffold|migration|boilerplate|add.?feature|build.?a|code.?for)\b/i],
  [INTENTS.REPORT, /\b(report|diagram|slide|presentation|chart|graph|visuali|mermaid|marp|draw|render)\b/i],
  [INTENTS.RESEARCH, /\b(research|benchmark|compare|evaluate|survey|trade.?off|pros.?cons|analysis|investigate)\b/i],
  [INTENTS.REUSE, /\b(already.have|existing|reuse|duplicate|reinvent|shared.?util|helper|library|common)\b/i],
  [INTENTS.ARCHITECTURE, /\b(architect|design|structure|module|service|component|layer|talk.?to|dependency|flow|pipeline|system|how.does.+work|connect|interact|communicat)\b/i],
  [INTENTS.ONBOARDING, /\b(onboard|overview|walkthrough|tutorial|new.?engineer|explain.+to|introduction|getting.started|for.?beginners)\b/i],
];

const HIGH_RISK = /\b(security|compliance|pii|gdpr|hipaa|credential|secret|key|token|password|auth|encrypt|permission|rbac|acl|cve|vulnerability|injection|deletion|drop.table|rm.\-rf|production|customer.data)\b/i;
const MED_RISK = /\b(migration|schema.change|breaking|backward|deprecat|refactor|database|payment|billing|transaction|rollback|deploy)\b/i;

const STOP_WORDS = new Set([
  'how', 'does', 'the', 'what', 'is', 'explain', 'show', 'can', 'you', 'please',
  'work', 'works', 'about', 'for', 'and', 'this', 'that', 'with', 'from', 'have',
  'are', 'module', 'architecture', 'design', 'pipeline', 'system', 'component', 'service',
]);

function classifyIntent(query) {
  for (const [intent, re] of INTENT_PATTERNS) { if (re.test(query)) return intent; }
  return INTENTS.GENERAL;
}

function assessRisk(query, intent) {
  if (intent === INTENTS.AUDIT) return 'high';
  if (HIGH_RISK.test(query)) return 'high';
  if (MED_RISK.test(query)) return 'medium';
  return 'low';
}

function extractEntityKey(query) {
  const compounds = query.match(/[a-zA-Z][a-zA-Z0-9]*(?:[_.:][a-zA-Z][a-zA-Z0-9]*)+/g);
  if (compounds) return compounds.sort((a, b) => b.length - a.length)[0].toLowerCase().replace(/::/g, '_').replace(/\./g, '_');
  const tokens = (query.match(/[a-zA-Z_]\w+/g) || []).filter(w => w.length > 3 && !STOP_WORDS.has(w.toLowerCase()));
  if (tokens.length) return tokens.sort((a, b) => b.length - a.length)[0].toLowerCase();
  return 'unknown';
}

// ── Epistemic Router ────────────────────────────────────────────────

class EpistemicRouter {
  constructor(vault, opts = {}) {
    this._vault = vault;
    this._missThreshold = opts.missThreshold || 3;
    this._freshnessHours = opts.freshnessHours || 24;
    this._minConfidence = opts.minConfidence || 0.6;
    this._missCounts = {};
    this._history = [];
  }

  route(query, isEvent = false, eventType = '') {
    const intent = isEvent ? this._classifyEventIntent(eventType) : classifyIntent(query);
    const coverage = this._checkCoverage(query);
    const risk = assessRisk(query, intent);
    const entityKey = extractEntityKey(query);
    const missCount = this._missCounts[entityKey] || 0;
    const [flow, reasoning] = this._selectFlow(intent, coverage, risk, isEvent, missCount);
    if (!coverage.exists) this._missCounts[entityKey] = missCount + 1;
    else this._missCounts[entityKey] = 0;
    const decision = { routing_id: randomUUID().slice(0, 12), flow, intent, risk, reasoning, coverage: { exists: coverage.exists, fresh: coverage.fresh, verified: coverage.verified, confidence: coverage.confidence, matching_claims: coverage.matching_claims }, timestamp: Date.now() / 1000 };
    this._history.push(decision);
    return decision;
  }

  stats() {
    const flowCounts = {}, intentCounts = {};
    for (const d of this._history) { flowCounts[d.flow] = (flowCounts[d.flow] || 0) + 1; intentCounts[d.intent] = (intentCounts[d.intent] || 0) + 1; }
    return { total_routed: this._history.length, flow_distribution: flowCounts, intent_distribution: intentCounts, active_miss_entities: Object.fromEntries(Object.entries(this._missCounts).filter(([, v]) => v > 0)), evolution_triggers: this._history.filter(d => d.flow === FLOWS.SELF_IMPROVEMENT).length };
  }

  _selectFlow(intent, coverage, risk, isEvent, missCount) {
    if (isEvent) return [FLOWS.CHANGE_DRIVEN, 'Event-triggered: route through Truth → Belief → Verification → Action'];
    if (missCount >= this._missThreshold) return [FLOWS.SELF_IMPROVEMENT, `Repeated miss (count=${missCount}) on this entity: logging for Evolution skill synthesis`];
    if (!coverage.exists) return [FLOWS.COMPILE_ON_DEMAND, 'No relevant beliefs found: compile from Truth first'];
    if (!coverage.fresh || coverage.confidence < this._minConfidence) return [FLOWS.VERIFY_BEFORE, `Beliefs exist but stale/low-confidence (fresh=${coverage.fresh}, conf=${coverage.confidence.toFixed(2)}): verify before answering`];
    if (!coverage.verified) return [FLOWS.VERIFY_BEFORE, 'Beliefs exist and are fresh but unverified: verify first'];
    if (risk === 'high') return [FLOWS.VERIFY_BEFORE, 'High-risk domain: re-verify before answering'];
    if (intent === INTENTS.CODE_GENERATION) return [FLOWS.VERIFY_BEFORE, 'Code generation: always verify before shipping'];
    return [FLOWS.FAST_ANSWER, `Beliefs are fresh, verified, and confident (conf=${coverage.confidence.toFixed(2)}): fast answer`];
  }

  _checkCoverage(query) {
    const empty = { exists: false, fresh: false, verified: false, confidence: 0, matching_claims: [] };
    const beliefs = this._vault.listBeliefs();
    const terms = new Set((query.match(/[a-zA-Z_]\w+/g) || []).filter(w => w.length > 2).map(w => w.toLowerCase()));
    if (!terms.size) return empty;
    const matching = []; let minConf = 1, allFresh = true, allVerified = true;
    for (const b of beliefs) {
      const entity = (b.entity || '').toLowerCase();
      const stem = path.basename(b.file || '', '.md').toLowerCase();
      let matched = false;
      for (const t of terms) { if (entity.includes(t) || stem.includes(t)) { matched = true; break; } }
      if (!matched) continue;
      matching.push(b.entity || stem);
      minConf = Math.min(minConf, b.confidence || 0);
      if (b.status === 'stale') allFresh = false;
      if (b.status !== 'verified') allVerified = false;
      if (b.last_checked) {
        try { const h = (Date.now() - new Date(b.last_checked).getTime()) / 3600000; if (h > this._freshnessHours) allFresh = false; } catch {}
      }
    }
    if (!matching.length) return empty;
    return { exists: true, fresh: allFresh, verified: allVerified, confidence: minConf <= 1 ? minConf : 0, matching_claims: matching };
  }

  _classifyEventIntent(et) {
    const t = et.toLowerCase();
    if (/pr|pull|merge/.test(t)) return INTENTS.PR_BRIEF;
    if (/incident|alert/.test(t)) return INTENTS.INCIDENT;
    if (/release|deploy/.test(t)) return INTENTS.RELEASE;
    if (/schedule|nightly|cron/.test(t)) return INTENTS.AUDIT;
    return INTENTS.GENERAL;
  }
}

// ── Belief Compiler ─────────────────────────────────────────────────

const ENTITY_PATTERNS = {
  py: [
    [/^class\s+(\w+)/gm, 'class'],
    [/^def\s+(\w+)/gm, 'function'],
    [/^(\w+)\s*=/gm, 'variable'],
  ],
  rs: [
    [/^pub\s+(?:fn|struct|enum|trait|type|const)\s+(\w+)/gm, 'rust_item'],
    [/^fn\s+(\w+)/gm, 'function'],
    [/^(?:struct|enum|trait)\s+(\w+)/gm, 'type'],
  ],
  ts: [
    [/^(?:export\s+)?(?:class|interface|type|enum)\s+(\w+)/gm, 'type'],
    [/^(?:export\s+)?(?:function|const|let)\s+(\w+)/gm, 'function'],
  ],
  js: [
    [/^(?:class|function)\s+(\w+)/gm, 'type'],
    [/^(?:const|let|var)\s+(\w+)/gm, 'variable'],
    [/^module\.exports\s*=\s*\{\s*([^}]+)\}/gm, 'exports'],
  ],
};
ENTITY_PATTERNS.tsx = ENTITY_PATTERNS.ts;
ENTITY_PATTERNS.jsx = ENTITY_PATTERNS.js;
ENTITY_PATTERNS.go = [[/^(?:func|type|var|const)\s+(\w+)/gm, 'go_item']];
ENTITY_PATTERNS.java = [[/^(?:public|private|protected)?\s*(?:class|interface|enum)\s+(\w+)/gm, 'type'], [/(?:public|private|protected)\s+\w+\s+(\w+)\s*\(/gm, 'method']];

function extractEntities(content, ext) {
  const patterns = ENTITY_PATTERNS[ext] || ENTITY_PATTERNS.js;
  const entities = [];
  for (const [re, kind] of patterns) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(content)) !== null) {
      const name = m[1];
      if (name && name.length > 1 && !name.startsWith('_')) entities.push({ name, kind });
    }
  }
  return entities;
}

class BeliefCompiler {
  constructor(vault) { this._vault = vault; }

  compileDirectory(directory, maxFiles = 200) {
    const sourceFiles = this._findSourceFiles(directory, maxFiles);
    let beliefsWritten = 0, entitiesExtracted = 0;
    const errors = [];
    for (const fp of sourceFiles) {
      try {
        const content = fs.readFileSync(fp, 'utf-8');
        const ext = path.extname(fp).slice(1);
        const entities = extractEntities(content, ext);
        const relPath = path.relative(directory, fp).replace(/\\/g, '/');
        for (const ent of entities) {
          const entity = `${relPath}::${ent.name}`;
          this._vault.writeBelief({
            claim_id: randomUUID(),
            entity,
            status: 'inferred',
            confidence: 0.6,
            sources: [`${relPath}`],
            last_checked: nowISO(),
            derived_from: ['belief_compiler'],
            title: `${ent.kind}: ${ent.name}`,
            body: `Extracted ${ent.kind} \`${ent.name}\` from \`${relPath}\`.\n\nAuto-compiled by entroly belief compiler.`,
          });
          beliefsWritten++;
          entitiesExtracted++;
        }
      } catch (e) { errors.push(`${fp}: ${e.message}`); }
    }
    return { status: 'compiled', files_processed: sourceFiles.length, beliefs_written: beliefsWritten, entities_extracted: entitiesExtracted, errors: errors.slice(0, 10), engine: 'javascript' };
  }

  _findSourceFiles(dir, max) {
    const results = [];
    const walk = (d) => {
      if (results.length >= max) return;
      let entries;
      try { entries = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
      for (const e of entries) {
        if (results.length >= max) break;
        if (e.name.startsWith('.') || e.name === 'node_modules' || e.name === '__pycache__' || e.name === 'target' || e.name === '.git') continue;
        const full = path.join(d, e.name);
        if (e.isDirectory()) walk(full);
        else if (SOURCE_EXTS.has(path.extname(e.name))) results.push(full);
      }
    };
    walk(dir);
    return results;
  }
}

// ── Verification Engine ─────────────────────────────────────────────

class VerificationEngine {
  constructor(vault, opts = {}) {
    this._vault = vault;
    this._freshnessHours = opts.freshnessHours || 24;
    this._minConfidence = opts.minConfidence || 0.5;
  }

  fullVerificationPass() {
    const beliefs = this._vault.listBeliefs();
    const stale = [], lowConf = [], contradictions = [];
    const now = Date.now();
    for (const b of beliefs) {
      if (b.status === 'stale') stale.push(b.entity);
      else if (b.last_checked) {
        try { const h = (now - new Date(b.last_checked).getTime()) / 3600000; if (h > this._freshnessHours) stale.push(b.entity); } catch {}
      }
      if (b.confidence < this._minConfidence) lowConf.push(b.entity);
    }
    // Simple contradiction detection: same base entity, divergent confidence
    const byBase = {};
    for (const b of beliefs) {
      const base = (b.entity || '').split('::')[0];
      if (!byBase[base]) byBase[base] = [];
      byBase[base].push(b);
    }
    for (const [base, group] of Object.entries(byBase)) {
      if (group.length < 2) continue;
      const confs = group.map(g => g.confidence);
      if (Math.max(...confs) - Math.min(...confs) > 0.4) contradictions.push({ entity: base, spread: (Math.max(...confs) - Math.min(...confs)).toFixed(2) });
    }
    const report = { total_beliefs: beliefs.length, stale_count: stale.length, low_confidence_count: lowConf.length, contradiction_count: contradictions.length, stale_entities: stale.slice(0, 20), low_confidence_entities: lowConf.slice(0, 20), contradictions: contradictions.slice(0, 10), engine: 'javascript' };
    // Write verification artifact
    this._vault.writeVerification({ challenges: 'full_pass', result: stale.length + lowConf.length + contradictions.length > 0 ? 'issues_found' : 'confirmed', confidence_delta: 0, checked_at: nowISO(), method: 'full_verification_pass', title: 'Full Verification Pass', body: `Found ${stale.length} stale, ${lowConf.length} low-confidence, ${contradictions.length} contradictions across ${beliefs.length} beliefs.` });
    return report;
  }

  blastRadius(changedFiles) {
    const beliefs = this._vault.listBeliefs();
    const changedStems = new Set(changedFiles.map(f => path.basename(f).replace(/\.[^.]+$/, '').toLowerCase()));
    const changedPaths = new Set(changedFiles.map(f => f.replace(/\\/g, '/').toLowerCase()));
    const affected = [], affectedEntities = [];
    for (const b of beliefs) {
      const entity = (b.entity || '').toLowerCase();
      let hit = [...changedStems].some(s => entity.includes(s));
      if (!hit) {
        // Read full belief to check sources
        const full = this._vault.readBelief(b.entity);
        if (full) {
          const srcs = extractSources(`---\n${Object.entries(full.frontmatter).map(([k, v]) => `${k}: ${v}`).join('\n')}\nsources:\n${(full.frontmatter.sources || '').split(',').map(s => `  - ${s.trim()}`).join('\n')}\n---`);
          hit = srcs.some(s => { const sp = s.split(':')[0].replace(/\\/g, '/').toLowerCase(); return changedPaths.has(sp) || changedStems.has(path.basename(sp).replace(/\.[^.]+$/, '')); });
        }
      }
      if (hit) { affected.push(b); affectedEntities.push(b.entity); }
    }
    const risk = affected.length > 10 ? 'high' : affected.length > 3 ? 'medium' : 'low';
    return { affected_beliefs: affected.length, affected_entities: affectedEntities.slice(0, 30), risk_level: risk, description: `${affected.length} beliefs affected by changes to ${changedFiles.length} file(s)`, engine: 'javascript' };
  }

  coverageGaps(directory) {
    const sourceFiles = new BeliefCompiler(this._vault)._findSourceFiles(directory, 500);
    const beliefs = this._vault.listBeliefs();
    const coveredStems = new Set(beliefs.map(b => {
      const e = (b.entity || '').toLowerCase();
      const parts = e.split('::');
      return parts[0].replace(/\.[^.]+$/, '');
    }));
    const gaps = [];
    for (const fp of sourceFiles) {
      const rel = path.relative(directory, fp).replace(/\\/g, '/');
      const stem = rel.replace(/\.[^.]+$/, '').toLowerCase();
      if (!coveredStems.has(stem)) gaps.push({ file: rel, reason: 'no_belief', suggested_entity: `${rel}::main` });
    }
    return { gaps: gaps.slice(0, 50), total_gaps: gaps.length, engine: 'javascript' };
  }
}

// ── Change Pipeline ─────────────────────────────────────────────────

function parseDiff(diffText) {
  const files = new Set(), added = [], removed = [];
  let linesAdded = 0, linesRemoved = 0;
  for (const line of diffText.split('\n')) {
    if (line.startsWith('diff --git') || line.startsWith('---') || line.startsWith('+++')) {
      const m = line.match(/[ab]\/(.+)/);
      if (m) files.add(m[1]);
    } else if (line.startsWith('+') && !line.startsWith('+++')) { linesAdded++; added.push(line.slice(1)); }
    else if (line.startsWith('-') && !line.startsWith('---')) { linesRemoved++; removed.push(line.slice(1)); }
  }
  return { files_modified: [...files], lines_added: linesAdded, lines_removed: linesRemoved, added_content: added.join('\n'), removed_content: removed.join('\n') };
}

function classifyDiffIntent(diffText, commitMsg) {
  const text = (commitMsg + ' ' + diffText).toLowerCase();
  if (/fix|bug|patch|hotfix|resolve/.test(text)) return 'bug-fix';
  if (/test|spec|assert/.test(text)) return 'test';
  if (/security|cve|vuln|auth/.test(text)) return 'security';
  if (/perf|optimi|speed|cache|fast/.test(text)) return 'performance';
  if (/refactor|rename|move|clean/.test(text)) return 'refactor';
  return 'feature';
}

function reviewDiff(addedContent) {
  const findings = [];
  const checks = [
    [/(?:password|secret|api.?key|token)\s*[=:]\s*['"][^'"]{8,}/gi, 'Possible hardcoded secret', 'critical'],
    [/TODO|FIXME|HACK|XXX/gi, 'TODO/FIXME marker', 'info'],
    [/except\s*:|catch\s*\(/gi, 'Broad exception handler', 'medium'],
    [/eval\s*\(|exec\s*\(/gi, 'Dynamic code execution', 'high'],
    [/rm\s+-rf|DROP\s+TABLE|DELETE\s+FROM/gi, 'Destructive operation', 'high'],
  ];
  for (const [re, msg, sev] of checks) {
    const matches = addedContent.match(re);
    if (matches) findings.push({ message: msg, severity: sev, count: matches.length });
  }
  return findings;
}

class ChangePipeline {
  constructor(vault, verifier) { this._vault = vault; this._verifier = verifier; }

  processDiff(diffText, commitMessage = '', prTitle = '') {
    const parsed = parseDiff(diffText);
    const intent = classifyDiffIntent(diffText, commitMessage);
    const findings = reviewDiff(parsed.added_content);
    const blast = this._verifier.blastRadius(parsed.files_modified);
    const summary = `${intent} change: ${parsed.files_modified.length} files, +${parsed.lines_added}/-${parsed.lines_removed} lines`;
    // Write action
    this._vault.writeAction(prTitle || summary, `## Summary\n${summary}\n\n## Intent\n${intent}\n\n## Files\n${parsed.files_modified.map(f => `- ${f}`).join('\n')}\n\n## Review Findings\n${findings.length ? findings.map(f => `- [${f.severity}] ${f.message} (${f.count}x)`).join('\n') : 'No issues found'}`, 'pr_brief');
    return { title: prTitle || summary, summary, risk_level: blast.risk_level, intent, files_modified: parsed.files_modified, lines_added: parsed.lines_added, lines_removed: parsed.lines_removed, findings_count: findings.length, findings, blast_radius: blast, engine: 'javascript' };
  }

  refreshDocs(changedFiles) {
    const result = this._vault.markBeliefsStaleForFiles(changedFiles);
    result.engine = 'javascript';
    return result;
  }
}

// ── Flow Orchestrator ───────────────────────────────────────────────

class FlowOrchestrator {
  constructor(vault, router, compiler, verifier, changePipe, sourceDir) {
    this._vault = vault;
    this._router = router;
    this._compiler = compiler;
    this._verifier = verifier;
    this._changePipe = changePipe;
    this._sourceDir = sourceDir;
  }

  execute(query, diffText = '', isEvent = false, eventType = '') {
    const decision = this._router.route(query, isEvent, eventType);
    const steps = [];
    const result = { flow: decision.flow, intent: decision.intent, risk: decision.risk, reasoning: decision.reasoning, steps, engine: 'javascript' };

    switch (decision.flow) {
      case FLOWS.FAST_ANSWER:
        steps.push({ step: 'belief_lookup', status: 'done' });
        result.beliefs = this._vault.coverageIndex();
        break;

      case FLOWS.VERIFY_BEFORE:
        steps.push({ step: 'belief_lookup', status: 'done' });
        const vr = this._verifier.fullVerificationPass();
        steps.push({ step: 'verification', status: 'done', findings: vr.stale_count + vr.low_confidence_count + vr.contradiction_count });
        result.verification = vr;
        break;

      case FLOWS.COMPILE_ON_DEMAND:
        const cr = this._compiler.compileDirectory(this._sourceDir);
        steps.push({ step: 'compile', status: 'done', beliefs_written: cr.beliefs_written });
        const vr2 = this._verifier.fullVerificationPass();
        steps.push({ step: 'verification', status: 'done' });
        result.compilation = cr;
        result.verification = vr2;
        break;

      case FLOWS.CHANGE_DRIVEN:
        if (diffText) {
          const pr = this._changePipe.processDiff(diffText);
          steps.push({ step: 'change_analysis', status: 'done', intent: pr.intent });
          result.change_brief = pr;
        }
        const cr2 = this._compiler.compileDirectory(this._sourceDir);
        steps.push({ step: 'recompile', status: 'done', beliefs_written: cr2.beliefs_written });
        result.compilation = cr2;
        break;

      case FLOWS.SELF_IMPROVEMENT:
        steps.push({ step: 'miss_logged', status: 'done' });
        result.message = 'Repeated misses logged for Evolution skill synthesis. Use create_skill to address the gap.';
        break;
    }

    return result;
  }
}

// ── Compile Docs ────────────────────────────────────────────────────

function compileDocs(vault, directory, maxFiles = 50) {
  const root = directory;
  const docPatterns = ['README', 'ARCHITECTURE', 'CONTRIBUTING', 'CHANGELOG', 'DESIGN', 'ADR'];
  let compiled = 0;
  const entities = [];
  const walk = (dir, depth = 0) => {
    if (depth > 2 || compiled >= maxFiles) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      if (compiled >= maxFiles) break;
      const full = path.join(dir, e.name);
      if (e.isDirectory() && (e.name === 'docs' || e.name === 'doc') && depth < 2) { walk(full, depth + 1); continue; }
      if (!e.isFile() || !e.name.endsWith('.md')) continue;
      const stem = e.name.replace(/\.md$/i, '').toUpperCase();
      if (!docPatterns.some(p => stem.startsWith(p)) && dir === root) continue;
      try {
        const content = fs.readFileSync(full, 'utf-8');
        const entity = `doc/${e.name.replace(/\.md$/i, '').toLowerCase()}`;
        vault.writeBelief({
          claim_id: randomUUID(),
          entity,
          status: 'inferred',
          confidence: 0.8,
          sources: [path.relative(root, full).replace(/\\/g, '/')],
          last_checked: nowISO(),
          derived_from: ['compile_docs'],
          title: `Documentation: ${e.name}`,
          body: content.slice(0, 3000),
        });
        entities.push(entity);
        compiled++;
      } catch {}
    }
  };
  walk(root);
  return { status: 'compiled', docs_found: compiled, docs_compiled: compiled, entities, engine: 'javascript' };
}

// ── Export Training Data ────────────────────────────────────────────

function exportTrainingData(vault, outputPath = 'training_data.jsonl') {
  const beliefs = vault.listBeliefs();
  const lines = [];
  let skipped = 0;
  for (const b of beliefs) {
    if (b.confidence < 0.5 || b.status === 'stale') { skipped++; continue; }
    const full = vault.readBelief(b.entity);
    if (!full) continue;
    const body = full.body.slice(0, 2000);
    lines.push(JSON.stringify({ messages: [
      { role: 'system', content: `You are an expert on the ${b.entity} codebase.` },
      { role: 'user', content: `What does ${b.entity} do?` },
      { role: 'assistant', content: body },
    ] }));
  }
  fs.writeFileSync(outputPath, lines.join('\n'), 'utf-8');
  return { status: 'exported', output_path: outputPath, format: 'jsonl', beliefs_used: lines.length, beliefs_skipped: skipped, training_pairs: lines.length, engine: 'javascript' };
}

module.exports = {
  EpistemicRouter, BeliefCompiler, VerificationEngine, ChangePipeline, FlowOrchestrator,
  classifyIntent, assessRisk, extractEntityKey, parseDiff, classifyDiffIntent, reviewDiff,
  compileDocs, exportTrainingData,
  INTENTS, FLOWS,
};
