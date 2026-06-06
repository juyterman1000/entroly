const crypto = require('crypto');

const SCHEMA_VERSION = 'context-receipt.v1';

function sha256(value) {
  return crypto.createHash('sha256').update(String(value).replace(/\r\n/g, '\n').replace(/\r/g, '\n')).digest('hex');
}

function fingerprint(text) {
  return `sha256:${sha256(text)}`;
}

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === 'object') {
    return Object.keys(value).sort().reduce((out, key) => {
      out[key] = stableValue(value[key]);
      return out;
    }, {});
  }
  return value;
}

function stableHash(value) {
  return sha256(JSON.stringify(stableValue(value)));
}

function tokenize(text) {
  return String(text)
    .replace(/-/g, ' ')
    .toLowerCase()
    .match(/[a-z0-9][a-z0-9_']*/g) || [];
}

function estimateTokens(text) {
  return Math.max(1, tokenize(text).length);
}

function titleForPath(sourcePath) {
  const name = String(sourcePath).split(/[\\/]/).pop() || sourcePath;
  return name.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ');
}

function normalizeDocs(documents) {
  if (!documents) return [];
  if (!Array.isArray(documents) && typeof documents === 'object') {
    return Object.entries(documents).map(([sourcePath, text]) => [String(sourcePath), String(text)]);
  }
  return documents.map((item, idx) => {
    if (Array.isArray(item)) return [String(item[0]), String(item[1] || '')];
    if (item && typeof item === 'object') {
      return [
        String(item.source_path || item.source || item.path || `document_${idx}.txt`),
        String(Object.prototype.hasOwnProperty.call(item, 'text') ? item.text : item.content || ''),
      ];
    }
    return [`document_${idx}.txt`, String(item || '')];
  });
}

function isHeading(line) {
  const trimmed = line.trim();
  return /^#{1,6}\s+/.test(trimmed)
    || /^(section|article|clause|exhibit|schedule|addendum)\s+[A-Za-z0-9.-]+/i.test(trimmed)
    || /^\d+(\.\d+)*\s+\S+/.test(trimmed);
}

function paragraphBlocks(text) {
  const blocks = [];
  let offset = 0;
  let start = null;
  let current = [];
  let heading = null;
  let page = null;
  const flush = (end) => {
    if (start !== null && current.length) {
      const raw = current.join('').trim();
      if (raw) blocks.push({ text: raw, start, end, heading, page });
    }
    start = null;
    current = [];
  };
  for (const line of String(text).match(/[^\n]*\n|[^\n]+$/g) || []) {
    const trimmed = line.trim();
    const pageMatch = trimmed.match(/\bpage\s+(\d+)\b/i);
    if (pageMatch) page = Number(pageMatch[1]);
    if (trimmed && isHeading(trimmed)) {
      flush(offset);
      heading = trimmed.replace(/^#+\s*/, '');
    }
    if (!trimmed) {
      flush(offset);
    } else {
      if (start === null) start = offset;
      current.push(line);
    }
    offset += line.length;
  }
  flush(String(text).length);
  return blocks;
}

function chunkDocument(sourcePath, text, documentId, docFingerprint, chunkTokens = 360) {
  const raw = [];
  let pending = [];
  let start = null;
  let end = 0;
  let heading = null;
  let page = null;
  let tokens = 0;
  const flush = () => {
    if (start !== null && pending.length) {
      raw.push({ text: pending.join('\n\n').trim(), start, end, heading, page });
    }
    pending = [];
    start = null;
    tokens = 0;
  };
  for (const block of paragraphBlocks(text)) {
    const count = estimateTokens(block.text);
    if (pending.length && tokens + count > chunkTokens) flush();
    if (start === null) {
      start = block.start;
      heading = block.heading;
      page = block.page;
    }
    pending.push(block.text);
    end = block.end;
    tokens += count;
  }
  flush();

  let running = 0;
  return raw.map((chunk, idx) => {
    const count = estimateTokens(chunk.text);
    const fp = fingerprint(`${docFingerprint}\n${chunk.start}:${chunk.end}\n${chunk.text}`);
    const chunkId = `chk_${stableHash({ doc: documentId, start: chunk.start, end: chunk.end, fp }).slice(0, 12)}`;
    const out = {
      chunk_id: chunkId,
      document_id: documentId,
      source_path: sourcePath,
      title: titleForPath(sourcePath),
      section_heading: chunk.heading,
      page_number: chunk.page,
      chunk_index: idx,
      byte_start: chunk.start,
      byte_end: chunk.end,
      token_start: running,
      token_end: running + count,
      token_count: count,
      fingerprint: fp,
      text: chunk.text,
    };
    running += count;
    return out;
  });
}

function ingestReceiptDocuments(documents, options = {}) {
  const chunkTokens = Math.max(40, Number(options.chunkTokens || options.chunk_tokens || 360));
  const normalized = normalizeDocs(documents).sort((a, b) => a[0].localeCompare(b[0]));
  const records = [];
  const chunks = [];
  const sourceFingerprints = {};
  for (const [sourcePath, text] of normalized) {
    const docFp = fingerprint(text);
    const documentId = `doc_${stableHash({ sourcePath, docFp }).slice(0, 12)}`;
    const docChunks = chunkDocument(sourcePath, text, documentId, docFp, chunkTokens);
    sourceFingerprints[sourcePath] = docFp;
    records.push({
      document_id: documentId,
      source_path: sourcePath,
      title: titleForPath(sourcePath),
      fingerprint: docFp,
      token_count: estimateTokens(text),
      byte_count: Buffer.byteLength(text),
      chunk_ids: docChunks.map((c) => c.chunk_id),
    });
    chunks.push(...docChunks);
  }
  return {
    schema_version: SCHEMA_VERSION,
    documents: records,
    chunks,
    chunk_token_limit: chunkTokens,
    chunk_overlap: Number(options.overlapTokens || options.overlap_tokens || 32),
    source_fingerprints: sourceFingerprints,
  };
}

function rankChunks(index, query) {
  const queryTerms = Array.from(new Set(tokenize(query).filter((t) => t.length > 1)));
  const docs = index.chunks.map((chunk) => ({ chunk, terms: tokenize(chunk.text) }));
  const df = {};
  for (const doc of docs) {
    for (const term of new Set(doc.terms)) df[term] = (df[term] || 0) + 1;
  }
  const docCount = Math.max(1, docs.length);
  const avgLen = docs.reduce((sum, doc) => sum + doc.terms.length, 0) / docCount || 1;
  return docs.map(({ chunk, terms }) => {
    const tf = {};
    for (const term of terms) tf[term] = (tf[term] || 0) + 1;
    let lexical = 0;
    const matched = new Set();
    const heading = String(chunk.section_heading || '').toLowerCase();
    const sourcePath = String(chunk.source_path || '').toLowerCase();
    for (const term of queryTerms) {
      const freq = tf[term] || 0;
      const inHeading = heading.includes(term);
      const inPath = sourcePath.includes(term);
      if (freq || inHeading || inPath) matched.add(term);
      const idf = Math.log((docCount - (df[term] || 0) + 0.5) / ((df[term] || 0) + 0.5) + 1);
      if (freq) lexical += idf * ((freq * 2.2) / (freq + 1.2 * (1 - 0.75 + 0.75 * terms.length / Math.max(1, avgLen))));
      if (inHeading) lexical += idf * 2.5;
      if (inPath) lexical += idf * 1.5;
    }
    lexical *= 1 + matched.size / Math.max(1, queryTerms.length);
    const reasons = matched.size ? [`lexical match: ${Array.from(matched).slice(0, 8).join(', ')}`] : ['low lexical overlap; retained as lower-ranked candidate'];
    if (/(as defined in|subject to|pursuant to|see section)/i.test(chunk.text)) reasons.push('contains explicit dependency/reference language');
    return {
      chunk_id: chunk.chunk_id,
      lexical_score: Number(lexical.toFixed(6)),
      semantic_score: 0,
      rerank_score: 0,
      final_score: Number(lexical.toFixed(6)),
      reasons,
    };
  }).sort((a, b) => b.final_score - a.final_score || a.chunk_id.localeCompare(b.chunk_id));
}

function norm(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function detectDependencies(index) {
  const termDefs = new Map();
  for (const chunk of index.chunks) {
    for (const match of chunk.text.matchAll(/"([^"]{2,80})"\s+(means|shall mean|is defined as|refers to)\b/gi)) {
      const term = norm(match[1]);
      if (!termDefs.has(term)) termDefs.set(term, []);
      termDefs.get(term).push(chunk);
    }
  }
  const links = [];
  const seen = new Set();
  const add = (source, target, relationType, evidence) => {
    const key = `${source.chunk_id}:${target ? target.chunk_id : ''}:${relationType}:${evidence}`;
    if (seen.has(key)) return;
    seen.add(key);
    links.push({
      source_chunk_id: source.chunk_id,
      target_chunk_id: target ? target.chunk_id : null,
      relation_type: relationType,
      evidence,
      source_document_id: source.document_id,
      target_document_id: target ? target.document_id : null,
      resolved: Boolean(target),
      warning: target ? null : `Unresolved reference: ${evidence}`,
    });
  };
  const resolve = (source, label) => {
    const target = norm(label);
    return index.chunks.find((chunk) => chunk.chunk_id !== source.chunk_id && (
      norm(chunk.section_heading || '').includes(target)
      || norm(chunk.source_path).includes(target)
      || norm(chunk.text.slice(0, 300)).includes(target)
    ));
  };
  for (const chunk of index.chunks) {
    const lower = norm(chunk.text);
    for (const [term, defs] of termDefs.entries()) {
      if (lower.includes(term) && defs.every((def) => def.chunk_id !== chunk.chunk_id)) add(chunk, defs[0], 'defined_term', term);
    }
    const patterns = [
      ['defined_in', /\bas defined in\s+((section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.-]+)/gi],
      ['subject_to', /\bsubject to\s+((section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.-]+)/gi],
      ['pursuant_to', /\bpursuant to\s+((section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.-]+)/gi],
      ['see_reference', /\bsee\s+((section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.-]+)/gi],
    ];
    for (const [relationType, re] of patterns) {
      for (const match of chunk.text.matchAll(re)) add(chunk, resolve(chunk, match[1]), relationType, match[1]);
    }
  }
  return links;
}

function selectReceiptContext(index, options = {}) {
  const query = options.query || '';
  const tokenBudget = Number(options.tokenBudget || options.token_budget || options.budget || 8000);
  const ranked = rankChunks(index, query);
  const deps = detectDependencies(index);
  const chunksById = Object.fromEntries(index.chunks.map((chunk) => [chunk.chunk_id, chunk]));
  const selectedIds = [];
  let selectedTokens = 0;
  const warnings = [];
  const add = (chunkId) => {
    const chunk = chunksById[chunkId];
    if (!chunk || selectedIds.includes(chunkId) || selectedTokens + chunk.token_count > tokenBudget) return false;
    selectedIds.push(chunkId);
    selectedTokens += chunk.token_count;
    return true;
  };
  for (const rank of ranked) {
    if (!add(rank.chunk_id)) continue;
    for (const dep of deps.filter((d) => d.source_chunk_id === rank.chunk_id)) {
      if (dep.target_chunk_id && !selectedIds.includes(dep.target_chunk_id) && !add(dep.target_chunk_id)) {
        warnings.push(`Dependency not included due to budget: ${rank.chunk_id} -> ${dep.target_chunk_id} (${dep.relation_type})`);
      } else if (!dep.resolved && dep.warning) {
        warnings.push(dep.warning);
      }
    }
  }
  const rankById = Object.fromEntries(ranked.map((rank) => [rank.chunk_id, rank]));
  const selected_context = selectedIds.map((chunkId) => {
    const chunk = chunksById[chunkId];
    const sourceDeps = deps.filter((d) => d.source_chunk_id === chunkId);
    return {
      chunk_id: chunk.chunk_id,
      source_path: chunk.source_path,
      section_heading: chunk.section_heading || null,
      page_number: chunk.page_number || null,
      byte_start: chunk.byte_start,
      byte_end: chunk.byte_end,
      token_start: chunk.token_start,
      token_end: chunk.token_end,
      token_count: chunk.token_count,
      score: rankById[chunkId] ? rankById[chunkId].final_score : 0,
      reasons: rankById[chunkId] ? rankById[chunkId].reasons : ['included as dependency'],
      dependencies_included: sourceDeps.map((d) => d.target_chunk_id).filter((id) => id && selectedIds.includes(id)),
      dependencies_missing: sourceDeps.map((d) => d.target_chunk_id || d.evidence).filter((id) => !selectedIds.includes(id)),
      fingerprint: chunk.fingerprint,
      text: chunk.text,
    };
  });
  const omitted_context = ranked.filter((rank) => !selectedIds.includes(rank.chunk_id)).slice(0, 20).map((rank) => {
    const chunk = chunksById[rank.chunk_id];
    return {
      chunk_id: chunk.chunk_id,
      source_path: chunk.source_path,
      section_heading: chunk.section_heading || null,
      page_number: chunk.page_number || null,
      token_count: chunk.token_count,
      score: rank.final_score,
      reasons: rank.reasons,
      omission_reason: selectedTokens + chunk.token_count > tokenBudget ? 'budget_limit' : 'lower ranked than selected context under token budget',
      fingerprint: chunk.fingerprint,
      text_preview: chunk.text.replace(/\s+/g, ' ').slice(0, 240),
    };
  });
  const sourceTokens = index.chunks.reduce((sum, chunk) => sum + chunk.token_count, 0);
  const selectedToSource = selectedTokens / Math.max(1, sourceTokens);
  const relevantOmitted = omitted_context.filter((item) => item.score > 0).length;
  if (relevantOmitted) warnings.push(`${relevantOmitted} relevant chunk(s) were omitted; inspect omitted_context.`);
  const unresolved = deps.filter((dep) => !dep.resolved).length;
  if (unresolved) warnings.push(`${unresolved} dependency reference(s) could not be resolved to an ingested chunk.`);
  const ranking_reasons = Object.fromEntries(ranked.map((rank) => [rank.chunk_id, rank.reasons]));
  const source_fingerprints = {
    documents: Object.fromEntries(index.documents.map((doc) => [doc.source_path, doc.fingerprint])),
    chunks: Object.fromEntries(index.chunks.map((chunk) => [chunk.chunk_id, chunk.fingerprint])),
  };
  const risk_summary = {
    coverage_score: Number((0.45 * selectedToSource + 0.35 * (selected_context.length / Math.max(1, index.chunks.length)) + 0.2 * (1 - unresolved / Math.max(1, deps.length))).toFixed(6)),
    review_level: relevantOmitted || unresolved ? 'medium' : 'low',
    selected_chunks: selected_context.length,
    total_chunks: index.chunks.length,
    omitted_relevant_chunks: relevantOmitted,
    unresolved_dependency_count: unresolved,
    controls: {
      dependency_closure: unresolved ? 'partial' : 'complete',
      omitted_evidence_pressure: relevantOmitted ? 'medium' : 'low',
      replayable_fingerprints: true,
      local_no_llm_judgment: true,
    },
  };
  const payload = {
    schema_version: SCHEMA_VERSION,
    query,
    token_budget: tokenBudget,
    selected_context,
    omitted_context,
    dependency_links: deps,
    ranking_reasons,
    compression_ratio: {
      source_tokens: sourceTokens,
      selected_tokens: selectedTokens,
      tokens_saved: Math.max(0, sourceTokens - selectedTokens),
      selected_to_source_ratio: Number(selectedToSource.toFixed(6)),
      source_to_selected_ratio: Number((sourceTokens / Math.max(1, selectedTokens)).toFixed(6)),
      reduction_pct: Number(((1 - selectedToSource) * 100).toFixed(3)),
    },
    source_fingerprints,
    risk_summary,
    warnings: Array.from(new Set(warnings)),
    outcome_links: [],
  };
  const reproducibility_hash = stableHash(payload);
  return { receipt_id: `cr_${reproducibility_hash.slice(0, 12)}`, ...payload, reproducibility_hash };
}

function createContextReceipt(documents, options = {}) {
  return selectReceiptContext(ingestReceiptDocuments(documents, options), options);
}

function renderContextReceipt(receipt) {
  const lines = [
    `# Context Receipt ${receipt.receipt_id}`,
    '',
    `Query: \`${receipt.query}\``,
    '',
    '## Coverage And Risk Controls',
    '',
    `- Coverage score: ${receipt.risk_summary.coverage_score}`,
    `- Review level: ${receipt.risk_summary.review_level}`,
    `- Dependency closure: ${receipt.risk_summary.controls.dependency_closure}`,
    '',
    '## Included Context',
    '',
  ];
  for (const item of receipt.selected_context || []) {
    lines.push(`### ${item.chunk_id}`, `- Source: \`${item.source_path}\``, `- Why: ${item.reasons.join('; ')}`, '');
  }
  lines.push('## Omitted Context', '');
  for (const item of receipt.omitted_context || []) {
    lines.push(`### ${item.chunk_id}`, `- Why omitted: ${item.omission_reason}`, `- Preview: ${item.text_preview}`, '');
  }
  return `${lines.join('\n')}\n`;
}

function explainReceiptOmission(receipt, chunkId) {
  const omitted = (receipt.omitted_context || []).find((item) => item.chunk_id === chunkId);
  if (omitted) return `${chunkId} was omitted from ${omitted.source_path}: ${omitted.omission_reason}.`;
  if ((receipt.selected_context || []).some((item) => item.chunk_id === chunkId)) return `${chunkId} was not omitted; it is present in selected_context.`;
  return `${chunkId} is not present in this receipt.`;
}

module.exports = {
  SCHEMA_VERSION,
  ingestReceiptDocuments,
  selectReceiptContext,
  createContextReceipt,
  renderContextReceipt,
  explainReceiptOmission,
};
