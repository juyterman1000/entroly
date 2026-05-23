/**
 * Multimodal Ingestion — pure-JS port of entroly/multimodal.py
 * Diff parser + intent classifier. Stubs for diagram/voice (require native deps).
 */

'use strict';

const { parseDiff, classifyDiffIntent } = require('./cogops');

function ingestDiff(diffText, source, commitMessage = '') {
  const parsed = parseDiff(diffText);
  const intent = classifyDiffIntent(diffText, commitMessage);

  // Extract symbols changed (simple heuristic: lines starting with def/fn/class/function)
  const symbolsChanged = [];
  const symRe = /^[+-]\s*(?:def|fn|pub fn|class|function|const|let|var|struct|enum|trait|impl|export)\s+(\w+)/gm;
  let m;
  while ((m = symRe.exec(diffText)) !== null) {
    if (!symbolsChanged.includes(m[1])) symbolsChanged.push(m[1]);
  }

  // Build structured summary text for ingestion
  const text = [
    `## Change Summary: ${source}`,
    commitMessage ? `\n**Commit:** ${commitMessage}` : '',
    `\n**Intent:** ${intent}`,
    `**Files changed:** ${parsed.files_modified.length} (${parsed.files_modified.join(', ')})`,
    `**Lines:** +${parsed.lines_added} / -${parsed.lines_removed}`,
    symbolsChanged.length ? `**Symbols changed:** ${symbolsChanged.join(', ')}` : '',
    '',
    '### Diff',
    '```diff',
    diffText.slice(0, 4000),
    '```',
  ].filter(Boolean).join('\n');

  return {
    text,
    token_estimate: Math.ceil(text.length / 4),
    metadata: {
      intent,
      files_changed: parsed.files_modified.length,
      added_lines: parsed.lines_added,
      removed_lines: parsed.lines_removed,
      symbols_changed: symbolsChanged,
    },
  };
}

function ingestDiagram() {
  return {
    error: 'ingest_diagram requires image processing (PIL/OCR) which is not available in the WASM/Node runtime. Use `pip install entroly` for diagram ingestion.',
    engine: 'javascript',
  };
}

function ingestVoice() {
  return {
    error: 'ingest_voice requires audio transcription (whisper) which is not available in the WASM/Node runtime. Use `pip install entroly` for voice ingestion.',
    engine: 'javascript',
  };
}

module.exports = { ingestDiff, ingestDiagram, ingestVoice };
