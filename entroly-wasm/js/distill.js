/**
 * Response Distillation — Output-Side Token Optimization (JavaScript)
 * ====================================================================
 *
 * Strips filler, pleasantries, hedging, and meta-commentary from LLM
 * responses while preserving all code blocks and technical content.
 *
 * Grounded in:
 *   - Selective Context (Li et al., EMNLP 2023) — self-information scoring
 *   - TRIM (arXiv 2025) — omit inferable words
 *
 * Usage:
 *   const { distillResponse } = require('entroly-wasm/js/distill');
 *   const { text, originalTokens, compressedTokens } = distillResponse(llmOutput);
 *
 * @module distill
 */

'use strict';

// ── Filler patterns: high-frequency, zero-information phrases ──

const FILLER_PATTERNS = [
  // Pleasantries (zero technical content)
  [/^(?:Sure!?\s*|Of course!?\s*|Absolutely!?\s*|Great question!?\s*|Happy to help!?\s*|I'd be happy to help\.?\s*|Let me help you with that\.?\s*|No problem!?\s*|Certainly!?\s*|You're welcome!?\s*)/i, ''],
  // Preamble ("I'm going to...")
  [/^(?:Let me |I'll |I will |I'm going to |I can |I need to )(?:take a look|look at|review|examine|analyze|check|investigate)\b[^.]*\.\s*/i, ''],
  // Meta-commentary ("Here's what I found")
  [/^(?:Here(?:'s| is) (?:what|how|the|my|a|an)\b[^:]*:\s*)/i, ''],
  // Hedging (near-zero information)
  [/\b(?:I think |I believe |It seems like |It looks like |It appears that |As far as I can tell,?\s*|From what I can see,?\s*|If I understand correctly,?\s*)/gi, ''],
  // Filler transitions
  [/^(?:Now,?\s*|Next,?\s*|Then,?\s*|After that,?\s*|Moving on,?\s*|With that said,?\s*|That being said,?\s*|Having said that,?\s*)(?:let(?:'s| us) )/i, ''],
  // Closing pleasantries
  [/(?:Let me know if (?:you (?:have|need)|there(?:'s| is)|that)\b[^.!]*[.!]?\s*$)/i, ''],
  [/(?:Feel free to (?:ask|reach out|let me know)\b[^.!]*[.!]?\s*$)/i, ''],
  [/(?:Hope (?:this|that) helps!?\s*$)/i, ''],
  [/(?:Is there anything else\b[^.?!]*[.?!]?\s*$)/i, ''],
  // Redundant acknowledgments
  [/^(?:I see\.|I understand\.|Got it\.|Right\.|Okay\.)\s*/i, ''],
  // Verbose connectors
  [/\bIn order to\b/i, 'To'],
  [/\bDue to the fact that\b/i, 'Because'],
  [/\bAt this point in time\b/i, 'Now'],
  [/\bFor the purpose of\b/i, 'For'],
  [/\bIn the event that\b/i, 'If'],
  [/\bWith regard to\b/i, 'About'],
  [/\bIt is important to note that\b/i, 'Note:'],
  [/\bAs (?:mentioned|noted|stated) (?:earlier|above|previously),?\s*/i, ''],
];

// Lines that are pure filler (entire line is fluff)
const PURE_FILLER = /^(?:Sure(?:,| thing).*|I(?:'d| would) (?:be happy|love) to.*|Let me (?:know|explain|walk you through).*|(?:Here(?:'s| is) (?:a|the) (?:summary|breakdown|overview|explanation).*)|(?:To (?:summarize|recap|sum up):?.*)|(?:In (?:summary|conclusion):?.*))$/i;

/**
 * Apply response distillation to LLM output text.
 *
 * @param {string} text - Raw LLM response text
 * @param {Object} [options]
 * @param {'lite'|'full'|'ultra'} [options.mode='full'] - Compression intensity
 * @returns {{ text: string, originalTokens: number, compressedTokens: number }}
 */
function distillResponse(text, options = {}) {
  const mode = options.mode || 'full';

  if (!text || text.length < 50) {
    const count = text ? text.split(/\s+/).filter(Boolean).length : 0;
    return { text: text || '', originalTokens: count, compressedTokens: count };
  }

  const originalTokens = text.split(/\s+/).filter(Boolean).length;

  // Split into code blocks and prose — NEVER touch code blocks
  const parts = text.split(/(```[\s\S]*?```)/);
  const compressed = parts.map(part => {
    if (part.startsWith('```')) return part;
    return _compressProse(part, mode);
  });

  let result = compressed.join('');
  result = result.replace(/\n{3,}/g, '\n\n').trim();

  const compressedTokens = result ? result.split(/\s+/).filter(Boolean).length : 0;

  return { text: result, originalTokens, compressedTokens };
}

/**
 * Compress a prose section (not inside a code block).
 * @private
 */
function _compressProse(text, mode) {
  const lines = text.split('\n');
  const output = [];

  for (const line of lines) {
    const stripped = line.trim();

    // Skip pure filler lines
    if (stripped && PURE_FILLER.test(stripped)) continue;

    let compressed = stripped;

    if (mode === 'full' || mode === 'ultra') {
      for (const [pattern, replacement] of FILLER_PATTERNS) {
        compressed = compressed.replace(pattern, replacement);
      }
    } else if (mode === 'lite') {
      // Lite: only first 4 patterns (pleasantries + preamble)
      for (const [pattern, replacement] of FILLER_PATTERNS.slice(0, 4)) {
        compressed = compressed.replace(pattern, replacement);
      }
    }

    // Ultra: also strip articles and filler words
    if (mode === 'ultra') {
      compressed = compressed.replace(/\b(?:the|a|an|just|simply|basically|essentially)\s+/g, '');
      compressed = compressed.replace(/\s{2,}/g, ' ');
    }

    if (compressed.trim()) {
      const leading = line.length - line.trimStart().length;
      output.push(' '.repeat(leading) + compressed.trim());
    } else if (!stripped) {
      output.push('');
    }
  }

  return output.join('\n');
}

/**
 * Distill a single SSE content delta for streaming responses.
 *
 * @param {string} chunk - SSE content delta text
 * @param {Object} [options]
 * @param {'lite'|'full'|'ultra'} [options.mode='full']
 * @returns {string} Distilled chunk
 */
function distillSSEChunk(chunk, options = {}) {
  const mode = options.mode || 'full';

  if (!chunk || chunk.length < 10) return chunk;
  if (chunk.includes('```') || chunk.startsWith('    ')) return chunk;

  let result = chunk;
  const patternsToApply = FILLER_PATTERNS.slice(0, 8);
  for (const [pattern, replacement] of patternsToApply) {
    result = result.replace(pattern, replacement);
  }

  return result;
}

module.exports = { distillResponse, distillSSEChunk };
