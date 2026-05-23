#!/usr/bin/env node
/**
 * Tests for Response Distillation (JavaScript)
 * ==============================================
 * Run: node test_distill.js
 */

'use strict';

const { distillResponse, distillSSEChunk } = require('./js/distill');

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (cond) {
    passed++;
  } else {
    failed++;
    console.error(`  FAIL: ${msg}`);
  }
}

// ── Test: Code blocks preserved ──
{
  const input = "Sure! Here's the fix:\n\n```python\ndef fix():\n    return 42\n```\n\nHope this helps!";
  const { text, originalTokens, compressedTokens } = distillResponse(input);
  assert(text.includes('def fix():'), 'Code block preserved');
  assert(text.includes('return 42'), 'Code content preserved');
  assert(!text.includes('Sure!'), 'Pleasantry stripped');
  assert(!text.includes('Hope this helps'), 'Closing filler stripped');
  assert(compressedTokens < originalTokens, 'Token count decreased');
}

// ── Test: Pleasantries stripped ──
{
  const input = "I'd be happy to help! Let me take a look at your code.\nThe issue is in the auth module.\nFeel free to ask if you need help!";
  const { text } = distillResponse(input);
  assert(!text.includes('happy to help'), 'Happy to help removed');
  assert(!text.includes('Let me take a look'), 'Preamble removed');
  assert(!text.includes('Feel free'), 'Closing removed');
  assert(text.includes('auth module'), 'Technical content preserved');
}

// ── Test: Hedging stripped ──
{
  const input = "I think the problem is in the database connection. It seems like the timeout is too short.";
  const { text } = distillResponse(input);
  assert(!text.match(/I think /i), 'Hedging "I think" removed');
  assert(!text.match(/It seems like /i), 'Hedging "It seems" removed');
  assert(text.includes('database connection'), 'Content preserved');
}

// ── Test: Verbose connectors simplified ──
{
  const input = "In order to fix this, due to the fact that the API changed.";
  const { text } = distillResponse(input);
  assert(!text.includes('In order to'), 'Verbose connector removed');
  assert(text.includes('To') || text.includes('to'), 'Terse replacement present');
}

// ── Test: Empty input ──
{
  const { text, originalTokens, compressedTokens } = distillResponse('');
  assert(text === '', 'Empty returns empty');
  assert(originalTokens === 0, 'Zero original tokens');
  assert(compressedTokens === 0, 'Zero compressed tokens');
}

// ── Test: Short input passthrough ──
{
  const input = 'Fix the bug.';
  const { text, originalTokens, compressedTokens } = distillResponse(input);
  assert(text === input, 'Short input unchanged');
  assert(originalTokens === compressedTokens, 'Token counts match');
}

// ── Test: Multiple code blocks ──
{
  const input = "Here's the fix:\n\n```js\nconsole.log('a')\n```\n\nAnd:\n\n```js\nassert(true)\n```\n\nLet me know if you need more!";
  const { text } = distillResponse(input);
  assert(text.includes("console.log('a')"), 'First code block preserved');
  assert(text.includes('assert(true)'), 'Second code block preserved');
  assert(!text.includes('Let me know'), 'Closing filler stripped');
}

// ── Test: SSE chunk ──
{
  const result = distillSSEChunk('Sure! The issue is...');
  assert(!result.includes('Sure!'), 'SSE chunk filler stripped');
}

// ── Test: SSE code passthrough ──
{
  const chunk = '```python\ndef foo():\n```';
  assert(distillSSEChunk(chunk) === chunk, 'SSE code chunk unchanged');
}

// ── Test: Lite mode less aggressive ──
{
  const input = "Sure! I think the issue is in the parser module. In order to fix this you need to update the regex configuration and tokenizer.";
  const { text } = distillResponse(input, { mode: 'lite' });
  assert(!text.includes('Sure!'), 'Lite strips pleasantries');
  assert(text.includes('In order to'), 'Lite keeps verbose connectors');
}

// ── Results ──
console.log(`\n  Response Distillation JS: ${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
