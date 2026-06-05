const assert = require('assert');
const {
  createEntrolyMiddleware,
  optimizeOpenAIParams,
  stableStringify,
  wrapAnthropic,
  wrapGemini,
  wrapOpenAI,
} = require('./js/app_sdk');

async function main() {
  const long = 'alpha '.repeat(4000);
  const params = {
    model: 'gpt-4o',
    temperature: 0.2,
    tool_choice: 'auto',
    tools: [{ type: 'function', function: { name: 'search', parameters: {} } }],
    messages: [
      { role: 'system', content: 'Be precise.' },
      { role: 'user', content: long },
      { role: 'assistant', content: long },
      { role: 'user', content: 'final question' },
    ],
  };

  const optimized = optimizeOpenAIParams(params, { budget: 500, preserveLastN: 1 });
  assert.strictEqual(optimized.params.model, params.model);
  assert.strictEqual(optimized.params.temperature, params.temperature);
  assert.deepStrictEqual(optimized.params.tools, params.tools);
  assert.strictEqual(optimized.decision.action, 'compress');
  assert(optimized.decision.tokensAfter < optimized.decision.tokensBefore);

  const middleware = createEntrolyMiddleware({ budget: 500, preserveLastN: 1 });
  const transformed = await middleware.transformParams({ params });
  assert.strictEqual(transformed.model, params.model);
  assert(transformed.messages[1].content.includes('[entroly: middle omitted'));

  let openaiCalled = false;
  const openai = wrapOpenAI({
    chat: { completions: { create: async p => { openaiCalled = true; return p; } } },
    responses: { create: async p => p },
  }, { budget: 500 });
  await openai.chat.completions.create(params);
  assert(openaiCalled);

  let anthropicCalled = false;
  const anthropic = wrapAnthropic({
    messages: { create: async p => { anthropicCalled = true; return p; } },
  }, { budget: 500 });
  await anthropic.messages.create({ system: 'sys', messages: params.messages, max_tokens: 1024 });
  assert(anthropicCalled);

  let geminiCalled = false;
  const gemini = wrapGemini({
    models: { generateContent: async p => { geminiCalled = true; return p; } },
  }, { budget: 500 });
  await gemini.models.generateContent({
    contents: [{ role: 'user', parts: [{ text: long }] }],
    generationConfig: { temperature: 0.1 },
  });
  assert(geminiCalled);

  assert.strictEqual(
    stableStringify({ b: 1, a: 2 }),
    stableStringify({ a: 2, b: 1 }),
  );
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
