// App-level SDK helpers for JavaScript/TypeScript applications.
//
// These helpers are dependency-free. They do not import provider SDKs or the
// Vercel AI SDK; instead they expose plain middleware/wrapper shapes that those
// libraries can consume. All transforms are local and preserve provider-owned
// controls such as model, temperature, tools, tool_choice, safetySettings, and
// thinking fields.

function estimateTokens(value) {
  if (value == null) return 0;
  if (typeof value === 'string') return Math.ceil(value.length / 4);
  if (Array.isArray(value)) return value.reduce((n, item) => n + estimateTokens(item), 0);
  if (typeof value === 'object') {
    if (typeof value.text === 'string') return estimateTokens(value.text);
    if (typeof value.content === 'string' || Array.isArray(value.content)) {
      return estimateTokens(value.content);
    }
    if (Array.isArray(value.parts)) return estimateTokens(value.parts);
  }
  return 0;
}

function cloneJson(value) {
  if (value == null) return value;
  return JSON.parse(JSON.stringify(value));
}

function stableStringify(value) {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function fingerprint(value) {
  const crypto = require('crypto');
  return crypto.createHash('sha256').update(stableStringify(value)).digest('hex').slice(0, 12);
}

function compactText(text, tokenBudget) {
  if (typeof text !== 'string' || tokenBudget <= 0 || estimateTokens(text) <= tokenBudget) {
    return text;
  }
  const maxChars = Math.max(16, tokenBudget * 4);
  const marker = '\n...[entroly: middle omitted; request stayed local]...\n';
  if (maxChars <= marker.length + 8) return text.slice(0, maxChars);
  const headChars = Math.floor((maxChars - marker.length) * 0.62);
  const tailChars = Math.max(0, maxChars - marker.length - headChars);
  return `${text.slice(0, headChars).trimEnd()}${marker}${text.slice(-tailChars).trimStart()}`;
}

function transformContent(content, tokenBudget) {
  if (typeof content === 'string') return compactText(content, tokenBudget);
  if (Array.isArray(content)) {
    return content.map(part => transformContent(part, tokenBudget));
  }
  if (content && typeof content === 'object') {
    const out = { ...content };
    if (typeof out.text === 'string') out.text = compactText(out.text, tokenBudget);
    else if (typeof out.content === 'string' || Array.isArray(out.content)) {
      out.content = transformContent(out.content, tokenBudget);
    } else if (Array.isArray(out.parts)) {
      out.parts = out.parts.map(part => transformContent(part, tokenBudget));
    }
    return out;
  }
  return content;
}

function optimizeMessages(messages, options) {
  if (!Array.isArray(messages)) return { messages, tokensBefore: 0, tokensAfter: 0 };
  const budget = Math.max(1, Number(options.budget || 32000));
  const preserveLastN = Math.max(1, Number(options.preserveLastN || 4));
  const out = cloneJson(messages);
  let tokensBefore = estimateTokens(out);
  if (tokensBefore <= budget) return { messages: out, tokensBefore, tokensAfter: tokensBefore };

  const targetOlderTokens = Math.max(1, budget - estimateTokens(out.slice(-preserveLastN)));
  const older = Math.max(1, out.length - preserveLastN);
  const perOldMessageBudget = Math.max(16, Math.floor(targetOlderTokens / older));
  for (let i = 0; i < out.length - preserveLastN; i += 1) {
    out[i].content = transformContent(out[i].content, perOldMessageBudget);
  }
  return { messages: out, tokensBefore, tokensAfter: estimateTokens(out) };
}

function optimizeOpenAIParams(params, options = {}) {
  const out = { ...params };
  const before = estimateTokens(out.messages || out.input || out.prompt);
  let after = before;
  if (Array.isArray(out.messages)) {
    const result = optimizeMessages(out.messages, options);
    out.messages = result.messages;
    after = result.tokensAfter;
  } else if (typeof out.input === 'string') {
    out.input = compactText(out.input, Number(options.budget || 32000));
    after = estimateTokens(out.input);
  } else if (typeof out.prompt === 'string') {
    out.prompt = compactText(out.prompt, Number(options.budget || 32000));
    after = estimateTokens(out.prompt);
  }
  return {
    params: out,
    decision: {
      provider: 'openai',
      action: after < before ? 'compress' : 'observe',
      tokensBefore: before,
      tokensAfter: after,
      requestFingerprint: fingerprint(out),
    },
  };
}

function optimizeAnthropicParams(params, options = {}) {
  const out = { ...params };
  const before = estimateTokens(out.messages || out.system);
  let after = before;
  if (Array.isArray(out.messages)) {
    const result = optimizeMessages(out.messages, options);
    out.messages = result.messages;
    after = result.tokensAfter + estimateTokens(out.system);
  }
  return {
    params: out,
    decision: {
      provider: 'anthropic',
      action: after < before ? 'compress' : 'observe',
      tokensBefore: before,
      tokensAfter: after,
      requestFingerprint: fingerprint(out),
    },
  };
}

function optimizeGeminiParams(params, options = {}) {
  const out = { ...params };
  const contents = Array.isArray(out.contents) ? cloneJson(out.contents) : out.contents;
  const before = estimateTokens(contents);
  let after = before;
  if (Array.isArray(contents)) {
    const asMessages = contents.map(item => ({
      role: item.role || 'user',
      content: Array.isArray(item.parts) ? item.parts : item,
    }));
    const result = optimizeMessages(asMessages, options);
    out.contents = result.messages.map(item => ({
      role: item.role,
      parts: Array.isArray(item.content) ? item.content : [{ text: String(item.content || '') }],
    }));
    after = estimateTokens(out.contents);
  }
  return {
    params: out,
    decision: {
      provider: 'gemini',
      action: after < before ? 'compress' : 'observe',
      tokensBefore: before,
      tokensAfter: after,
      requestFingerprint: fingerprint(out),
    },
  };
}

function optimizeRequestParams(params, options = {}) {
  const provider = (options.provider || '').toLowerCase();
  if (provider === 'anthropic' || (params && Array.isArray(params.messages) && typeof params.system === 'string')) {
    return optimizeAnthropicParams(params, options);
  }
  if (provider === 'gemini' || Array.isArray(params && params.contents)) {
    return optimizeGeminiParams(params, options);
  }
  return optimizeOpenAIParams(params, options);
}

function createEntrolyMiddleware(options = {}) {
  return {
    specificationVersion: 'v3',
    transformParams: async ({ params }) => optimizeRequestParams(params, options).params,
    wrapGenerate: async ({ doGenerate }) => doGenerate(),
    wrapStream: async ({ doStream }) => doStream(),
  };
}

function wrapOpenAI(client, options = {}) {
  return {
    raw: client,
    chat: {
      completions: {
        create: (params, ...rest) => client.chat.completions.create(
          optimizeOpenAIParams(params, options).params,
          ...rest,
        ),
      },
    },
    responses: client.responses ? {
      create: (params, ...rest) => client.responses.create(
        optimizeOpenAIParams(params, options).params,
        ...rest,
      ),
    } : undefined,
  };
}

function wrapAnthropic(client, options = {}) {
  return {
    raw: client,
    messages: {
      create: (params, ...rest) => client.messages.create(
        optimizeAnthropicParams(params, options).params,
        ...rest,
      ),
    },
  };
}

function wrapGemini(client, options = {}) {
  return {
    raw: client,
    models: {
      generateContent: (params, ...rest) => client.models.generateContent(
        optimizeGeminiParams(params, options).params,
        ...rest,
      ),
    },
  };
}

module.exports = {
  createEntrolyMiddleware,
  estimateTokens,
  optimizeAnthropicParams,
  optimizeGeminiParams,
  optimizeMessages,
  optimizeOpenAIParams,
  optimizeRequestParams,
  stableStringify,
  wrapAnthropic,
  wrapGemini,
  wrapOpenAI,
};
