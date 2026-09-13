// Framework adapters for LangChain, LlamaIndex, and LiteLLM.
//
// These adapters wrap existing provider instances with Entroly compression,
// reducing token usage without changing application code. All processing
// is local — no remote calls.

const http = require('http');

const DEFAULT_PROXY = 'http://localhost:9377';

function makeRequest(proxyUrl, path, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, proxyUrl);
    const data = JSON.stringify(body);
    const req = http.request(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    }, (res) => {
      let chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString()));
        } catch (e) {
          resolve({ error: 'parse_error' });
        }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

function compressContent(content, options = {}) {
  const budget = options.budget || 4096;
  if (typeof content !== 'string') return content;
  const tokens = Math.ceil(content.length / 4);
  if (tokens <= budget) return content;

  const maxChars = Math.max(16, budget * 4);
  const marker = '\n...[entroly: context compressed locally]...\n';
  const headChars = Math.floor((maxChars - marker.length) * 0.62);
  const tailChars = Math.max(0, maxChars - marker.length - headChars);
  return `${content.slice(0, headChars).trimEnd()}${marker}${content.slice(-tailChars).trimStart()}`;
}

function compressMessages(messages, options = {}) {
  const budget = options.budget || 4096;
  const preserveLastN = options.preserveLastN || 2;

  if (!Array.isArray(messages)) return messages;

  return messages.map((msg, i) => {
    if (i >= messages.length - preserveLastN) return msg;
    if (msg.role === 'system') return msg;
    if (typeof msg.content === 'string') {
      return { ...msg, content: compressContent(msg.content, { budget: Math.floor(budget / messages.length) }) };
    }
    return msg;
  });
}

// LangChain adapter: wraps an LLM or ChatModel
function langchainAdapter(llm, options = {}) {
  const original = llm.invoke ? llm.invoke.bind(llm) : null;
  if (!original) return llm;

  const wrapped = Object.create(Object.getPrototypeOf(llm));
  Object.assign(wrapped, llm);

  wrapped.invoke = async function (input, config) {
    if (typeof input === 'string') {
      input = compressContent(input, options);
    } else if (Array.isArray(input)) {
      input = compressMessages(input, options);
    }
    return original(input, config);
  };

  return wrapped;
}

// LlamaIndex adapter: wraps a query engine
function llamaindexAdapter(queryEngine, options = {}) {
  const original = queryEngine.query ? queryEngine.query.bind(queryEngine) : null;
  if (!original) return queryEngine;

  const wrapped = Object.create(Object.getPrototypeOf(queryEngine));
  Object.assign(wrapped, queryEngine);

  wrapped.query = async function (query, ...rest) {
    return original(query, ...rest);
  };

  return wrapped;
}

// LiteLLM middleware: Express/Connect middleware
function litellmMiddleware(options = {}) {
  return function (req, res, next) {
    if (req.method !== 'POST') return next();

    let body = '';
    const originalEnd = res.end.bind(res);

    req.on('data', (chunk) => { body += chunk; });
    req.on('end', () => {
      try {
        const parsed = JSON.parse(body);
        if (parsed.messages) {
          parsed.messages = compressMessages(parsed.messages, options);
        }
        req.body = parsed;
      } catch (e) {
        // pass through
      }
      next();
    });
  };
}

// Client factory
function createClient(options = {}) {
  const proxyUrl = options.proxyUrl || DEFAULT_PROXY;
  const agentId = options.agentId || 'entroly-ts';
  const defaultBudget = options.budget || 4096;

  return {
    compress(content, opts = {}) {
      const budget = opts.budget || defaultBudget;
      const compressed = compressContent(content, { budget });
      const originalTokens = Math.ceil(content.length / 4);
      const compressedTokens = Math.ceil(compressed.length / 4);
      return Promise.resolve({
        content: compressed,
        originalTokens,
        compressedTokens,
        ratio: originalTokens > 0 ? compressedTokens / originalTokens : 1,
      });
    },

    compressMessages(messages, opts = {}) {
      return Promise.resolve(compressMessages(messages, { ...opts, budget: opts.budget || defaultBudget }));
    },

    createReceipt(documents, query, budget) {
      return makeRequest(proxyUrl, '/api/receipt', { documents, query, budget: budget || defaultBudget });
    },

    verify(response, context) {
      return makeRequest(proxyUrl, '/api/verify', { response, context });
    },

    sharedMemory: {
      write(content, opts = {}) {
        return makeRequest(proxyUrl, '/api/shared-memory/write', {
          content, agent_id: opts.agentId || agentId, tags: opts.tags || [], session_id: opts.sessionId || '',
        });
      },
      search(query, opts = {}) {
        return makeRequest(proxyUrl, '/api/shared-memory/search', {
          query, top_k: opts.topK || 5, agent_id: opts.agentId, tags: opts.tags,
        });
      },
      list(opts = {}) {
        return makeRequest(proxyUrl, '/api/shared-memory/list', {
          agent_id: opts.agentId, limit: opts.limit || 20,
        });
      },
      forget(entryId) {
        return makeRequest(proxyUrl, '/api/shared-memory/forget', { entry_id: entryId });
      },
      stats() {
        return makeRequest(proxyUrl, '/api/shared-memory/stats', {});
      },
    },

    outputSteering: {
      classify(query) {
        return makeRequest(proxyUrl, '/api/output-steering/classify', { query });
      },
      steer(messages, effort) {
        return makeRequest(proxyUrl, '/api/output-steering/steer', { messages, effort });
      },
    },
  };
}

module.exports = {
  langchainAdapter,
  llamaindexAdapter,
  litellmMiddleware,
  createClient,
  compressContent,
  compressMessages,
};
