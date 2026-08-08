// Receipt-first client for Entroly's local /v1/compress sidecar contract.
//
// Unlike the dependency-free local truncation helpers in app_sdk.js, this
// client delegates to the Python proxy so every changed payload is backed by
// Entroly's exact recovery store. Remote gateways are rejected by default.

function normalizedBaseUrl(value, allowRemote) {
  const url = new URL(value || 'http://127.0.0.1:9377');
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new TypeError('Entroly gateway URL must use http or https');
  }
  const host = url.hostname.toLowerCase();
  if (url.username || url.password) {
    throw new TypeError('Entroly gateway URL must not contain credentials');
  }
  const loopback = host === '127.0.0.1' || host === 'localhost' || host === '::1' || host === '[::1]';
  if (!loopback && !allowRemote) {
    throw new Error('Remote Entroly gateways require allowRemote: true');
  }
  url.pathname = url.pathname.replace(/\/$/, '');
  return url;
}

function fetchImplementation(explicitFetch) {
  const implementation = explicitFetch || globalThis.fetch;
  if (typeof implementation !== 'function') {
    throw new Error('No fetch implementation is available; pass { fetch } on Node 16');
  }
  return implementation;
}

function responseHeaders(headers) {
  const result = {};
  if (!headers || typeof headers.forEach !== 'function') return result;
  headers.forEach((value, key) => {
    if (String(key).toLowerCase().startsWith('x-entroly-')) result[key.toLowerCase()] = value;
  });
  return result;
}

function validatedProvider(value) {
  if (!['openai', 'anthropic', 'gemini'].includes(value)) {
    throw new TypeError(`Unsupported Entroly provider: ${value}`);
  }
  return value;
}

function validatedBudget(value) {
  const budget = Number(value);
  if (!Number.isInteger(budget) || budget <= 0) {
    throw new TypeError('Entroly gateway budgetTokens must be a positive integer');
  }
  return budget;
}

async function jsonResponse(response) {
  let body;
  try {
    body = await response.json();
  } catch (error) {
    throw new Error(`Entroly gateway returned non-JSON status ${response.status}`);
  }
  if (!response.ok) {
    const code = body && (body.error || body.detail);
    throw new Error(`Entroly gateway request failed (${response.status}): ${code || 'unknown error'}`);
  }
  return body;
}

class EntrolyGatewayClient {
  constructor(options = {}) {
    this.baseUrl = normalizedBaseUrl(options.baseUrl, Boolean(options.allowRemote));
    this.provider = validatedProvider(options.provider || 'openai');
    this.budgetTokens = validatedBudget(options.budgetTokens === undefined ? 32000 : options.budgetTokens);
    this.accessToken = options.accessToken || '';
    this.sidecarToken = options.sidecarToken || '';
    this.fetch = fetchImplementation(options.fetch);
  }

  headers() {
    const headers = { 'content-type': 'application/json', accept: 'application/json' };
    if (this.accessToken) headers['x-entroly-access-token'] = this.accessToken;
    if (this.sidecarToken) headers['x-entroly-sidecar-token'] = this.sidecarToken;
    return headers;
  }

  async compress(payload, options = {}) {
    const url = new URL('/v1/compress', this.baseUrl);
    url.searchParams.set('provider', validatedProvider(options.provider || this.provider));
    const budget = options.budgetTokens === undefined ? this.budgetTokens : options.budgetTokens;
    url.searchParams.set('budget_tokens', String(validatedBudget(budget)));
    const response = await this.fetch(url, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(payload),
      signal: options.signal,
    });
    const body = await jsonResponse(response);
    const headers = responseHeaders(response.headers);
    return {
      payload: body,
      receipt: {
        count: Number(headers['x-entroly-receipt-count'] || 0),
        recovery: headers['x-entroly-recovery'] || 'unknown',
        compression: headers['x-entroly-compression'] || 'unknown',
        headers,
      },
    };
  }

  async retrieve({ receiptId, spanId, retrievalId = '', signal } = {}) {
    if (!receiptId || !spanId) throw new TypeError('receiptId and spanId are required');
    const url = new URL('/retrieve', this.baseUrl);
    url.searchParams.set('receipt_id', receiptId);
    url.searchParams.set('span_id', spanId);
    if (retrievalId) url.searchParams.set('retrieval_id', retrievalId);
    const response = await this.fetch(url, {
      method: 'GET',
      headers: this.headers(),
      signal,
    });
    return jsonResponse(response);
  }

  async retrieveImage({ receiptId, signal } = {}) {
    if (!receiptId) throw new TypeError('receiptId is required');
    const url = new URL('/retrieve-image', this.baseUrl);
    url.searchParams.set('receipt_id', receiptId);
    const response = await this.fetch(url, {
      method: 'GET',
      headers: this.headers(),
      signal,
    });
    return jsonResponse(response);
  }
}

function createGatewayMiddleware(options = {}) {
  const client = options.client || new EntrolyGatewayClient(options);
  return {
    specificationVersion: 'v3',
    transformParams: async ({ params }) => (await client.compress(params, options)).payload,
    wrapGenerate: async ({ doGenerate }) => doGenerate(),
    wrapStream: async ({ doStream }) => doStream(),
    entrolyGateway: client,
  };
}

function wrapOpenAIWithGateway(client, options = {}) {
  const gateway = options.gatewayClient || new EntrolyGatewayClient({ ...options, provider: 'openai' });
  return {
    raw: client,
    entrolyGateway: gateway,
    chat: {
      completions: {
        create: async (params, ...rest) => client.chat.completions.create(
          (await gateway.compress(params, { ...options, provider: 'openai' })).payload,
          ...rest,
        ),
      },
    },
    responses: client.responses ? {
      create: async (params, ...rest) => client.responses.create(
        (await gateway.compress(params, { ...options, provider: 'openai' })).payload,
        ...rest,
      ),
    } : undefined,
  };
}

function wrapAnthropicWithGateway(client, options = {}) {
  const gateway = options.gatewayClient || new EntrolyGatewayClient({ ...options, provider: 'anthropic' });
  return {
    raw: client,
    entrolyGateway: gateway,
    messages: {
      create: async (params, ...rest) => client.messages.create(
        (await gateway.compress(params, { ...options, provider: 'anthropic' })).payload,
        ...rest,
      ),
    },
  };
}

function wrapGeminiWithGateway(client, options = {}) {
  const gateway = options.gatewayClient || new EntrolyGatewayClient({ ...options, provider: 'gemini' });
  return {
    raw: client,
    entrolyGateway: gateway,
    models: {
      generateContent: async (params, ...rest) => client.models.generateContent(
        (await gateway.compress(params, { ...options, provider: 'gemini' })).payload,
        ...rest,
      ),
    },
  };
}

module.exports = {
  EntrolyGatewayClient,
  createGatewayMiddleware,
  wrapAnthropicWithGateway,
  wrapGeminiWithGateway,
  wrapOpenAIWithGateway,
};
