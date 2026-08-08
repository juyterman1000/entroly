const assert = require('assert');
const {
  EntrolyGatewayClient,
  createGatewayMiddleware,
  wrapOpenAIWithGateway,
} = require('./js/gateway_client');

function headers(values) {
  return { forEach: fn => Object.entries(values).forEach(([key, value]) => fn(value, key)) };
}

async function main() {
  const calls = [];
  const mockFetch = async (url, init) => {
    calls.push({ url: String(url), init });
    if (String(url).includes('/retrieve-image')) {
      return { ok: true, status: 200, headers: headers({}), json: async () => ({ original_base64: 'ZXhhY3Q=' }) };
    }
    if (String(url).includes('/retrieve')) {
      return { ok: true, status: 200, headers: headers({}), json: async () => ({ original_content: 'exact' }) };
    }
    return {
      ok: true,
      status: 200,
      headers: headers({
        'x-entroly-receipt-count': '1',
        'x-entroly-recovery': 'stored',
        'x-entroly-compression': 'changed',
      }),
      json: async () => ({ messages: [{ role: 'user', content: 'compressed [entroly-recovery:r:s]' }] }),
    };
  };
  const client = new EntrolyGatewayClient({
    fetch: mockFetch,
    budgetTokens: 900,
    accessToken: 'x'.repeat(40),
    sidecarToken: 'sidecar-secret',
  });
  const compressed = await client.compress({ messages: [{ role: 'user', content: 'long' }] });
  assert.strictEqual(compressed.receipt.count, 1);
  assert.strictEqual(compressed.receipt.recovery, 'stored');
  assert(calls[0].url.includes('budget_tokens=900'));
  assert.strictEqual(calls[0].init.headers['x-entroly-access-token'], 'x'.repeat(40));
  assert.strictEqual(calls[0].init.headers['x-entroly-sidecar-token'], 'sidecar-secret');

  const recovered = await client.retrieve({ receiptId: 'r', spanId: 's' });
  assert.strictEqual(recovered.original_content, 'exact');
  const recoveredImage = await client.retrieveImage({ receiptId: 'img:0123456789abcdef01234567' });
  assert.strictEqual(recoveredImage.original_base64, 'ZXhhY3Q=');

  const middleware = createGatewayMiddleware({ client });
  const transformed = await middleware.transformParams({ params: { messages: [] } });
  assert(Array.isArray(transformed.messages));

  let forwarded;
  const wrapped = wrapOpenAIWithGateway({
    chat: { completions: { create: async value => { forwarded = value; return value; } } },
  }, { gatewayClient: client });
  await wrapped.chat.completions.create({ messages: [{ role: 'user', content: 'raw' }] });
  assert(forwarded.messages[0].content.includes('entroly-recovery'));

  assert.throws(
    () => new EntrolyGatewayClient({ baseUrl: 'https://example.com', fetch: mockFetch }),
    /allowRemote/,
  );
  assert.throws(
    () => new EntrolyGatewayClient({ fetch: mockFetch, budgetTokens: 0 }),
    /positive integer/,
  );
  assert.doesNotThrow(
    () => new EntrolyGatewayClient({ baseUrl: 'http://[::1]:9377', fetch: mockFetch }),
  );
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
