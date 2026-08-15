# Entroly community savings on Cloudflare

This Worker is the free-tier deployment target for Entroly's privacy-safe,
opt-in product-health collector and public community-savings lower bound. It
preserves the Python collector's closed wire schema while replacing the hosted
Starlette process and SQLite file with Cloudflare Workers and D1.

## Privacy and trust boundary

- Telemetry remains disabled until a user explicitly runs `entroly telemetry on`.
- The Worker rejects unknown event fields, prompts, content, paths, model names,
  prices, exact token counts, exact costs, free text, and malformed dates.
- Savings reach the Worker only after the client rounds down to whole 1,000-token
  units and whole cents.
- Monthly pseudonyms are retained for at most 90 days. Older savings are folded
  into an identifier-free cumulative row; other expired events are deleted.
- A deletion request removes identifiable rows still inside the retention
  window. It cannot reverse already anonymized aggregate history.
- Worker observability is disabled in `wrangler.jsonc`; application code does
  not store IP addresses, request headers, or user agents.
- `GET /v1/public-savings` exposes totals and fixed methodology labels only.
  `GET /v1/summary` is hidden unless `ADMIN_TOKEN` is configured and supplied.

The optional `INGEST_TOKEN` secret can restrict uploads. If it is omitted, the
closed schema and two rate-limit bindings protect the public opt-in endpoint.

## Free-tier deployment

The free Workers plan currently includes 100,000 requests per day. D1 includes
5 million rows read per day, 100,000 rows written per day, and 5 GB total
storage. When a free limit is exhausted, requests fail instead of creating an
uncapped pay-as-you-go bill. Check Cloudflare's current
[Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/)
and [D1 pricing](https://developers.cloudflare.com/d1/platform/pricing/)
before deploying.

```powershell
cd deploy/cloudflare-community-savings
npm install
npm test
npm run check

npx wrangler login
npx wrangler d1 create entroly-community-savings
```

Copy the returned `database_id` into `wrangler.jsonc` (the Entroly production
database ID is already checked in), then initialize and deploy:

```powershell
npx wrangler d1 execute entroly-community-savings --remote --file=schema.sql
npx wrangler secret put ADMIN_TOKEN
npx wrangler deploy
```

`ADMIN_TOKEN` should be a new random secret. Do not commit it. `INGEST_TOKEN`
is optional; if configured, every contributing installation must also receive
it via `ENTROLY_TELEMETRY_TOKEN`.

After deployment:

1. Verify `/health` and `/v1/public-savings`.
2. Set `ENTROLY_TELEMETRY_ENDPOINT` only through the user's explicit consent
   flow; an environment variable alone cannot turn consent into upload.
3. The production aggregate endpoint is
   `https://entroly-community-savings.entroly-community-savings-worker.workers.dev/v1/public-savings`.
   Keep that URL in the `entroly-community-savings-endpoint` meta tag in
   `docs/index.html`.
4. Run the website tests and verify the cross-origin response from
   `https://juyterman1000.github.io` before publication.

## Rollback

Clear the website meta tag first so the site immediately returns to its
checked-in proof. Then roll back or delete the Worker. D1 remains separate and
can be exported or deleted from the Cloudflare dashboard.
