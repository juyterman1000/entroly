const LEGACY_EVENT_SCHEMA = "entroly.product-telemetry.v2";
const EVENT_SCHEMA = "entroly.product-telemetry.v3";
const LEGACY_BATCH_SCHEMA = "entroly.product-telemetry-batch.v2";
const BATCH_SCHEMA = "entroly.product-telemetry-batch.v3";
const DELETE_SCHEMA = "entroly.product-telemetry-delete.v1";
const MAX_BATCH_EVENTS = 20;
const MAX_BATCH_BYTES = 64 * 1024;

const COMMANDS = new Set([
  "attach", "audit", "autotune", "batch", "benchmark", "cache",
  "capabilities", "clean", "compile", "compress", "config",
  "context-commit", "daemon", "dashboard", "demo", "digest", "docs",
  "doctor", "drift", "explain", "export", "feedback", "finetune", "go",
  "health", "import", "ingest", "init", "learn", "migrate", "optimize",
  "perf", "profile", "proof", "proxy", "ravs", "receipt", "recover",
  "role", "search", "select", "serve", "share", "simulate", "status",
  "sync", "telemetry", "uninstall", "unwrap", "value", "verify",
  "verify-claims", "verify-code", "witness", "wrap",
]);
const SURFACES = new Set([
  "cli", "compression_mcp", "mcp", "other", "proxy", "repository_mcp",
  "sdk_compress", "sdk_messages",
]);
const RESULTS = new Set(["success", "error", "interrupted"]);
const DURATION_BUCKETS = new Set(["lt_100ms", "lt_1s", "lt_10s", "lt_60s", "gte_60s"]);
const TOKEN_BUCKETS = new Set(["none", "1_99", "100_999", "1k_9k", "10k_99k", "100k_plus"]);
const REDUCTION_BUCKETS = new Set(["none", "lt_10", "10_29", "30_49", "50_69", "70_89", "90_plus"]);
const MEASUREMENT_SCOPES = new Set(["local_estimate", "provider_bound_estimate"]);
const COST_EVIDENCE = new Set(["not_available", "modeled_positive"]);
const EXIT_REASONS = new Set([
  "cost_concern", "hard_to_use", "install_problem", "integration_missing",
  "no_observed_benefit", "performance_problem", "privacy_concern",
  "quality_problem", "runtime_error", "switched_tool", "temporary_trial",
  "other", "prefer_not_to_say",
]);
const BENEFIT_OUTCOMES = new Set(["yes", "no", "unsure", "not_measured"]);
const USE_DURATIONS = new Set(["not_started", "lt_1d", "1_7d", "8_30d", "31_90d", "gt_90d", "unknown"]);
const ERROR_TYPES = new Set([
  "AssertionError", "ConnectionError", "ImportError", "LookupError",
  "MemoryError", "OSError", "PermissionError", "RuntimeError",
  "TimeoutError", "TypeError", "ValueError", "OtherError",
]);

const LEGACY_PROPERTIES = {
  activation: ["surface"],
  command: ["command", "result", "duration_bucket", "error_type"],
  exit_feedback: ["reason", "benefit_outcome", "primary_surface", "use_duration_bucket"],
  optimization_outcome: ["surface", "measurement_scope", "tokens_saved_bucket", "reduction_percent_bucket", "cost_evidence"],
  surface_started: ["surface"],
  surface_error: ["surface", "error_type"],
};
const EVENT_PROPERTIES = {
  ...LEGACY_PROPERTIES,
  savings_contribution: ["tokens_saved_thousands", "modeled_cost_saved_cents"],
};
const REQUIRED_PROPERTIES = {
  activation: ["surface"],
  command: ["command", "result", "duration_bucket"],
  exit_feedback: EVENT_PROPERTIES.exit_feedback,
  optimization_outcome: EVENT_PROPERTIES.optimization_outcome,
  surface_started: ["surface"],
  surface_error: ["surface", "error_type"],
  savings_contribution: EVENT_PROPERTIES.savings_contribution,
};
const EVENT_KEYS = [
  "schema_version", "event_id", "occurred_on", "installation_id",
  "event_name", "version", "platform", "python", "properties",
];

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function hasExactKeys(value, expected) {
  if (!isPlainObject(value)) return false;
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
}

function isIsoDate(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const [year, month, day] = value.split("-").map(Number);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  return parsed.getUTCFullYear() === year
    && parsed.getUTCMonth() === month - 1
    && parsed.getUTCDate() === day;
}

function dayString(date) {
  return date.toISOString().slice(0, 10);
}

function addDays(date, delta) {
  const next = new Date(date);
  next.setUTCDate(next.getUTCDate() + delta);
  return next;
}

function boundedRetention(env) {
  const parsed = Number.parseInt(env.RETENTION_DAYS || "90", 10);
  return Number.isInteger(parsed) ? Math.max(1, Math.min(parsed, 365)) : 90;
}

function validString(value, allowed) {
  return typeof value === "string" && allowed.has(value);
}

function validateProperties(eventName, schemaVersion, properties) {
  if (!isPlainObject(properties)) return false;
  const vocabulary = schemaVersion === LEGACY_EVENT_SCHEMA ? LEGACY_PROPERTIES : EVENT_PROPERTIES;
  const allowed = vocabulary[eventName];
  if (!allowed || Object.keys(properties).some((key) => !allowed.includes(key))) return false;
  if (REQUIRED_PROPERTIES[eventName].some((key) => !(key in properties))) return false;

  for (const [key, value] of Object.entries(properties)) {
    if (typeof value !== "string") return false;
    if (key === "command" && !COMMANDS.has(value)) return false;
    if ((key === "surface" || key === "primary_surface") && !SURFACES.has(value)) return false;
    if (key === "result" && !RESULTS.has(value)) return false;
    if (key === "duration_bucket" && !DURATION_BUCKETS.has(value)) return false;
    if (key === "error_type" && !ERROR_TYPES.has(value)) return false;
    if (key === "tokens_saved_bucket" && !TOKEN_BUCKETS.has(value)) return false;
    if (key === "reduction_percent_bucket" && !REDUCTION_BUCKETS.has(value)) return false;
    if (key === "measurement_scope" && !MEASUREMENT_SCOPES.has(value)) return false;
    if (key === "cost_evidence" && !COST_EVIDENCE.has(value)) return false;
    if (key === "reason" && !EXIT_REASONS.has(value)) return false;
    if (key === "benefit_outcome" && !BENEFIT_OUTCOMES.has(value)) return false;
    if (key === "use_duration_bucket" && !USE_DURATIONS.has(value)) return false;
    if ((key === "tokens_saved_thousands" || key === "modeled_cost_saved_cents")
      && !/^[0-9]{1,12}$/.test(value)) return false;
  }
  if (eventName === "savings_contribution") {
    return Number(properties.tokens_saved_thousands) > 0
      || Number(properties.modeled_cost_saved_cents) > 0;
  }
  return true;
}

export function validateEvent(value, now = new Date(), retentionDays = 90) {
  if (!hasExactKeys(value, EVENT_KEYS)) return false;
  if (![LEGACY_EVENT_SCHEMA, EVENT_SCHEMA].includes(value.schema_version)) return false;
  const vocabulary = value.schema_version === LEGACY_EVENT_SCHEMA ? LEGACY_PROPERTIES : EVENT_PROPERTIES;
  if (!(value.event_name in vocabulary)) return false;
  if (typeof value.event_id !== "string" || !/^[0-9a-f]{32}$/.test(value.event_id)) return false;
  if (typeof value.installation_id !== "string" || !/^[0-9a-f]{24}$/.test(value.installation_id)) return false;
  if (!isIsoDate(value.occurred_on)) return false;
  const today = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  const oldest = dayString(addDays(today, -retentionDays));
  const newest = dayString(addDays(today, 1));
  if (value.occurred_on < oldest || value.occurred_on > newest) return false;
  if (typeof value.version !== "string" || !/^[0-9A-Za-z.+_-]{1,32}$/.test(value.version)) return false;
  if (!validString(value.platform, new Set(["linux", "macos", "windows", "other"]))) return false;
  if (typeof value.python !== "string" || !/^(?:[0-9]{1,2}\.[0-9]{1,2}|unknown)$/.test(value.python)) return false;
  return validateProperties(value.event_name, value.schema_version, value.properties);
}

export function validateBatch(value, now = new Date(), retentionDays = 90) {
  if (!hasExactKeys(value, ["schema_version", "events"])) return null;
  if (![LEGACY_BATCH_SCHEMA, BATCH_SCHEMA].includes(value.schema_version)) return null;
  if (!Array.isArray(value.events) || value.events.length < 1 || value.events.length > MAX_BATCH_EVENTS) return null;
  if (!value.events.every((event) => validateEvent(event, now, retentionDays))) return null;
  return value.events;
}

export function validateDeletion(value) {
  if (!hasExactKeys(value, ["schema_version", "installation_ids"])) return null;
  if (value.schema_version !== DELETE_SCHEMA) return null;
  if (!Array.isArray(value.installation_ids)
    || value.installation_ids.length < 1
    || value.installation_ids.length > 4) return null;
  if (!value.installation_ids.every((item) => typeof item === "string" && /^[0-9a-f]{24}$/.test(item))) return null;
  return [...new Set(value.installation_ids)];
}

function json(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...headers },
  });
}

async function authorized(request, token) {
  if (!token) return true;
  const encoder = new TextEncoder();
  const [providedHash, expectedHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(request.headers.get("Authorization") || "")),
    crypto.subtle.digest("SHA-256", encoder.encode(`Bearer ${token}`)),
  ]);
  return crypto.subtle.timingSafeEqual(providedHash, expectedHash);
}

async function readJson(request, maxBytes) {
  const declared = Number.parseInt(request.headers.get("Content-Length") || "0", 10);
  if (Number.isFinite(declared) && declared > maxBytes) return { error: "payload_too_large" };
  if (!request.body) return { error: "invalid_schema" };
  const reader = request.body.getReader();
  const chunks = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > maxBytes) {
      await reader.cancel();
      return { error: "payload_too_large" };
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return { value: JSON.parse(text) };
  } catch {
    return { error: "invalid_schema" };
  }
}

async function withinRateLimit(env, installationIds) {
  if (env.GLOBAL_RATE_LIMITER) {
    const global = await env.GLOBAL_RATE_LIMITER.limit({ key: "v1-events" });
    if (!global.success) return false;
  }
  if (env.INGEST_RATE_LIMITER) {
    for (const installationId of new Set(installationIds)) {
      const result = await env.INGEST_RATE_LIMITER.limit({ key: installationId });
      if (!result.success) return false;
    }
  }
  return true;
}

async function prune(env, now) {
  const cutoff = dayString(addDays(now, -boundedRetention(env)));
  await env.DB.batch([
    env.DB.prepare(`
      INSERT INTO savings_archive (
        id, tokens_saved_thousands, modeled_cost_saved_cents,
        contribution_events, archived_through
      )
      SELECT 1,
             COALESCE(SUM(tokens_saved_thousands), 0),
             COALESCE(SUM(modeled_cost_saved_cents), 0),
             COUNT(*), ?
        FROM savings_contributions
       WHERE occurred_on < ?
      ON CONFLICT(id) DO UPDATE SET
        tokens_saved_thousands = tokens_saved_thousands + excluded.tokens_saved_thousands,
        modeled_cost_saved_cents = modeled_cost_saved_cents + excluded.modeled_cost_saved_cents,
        contribution_events = contribution_events + excluded.contribution_events,
        archived_through = excluded.archived_through
    `).bind(cutoff, cutoff),
    env.DB.prepare("DELETE FROM savings_contributions WHERE occurred_on < ?").bind(cutoff),
    env.DB.prepare("DELETE FROM product_events WHERE occurred_on < ?").bind(cutoff),
  ]);
}

async function ingest(env, events) {
  const statements = [];
  const eventStatementIndexes = [];
  for (const event of events) {
    eventStatementIndexes.push(statements.length);
    statements.push(env.DB.prepare(`
      INSERT OR IGNORE INTO product_events (
        event_id, occurred_on, installation_id, event_name,
        version, platform, python, properties_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      event.event_id,
      event.occurred_on,
      event.installation_id,
      event.event_name,
      event.version,
      event.platform,
      event.python,
      JSON.stringify(event.properties),
    ));
    if (event.event_name === "savings_contribution") {
      statements.push(env.DB.prepare(`
        INSERT OR IGNORE INTO savings_contributions (
          event_id, installation_id, occurred_on,
          tokens_saved_thousands, modeled_cost_saved_cents
        ) VALUES (?, ?, ?, ?, ?)
      `).bind(
        event.event_id,
        event.installation_id,
        event.occurred_on,
        Number(event.properties.tokens_saved_thousands),
        Number(event.properties.modeled_cost_saved_cents),
      ));
    }
  }
  const results = await env.DB.batch(statements);
  return eventStatementIndexes.reduce(
    (total, index) => total + Math.max(0, Number(results[index]?.meta?.changes || 0)),
    0,
  );
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const allowed = env.PUBLIC_ORIGIN || "https://juyterman1000.github.io";
  const headers = { "Cache-Control": "public, max-age=30", Vary: "Origin" };
  if (origin === allowed) headers["Access-Control-Allow-Origin"] = allowed;
  return headers;
}

export function publicSavingsPayload(row, now = new Date()) {
  const tokenUnits = Math.max(0, Number(row?.token_units || 0));
  const centUnits = Math.max(0, Number(row?.cent_units || 0));
  return {
    schema_version: "entroly.community-savings.v1",
    as_of_utc: now.toISOString().replace(/\.\d{3}Z$/, "+00:00"),
    reported_provider_tokens_saved: tokenUnits * 1000,
    reported_modeled_input_cost_avoided_usd: Number((centUnits / 100).toFixed(2)),
    evidence: {
      scope: "opt-in provider-bound Entroly proxy reductions",
      tokens: "each contribution rounded down to 1,000-token units",
      money: "modeled input cost rounded down to cents; not an invoice",
      historical_coverage: "since the community counter was enabled",
    },
    privacy: {
      prompts_or_content_exposed: false,
      identifiers_exposed: false,
      model_or_price_exposed: false,
      exact_per_request_values_exposed: false,
    },
  };
}

async function publicSavings(env, now) {
  await prune(env, now);
  const row = await env.DB.prepare(`
    SELECT
      a.tokens_saved_thousands + COALESCE(SUM(c.tokens_saved_thousands), 0) AS token_units,
      a.modeled_cost_saved_cents + COALESCE(SUM(c.modeled_cost_saved_cents), 0) AS cent_units
    FROM savings_archive AS a
    LEFT JOIN savings_contributions AS c ON 1 = 1
    WHERE a.id = 1
    GROUP BY a.id, a.tokens_saved_thousands, a.modeled_cost_saved_cents
  `).first();
  return publicSavingsPayload(row, now);
}

function increment(map, key) {
  map[key] = (map[key] || 0) + 1;
}

function sortedObject(value) {
  return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)));
}

function setFor(map, key) {
  if (!map[key]) map[key] = new Set();
  return map[key];
}

export function summarizeRows(rows, days, since, communitySavings) {
  const active = new Set();
  const activated = new Set();
  const benefited = new Set();
  const providerBenefited = new Set();
  const modeledCost = new Set();
  const errors = new Set();
  const exits = new Set();
  const platformActive = {};
  const platformBenefited = {};
  const counts = {
    events: {}, commands: {}, commandErrors: {}, errorTypes: {}, surfaces: {},
    versions: {}, tokenBuckets: {}, reductionBuckets: {}, scopes: {}, costs: {},
    exitReasons: {}, exitBenefits: {}, exitSurfaces: {}, exitDurations: {},
    platformEvents: {}, platformAttempts: {}, platformFailures: {}, platformExits: {},
  };
  let commandSuccesses = 0;
  let commandFailures = 0;
  let optimizationObservations = 0;
  let positiveObservations = 0;

  for (const row of rows) {
    const installation = String(row.installation_id);
    const eventName = String(row.event_name);
    const platform = String(row.platform);
    let properties = {};
    try { properties = JSON.parse(row.properties_json); } catch { properties = {}; }
    if (eventName !== "exit_feedback") {
      active.add(installation);
      setFor(platformActive, platform).add(installation);
    }
    increment(counts.platformEvents, platform);
    increment(counts.events, eventName);
    increment(counts.versions, String(row.version));
    if (eventName === "activation") activated.add(installation);
    if (["surface_started", "surface_error"].includes(eventName)) increment(counts.surfaces, String(properties.surface || "other"));
    if (eventName === "surface_error") {
      errors.add(installation);
      increment(counts.errorTypes, String(properties.error_type || "OtherError"));
    }
    if (eventName === "command") {
      const command = String(properties.command || "other");
      const result = String(properties.result || "error");
      increment(counts.commands, command);
      increment(counts.platformAttempts, platform);
      if (result === "success") commandSuccesses += 1;
      if (result === "error") {
        errors.add(installation);
        commandFailures += 1;
        increment(counts.platformFailures, platform);
        increment(counts.commandErrors, command);
        increment(counts.errorTypes, String(properties.error_type || "OtherError"));
      }
    }
    if (eventName === "optimization_outcome") {
      optimizationObservations += 1;
      const tokenBucket = String(properties.tokens_saved_bucket || "none");
      const reductionBucket = String(properties.reduction_percent_bucket || "none");
      const scope = String(properties.measurement_scope || "local_estimate");
      const cost = String(properties.cost_evidence || "not_available");
      increment(counts.tokenBuckets, tokenBucket);
      increment(counts.reductionBuckets, reductionBucket);
      increment(counts.scopes, scope);
      increment(counts.costs, cost);
      if (tokenBucket !== "none") {
        positiveObservations += 1;
        benefited.add(installation);
        setFor(platformBenefited, platform).add(installation);
        if (scope === "provider_bound_estimate") providerBenefited.add(installation);
        if (cost === "modeled_positive") modeledCost.add(installation);
      }
    }
    if (eventName === "exit_feedback") {
      exits.add(installation);
      increment(counts.platformExits, platform);
      increment(counts.exitReasons, String(properties.reason || "other"));
      increment(counts.exitBenefits, String(properties.benefit_outcome || "not_measured"));
      increment(counts.exitSurfaces, String(properties.primary_surface || "cli"));
      increment(counts.exitDurations, String(properties.use_duration_bucket || "unknown"));
    }
  }

  const attempted = commandSuccesses + commandFailures;
  const platformNames = new Set([...Object.keys(platformActive), ...Object.keys(counts.platformExits)]);
  const platforms = {};
  for (const platform of [...platformNames].sort()) {
    const platformAttempted = counts.platformAttempts[platform] || 0;
    const platformFailed = counts.platformFailures[platform] || 0;
    platforms[platform] = {
      active_monthly_pseudonyms: (platformActive[platform] || new Set()).size,
      events: counts.platformEvents[platform] || 0,
      command_observations: platformAttempted,
      failed_command_observations: platformFailed,
      observed_command_error_rate: platformAttempted ? Number((platformFailed / platformAttempted).toFixed(6)) : null,
      benefited_monthly_pseudonyms: (platformBenefited[platform] || new Set()).size,
      exit_feedback_responses: counts.platformExits[platform] || 0,
    };
  }
  const intersectionSize = (left, right) => [...left].filter((item) => right.has(item)).length;
  return {
    schema_version: "entroly.product-telemetry-summary.v3",
    window_days: days,
    since,
    privacy: {
      identifier: "monthly rotating pseudonym",
      unique_user_claim_allowed: false,
      ip_or_headers_stored: false,
      raw_events_returned: false,
      exact_tokens_or_costs_stored: false,
      model_identifiers_stored: false,
      usage_volume_claim_allowed: false,
      free_text_feedback_stored: false,
    },
    active_monthly_pseudonyms: active.size,
    activation_monthly_pseudonyms: activated.size,
    events: sortedObject(counts.events),
    commands: {
      observations: attempted,
      successful_observations: commandSuccesses,
      failed_observations: commandFailures,
      observed_error_rate: attempted ? Number((commandFailures / attempted).toFixed(6)) : null,
      observations_by_command: sortedObject(counts.commands),
      error_observations_by_command: sortedObject(counts.commandErrors),
      note: "Daily category observations are not command usage volume.",
    },
    benefit: {
      observations: optimizationObservations,
      positive_observations: positiveObservations,
      monthly_pseudonyms_with_positive_reduction: benefited.size,
      provider_bound_monthly_pseudonyms_with_positive_reduction: providerBenefited.size,
      monthly_pseudonyms_with_modeled_cost_reduction: modeledCost.size,
      observed_benefit_rate_among_active_pseudonyms: active.size ? Number((benefited.size / active.size).toFixed(6)) : null,
      token_savings_buckets: sortedObject(counts.tokenBuckets),
      reduction_percent_buckets: sortedObject(counts.reductionBuckets),
      measurement_scopes: sortedObject(counts.scopes),
      cost_evidence: sortedObject(counts.costs),
      money_savings_verified: false,
      note: "Modeled cost reduction is derived from provider-bound token reduction and configured rates; it is not a provider invoice.",
    },
    exit_feedback: {
      responses: Object.values(counts.exitReasons).reduce((total, value) => total + value, 0),
      monthly_or_one_event_pseudonyms: exits.size,
      reasons: sortedObject(counts.exitReasons),
      self_reported_benefit: sortedObject(counts.exitBenefits),
      primary_surfaces: sortedObject(counts.exitSurfaces),
      use_duration_buckets: sortedObject(counts.exitDurations),
      monthly_pseudonyms_with_prior_positive_reduction: intersectionSize(exits, benefited),
      monthly_pseudonyms_with_prior_error_observation: intersectionSize(exits, errors),
      note: "Structured opt-in responses only; direct package-manager uninstalls are not observable.",
    },
    error_types: sortedObject(counts.errorTypes),
    platforms,
    surfaces: sortedObject(counts.surfaces),
    versions: sortedObject(counts.versions),
    community_savings: communitySavings,
  };
}

async function summary(env, now, requestedDays) {
  const retention = boundedRetention(env);
  const parsed = Number.parseInt(requestedDays || "30", 10);
  if (!Number.isInteger(parsed)) return null;
  const days = Math.max(1, Math.min(parsed, retention));
  const since = dayString(addDays(now, -(days - 1)));
  await prune(env, now);
  const result = await env.DB.prepare(`
    SELECT occurred_on, installation_id, event_name, version,
           platform, python, properties_json
      FROM product_events
     WHERE occurred_on >= ?
  `).bind(since).all();
  return summarizeRows(result.results || [], days, since, await publicSavings(env, now));
}

async function handle(request, env) {
  const url = new URL(request.url);
  const now = new Date();

  if (request.method === "GET" && url.pathname === "/health") {
    return json({ status: "ok", storage: "cloudflare-d1" });
  }
  if (request.method === "GET" && url.pathname === "/v1/public-savings") {
    return json(await publicSavings(env, now), 200, corsHeaders(request, env));
  }
  if (request.method === "GET" && url.pathname === "/v1/summary") {
    if (!env.ADMIN_TOKEN || !await authorized(request, env.ADMIN_TOKEN)) return json({ error: "not_found" }, 404);
    const report = await summary(env, now, url.searchParams.get("days"));
    return report ? json(report) : json({ error: "invalid_days" }, 400);
  }
  if (request.method === "POST" && url.pathname === "/v1/events") {
    if (!await authorized(request, env.INGEST_TOKEN)) return json({ error: "unauthorized" }, 401);
    const body = await readJson(request, MAX_BATCH_BYTES);
    if (body.error) return json({ error: body.error }, body.error === "payload_too_large" ? 413 : 400);
    const events = validateBatch(body.value, now, boundedRetention(env));
    if (!events) return json({ error: "invalid_schema" }, 400);
    if (!await withinRateLimit(env, events.map((event) => event.installation_id))) {
      return json({ error: "rate_limited" }, 429, { "Retry-After": "60" });
    }
    await prune(env, now);
    const inserted = await ingest(env, events);
    return json({ accepted: events.length, inserted });
  }
  if (request.method === "DELETE" && url.pathname === "/v1/events") {
    if (!await authorized(request, env.INGEST_TOKEN)) return json({ error: "unauthorized" }, 401);
    const body = await readJson(request, 4096);
    if (body.error) return json({ error: body.error }, body.error === "payload_too_large" ? 413 : 400);
    const installationIds = validateDeletion(body.value);
    if (!installationIds) return json({ error: "invalid_schema" }, 400);
    const placeholders = installationIds.map(() => "?").join(", ");
    const results = await env.DB.batch([
      env.DB.prepare(`DELETE FROM savings_contributions WHERE installation_id IN (${placeholders})`).bind(...installationIds),
      env.DB.prepare(`DELETE FROM product_events WHERE installation_id IN (${placeholders})`).bind(...installationIds),
    ]);
    return json({ deleted: Math.max(0, Number(results[1]?.meta?.changes || 0)) });
  }
  return json({ error: "not_found" }, 404);
}

async function fetchHandler(request, env) {
  try {
    return await handle(request, env);
  } catch {
    return json(
      { error: "internal_error" },
      500,
      { "Cache-Control": "no-store" },
    );
  }
}

export default { fetch: fetchHandler };
export { fetchHandler, handle };
