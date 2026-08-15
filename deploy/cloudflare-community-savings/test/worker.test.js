import assert from "node:assert/strict";
import test from "node:test";

import {
  fetchHandler,
  handle,
  publicSavingsPayload,
  summarizeRows,
  validateBatch,
  validateDeletion,
  validateEvent,
} from "../src/index.js";

const NOW = new Date("2026-08-14T12:00:00Z");

function event(overrides = {}) {
  return {
    schema_version: "entroly.product-telemetry.v3",
    event_id: "a".repeat(32),
    occurred_on: "2026-08-14",
    installation_id: "b".repeat(24),
    event_name: "savings_contribution",
    version: "1.0.77",
    platform: "windows",
    python: "3.13",
    properties: {
      tokens_saved_thousands: "12",
      modeled_cost_saved_cents: "12",
    },
    ...overrides,
  };
}

test("wire validation accepts the Python v3 savings contract", () => {
  const sample = event();
  assert.equal(validateEvent(sample, NOW, 90), true);
  assert.deepEqual(validateBatch({
    schema_version: "entroly.product-telemetry-batch.v3",
    events: [sample],
  }, NOW, 90), [sample]);
});

test("wire validation rejects content, exact values, invalid dates, and zero deltas", () => {
  assert.equal(validateEvent(event({ properties: {
    tokens_saved_thousands: "12",
    modeled_cost_saved_cents: "12",
    prompt: "private",
  } }), NOW, 90), false);
  assert.equal(validateEvent(event({ properties: {
    tokens_saved_thousands: 12,
    modeled_cost_saved_cents: "12",
  } }), NOW, 90), false);
  assert.equal(validateEvent(event({ occurred_on: "2026-02-31" }), NOW, 90), false);
  assert.equal(validateEvent(event({ properties: {
    tokens_saved_thousands: "0",
    modeled_cost_saved_cents: "0",
  } }), NOW, 90), false);
});

test("legacy batches cannot smuggle the v3 savings event", () => {
  const legacy = event({ schema_version: "entroly.product-telemetry.v2" });
  assert.equal(validateBatch({
    schema_version: "entroly.product-telemetry-batch.v2",
    events: [legacy],
  }, NOW, 90), null);
});

test("deletion accepts only bounded monthly pseudonyms", () => {
  assert.deepEqual(validateDeletion({
    schema_version: "entroly.product-telemetry-delete.v1",
    installation_ids: ["b".repeat(24), "b".repeat(24)],
  }), ["b".repeat(24)]);
  assert.equal(validateDeletion({
    schema_version: "entroly.product-telemetry-delete.v1",
    installation_ids: ["not-an-id"],
  }), null);
});

test("public payload exposes only conservative totals and fixed evidence", () => {
  const payload = publicSavingsPayload({ token_units: 12, cent_units: 12 }, NOW);
  assert.equal(payload.reported_provider_tokens_saved, 12_000);
  assert.equal(payload.reported_modeled_input_cost_avoided_usd, 0.12);
  assert.equal(JSON.stringify(payload).includes("installation"), false);
  assert.equal(payload.privacy.identifiers_exposed, false);
});

test("private summary matches adoption, error, benefit, and exit semantics", () => {
  const installation = "b".repeat(24);
  const rows = [
    ["activation", { surface: "cli" }],
    ["command", { command: "doctor", result: "success", duration_bucket: "lt_1s" }],
    ["command", { command: "proxy", result: "error", duration_bucket: "lt_1s", error_type: "ValueError" }],
    ["optimization_outcome", {
      surface: "proxy", measurement_scope: "provider_bound_estimate",
      tokens_saved_bucket: "1k_9k", reduction_percent_bucket: "70_89",
      cost_evidence: "modeled_positive",
    }],
    ["exit_feedback", {
      reason: "runtime_error", benefit_outcome: "yes", primary_surface: "proxy",
      use_duration_bucket: "1_7d",
    }],
  ].map(([eventName, properties]) => ({
    occurred_on: "2026-08-14",
    installation_id: installation,
    event_name: eventName,
    version: "1.0.77",
    platform: "windows",
    python: "3.13",
    properties_json: JSON.stringify(properties),
  }));
  const report = summarizeRows(rows, 30, "2026-07-16", publicSavingsPayload({}, NOW));
  assert.equal(report.active_monthly_pseudonyms, 1);
  assert.equal(report.commands.observed_error_rate, 0.5);
  assert.equal(report.benefit.monthly_pseudonyms_with_positive_reduction, 1);
  assert.equal(report.exit_feedback.reasons.runtime_error, 1);
  assert.equal(report.platforms.windows.failed_command_observations, 1);
  assert.equal(JSON.stringify(report).includes(installation), false);
});

test("health route is storage-specific and does not require D1", async () => {
  const response = await handle(new Request("https://example.com/health"), {});
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { status: "ok", storage: "cloudflare-d1" });
});

test("runtime failures return a content-blind structured error", async () => {
  const response = await fetchHandler(
    new Request("https://example.com/v1/public-savings"),
    {},
  );
  assert.equal(response.status, 500);
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.deepEqual(await response.json(), { error: "internal_error" });
});

test("bounded reader rejects an oversized streaming body before JSON parsing", async () => {
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array(40 * 1024));
      controller.enqueue(new Uint8Array(40 * 1024));
      controller.close();
    },
  });
  const response = await handle(
    new Request("https://example.com/v1/events", { method: "POST", body, duplex: "half" }),
    {},
  );
  assert.equal(response.status, 413);
  assert.deepEqual(await response.json(), { error: "payload_too_large" });
});
