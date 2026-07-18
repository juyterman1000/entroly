import { createHash } from "node:crypto";

import { ENTROLY_BRIDGE_SCHEMA } from "./bridge-client.js";

function sha256(value) {
  return createHash("sha256").update(String(value), "utf8").digest("hex");
}

function safeDiagnostic(value, limit = 400) {
  return String(value?.message ?? value ?? "unknown error")
    .replace(/\s+/g, " ")
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer [REDACTED]")
    .replace(
      /\b(api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*["']?[^\s,;"']+/gi,
      "$1=[REDACTED]",
    )
    .slice(0, limit);
}

function positiveInteger(value, fallback, maximum) {
  return typeof value === "number" &&
    Number.isInteger(value) &&
    value > 0 &&
    value <= maximum
    ? value
    : fallback;
}

function validateProofResult(result) {
  if (
    !result ||
    result.ok !== true ||
    result.schema_version !== ENTROLY_BRIDGE_SCHEMA ||
    result.provider_call_performed !== false ||
    result.local_only !== true ||
    typeof result.status !== "string" ||
    typeof result.verified_output !== "string" ||
    !Array.isArray(result.recovered_messages) ||
    typeof result.audit_artifact_id !== "string"
  ) {
    throw new Error("Entroly bridge returned an invalid proof-guided result");
  }
  if (
    result.status === "retry_with_exact_evidence" &&
    (typeof result.retry_instruction !== "string" || !result.retry_instruction.trim())
  ) {
    throw new Error("Entroly bridge omitted the exact-evidence retry instruction");
  }
  return result;
}

function mergeRecoveredMessages(existing, recovered) {
  const seen = new Set(existing.map((message) => sha256(JSON.stringify(message))));
  const merged = [...existing];
  for (const message of recovered) {
    const fingerprint = sha256(JSON.stringify(message));
    if (seen.has(fingerprint)) continue;
    seen.add(fingerprint);
    merged.push(message);
  }
  return merged;
}

export function createProofGuidedHooks({
  bridge,
  config = {},
  logger = console,
  proofStateBySession = new Map(),
  statusBySession = new Map(),
}) {
  const maxRounds = positiveInteger(config.proofGuidedMaxRounds, 2, 4);
  const recoveryTokenBudget = positiveInteger(
    config.proofGuidedRecoveryTokens,
    1200,
    100000,
  );
  const maxRecoveryMessages = positiveInteger(
    config.proofGuidedMaxMessages,
    3,
    16,
  );

  return {
    async onLlmOutput(event) {
      const state = proofStateBySession.get(event?.sessionId);
      if (!state || state.disabled || state.attempts >= maxRounds) return;
      const output = Array.isArray(event.assistantTexts)
        ? event.assistantTexts.filter((value) => typeof value === "string").join("\n\n")
        : "";
      if (!output.trim()) return;
      const outputSha = sha256(output);
      if (state.lastOutputSha256 === outputSha && state.lastProofResult) return;
      try {
        const result = validateProofResult(
          await bridge.request({
            operation: "verify_proof_guided_output",
            session_id: event.sessionId,
            run_id: event.runId,
            round_index: state.attempts,
            source_messages: state.sourceMessages,
            assembled_messages: state.assembledMessages,
            recovered_messages: state.recoveredMessages,
            model_output: output,
            workspace_dir: state.workspaceDir,
            profile: config.proofGuidedProfile ?? "rag",
            recovery_token_budget: recoveryTokenBudget,
            max_recovery_messages: maxRecoveryMessages,
          }),
        );
        state.attempts += 1;
        state.runId = event.runId;
        state.lastOutputSha256 = outputSha;
        state.lastProofResult = result;
        state.recoveredMessages = mergeRecoveredMessages(
          state.recoveredMessages,
          result.recovered_messages,
        );
        state.retryIssued = false;
        statusBySession.set(event.sessionId, {
          ...(statusBySession.get(event.sessionId) ?? {}),
          proof_guided_status: result.status,
          proof_guided_attempts: state.attempts,
          proof_guided_audit_artifact_id: result.audit_artifact_id,
        });
      } catch (error) {
        state.disabled = true;
        state.error = safeDiagnostic(error);
        state.runId = event?.runId;
        state.lastProofResult = {
          status: "verification_error",
          verified_output:
            "Entroly withheld this response because local proof verification failed. " +
            "Review the OpenClaw plugin log and retry after resolving the reported error.",
          changed: true,
          recovered_messages: [],
          audit_artifact_id: "",
        };
        statusBySession.set(event.sessionId, {
          ...(statusBySession.get(event.sessionId) ?? {}),
          proof_guided_status: "verification_error",
          proof_guided_attempts: state.attempts,
          error: state.error,
        });
        logger.warn?.(
          `entroly: proof-guided verification failed; delivery will be withheld: ${state.error}`,
        );
      }
    },

    async onBeforeAgentFinalize(event) {
      const state = proofStateBySession.get(event?.sessionId);
      const result = state?.lastProofResult;
      if (
        !state ||
        !result ||
        state.runId !== event?.runId ||
        result.status !== "retry_with_exact_evidence" ||
        state.attempts >= maxRounds ||
        state.retryIssued
      ) {
        return;
      }
      state.retryIssued = true;
      return {
        action: "revise",
        reason: "Entroly found unsupported claims and recovered exact omitted evidence.",
        retry: {
          instruction: result.retry_instruction,
          idempotencyKey: `entroly-proof-${sha256(
            `${event.runId}:${result.audit_artifact_id}:${state.attempts}`,
          ).slice(0, 24)}`,
          maxAttempts: 1,
        },
      };
    },

    async onReplyPayloadSending(event) {
      const sessionId = event?.usageState?.sessionId;
      const state = sessionId ? proofStateBySession.get(sessionId) : undefined;
      const result = state?.lastProofResult;
      if (
        !state ||
        !result ||
        (event?.runId && state.runId !== event.runId) ||
        result.changed !== true ||
        typeof event?.payload?.text !== "string"
      ) {
        return;
      }
      return {
        payload: {
          ...event.payload,
          text: result.verified_output,
        },
      };
    },
  };
}
