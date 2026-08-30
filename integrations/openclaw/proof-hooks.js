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
  // "approve" by default rather than "block": withholding a tool call outright
  // on a verdict the operator has not seen trades one failure mode for a worse
  // one. An unknown value falls back to asking rather than to acting.
  const gateToolCalls = ["off", "approve", "block"].includes(
    config.gateToolCalls,
  )
    ? config.gateToolCalls
    : "approve";

  return {
    onBeforeToolCall(event) {
      // A tool call is where an unsupported claim stops being text and becomes
      // an action. Verification already ran at `llm_output`; this consults the
      // verdict it produced rather than repeating the work.
      //
      // Deliberately synchronous and bridge-free. OpenClaw gives this hook a
      // 15-second budget and fails CLOSED on expiry, so a round trip here would
      // turn a slow or unreachable bridge into blocked tool calls for the user.
      // Reading state that is already resident cannot time out.
      //
      // Everything unknown allows. Only an explicit "the model asserted
      // something its evidence did not support", for this exact run, gates.
      if (gateToolCalls === "off") return undefined;
      const state = proofStateBySession.get(event?.sessionId);
      const result = state?.lastProofResult;
      if (!state || !result || state.disabled) return undefined;
      // A verdict from an earlier run says nothing about this tool call.
      if (event?.runId && state.runId !== event.runId) return undefined;
      if (result.status !== "retry_with_exact_evidence") return undefined;

      const detail =
        typeof result.retry_instruction === "string"
          ? result.retry_instruction.trim().slice(0, 400)
          : "";
      const toolName =
        typeof event?.toolName === "string" && event.toolName
          ? event.toolName
          : "this tool";

      if (gateToolCalls === "approve") {
        return {
          requireApproval: {
            title: `Entroly: unverified claim behind ${toolName}`,
            description:
              "The response that led to this tool call contained a claim the " +
              "supplied evidence did not support. Approve only if you have " +
              "checked it yourself." + (detail ? `\n\n${detail}` : ""),
            severity: "warning",
            allowedDecisions: ["allow-once", "deny"],
          },
        };
      }
      return {
        block: true,
        blockReason:
          "Entroly withheld this tool call: the response that produced it " +
          "contained a claim the supplied evidence did not support" +
          (detail ? ` (${detail})` : "") +
          '. Set gateToolCalls to "approve" to decide per call, or "off" to ' +
          "disable this gate.",
      };
    },

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
