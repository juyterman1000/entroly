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

// Untrusted text reaches a security dialog through two paths: `toolName` comes
// from the host's tool registry (an MCP server may name its own tools) and
// `retry_instruction` embeds verbatim recovered message text, which a prior
// tool call may have fetched from the open web. Either can carry newlines and
// forge a line the operator reads as ours -- "STATUS: VERIFIED, safe to
// approve" as the last line of a warning is worse than no warning. Collapse
// whitespace and cap length, exactly as safeDiagnostic already does for
// diagnostics.
function forDisplay(value, limit) {
  if (typeof value !== "string") return "";
  const flattened = value.replace(/\s+/g, " ").trim();
  return flattened.length > limit
    ? `${flattened.slice(0, limit - 1)}…`
    : flattened;
}

// One verdict, decided where the result is already validated, read everywhere.
// Three hooks previously re-derived "is this run's output unsupported" from
// raw fields with three different staleness rules; that is what let a fix on
// one guard leave its twin behind. A single record with an explicit run and
// output binding makes the divergence structurally impossible.
function buildVerdict({ runId, outputSha256, result, stale = false }) {
  return {
    runId: runId ?? null,
    outputSha256: outputSha256 ?? null,
    stale,
    unsupported: result?.status === "retry_with_exact_evidence",
    errored: result?.status === "verification_error",
    detail: forDisplay(result?.retry_instruction, 400),
  };
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
  if (
    config.gateToolCalls !== undefined &&
    config.gateToolCalls !== gateToolCalls
  ) {
    // Silently coercing "blcok" to "approve" leaves an operator believing they
    // withheld calls when they are only prompting for them.
    logger.warn?.(
      `entroly: ignoring unrecognised gateToolCalls ${JSON.stringify(
        config.gateToolCalls,
      )}; using "${gateToolCalls}"`,
    );
  }
  const approvalTimeoutMs = positiveInteger(
    config.gateToolCallsTimeoutMs,
    120000,
    3600000,
  );

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
      if (gateToolCalls === "off") return undefined;

      // Hooks disagree about where the session id lives -- onReplyPayloadSending
      // reads `usageState.sessionId` -- and the shape for this event was never
      // captured from a real host. Accept both rather than silently resolving
      // `undefined` and becoming a no-op that no test can see.
      const sessionId = event?.sessionId ?? event?.usageState?.sessionId;
      const state = sessionId ? proofStateBySession.get(sessionId) : undefined;
      const verdict = state?.verdict;
      if (!state || !verdict) return undefined;

      // The verdict must be provably about THIS run. A positive match is
      // required rather than skipping the check when `runId` is absent: proof
      // state outlives a run, because engine.js rebuilds it only when the
      // prompt changes, so a repeated prompt carries the previous verdict
      // forward. Failing closed on missing information is the opposite of what
      // the rest of this handler does.
      if (!event?.runId || verdict.runId !== event.runId) return undefined;

      // Saturating maxRounds stops verification but not the run, so later
      // outputs in the same run are never checked. Attributing a frozen verdict
      // to them would blame one response for another's tool call.
      if (verdict.stale) return undefined;

      // A verification failure must not leave the higher-consequence surface
      // open while the reply is withheld. `disabled` and `errored` mean the
      // check did not happen: say so rather than implying the call was cleared.
      const brokenVerifier = state.disabled || verdict.errored;
      if (!brokenVerifier && !verdict.unsupported) return undefined;
      if (brokenVerifier && gateToolCalls !== "block") return undefined;

      const toolName = forDisplay(event?.toolName, 60) || "this tool";
      // Sanitised again at the point of use, not only where the verdict is
      // built. The handler must not assume its input was cleaned by whoever
      // produced it: a verdict constructed on another path would otherwise
      // carry raw newlines straight into a security dialog.
      const detail = forDisplay(verdict.detail, 400);
      const record = (decision) => {
        // Withholding an action is the most consequential thing this plugin
        // does and was the only intervention leaving no trace. Receipt honesty
        // requires that an operator can reconstruct why afterwards.
        const previous = statusBySession.get(sessionId) ?? {};
        statusBySession.set(sessionId, {
          ...previous,
          tool_gate_decision: decision,
          tool_gate_tool: toolName,
          tool_gate_run_id: event.runId,
          tool_gate_count: (previous.tool_gate_count ?? 0) + 1,
          tool_gate_reason: brokenVerifier ? "verification_error" : "unsupported",
        });
      };

      if (brokenVerifier) {
        record("blocked");
        return {
          block: true,
          blockReason:
            "Entroly withheld this tool call: local verification did not " +
            "complete, so the claim behind it was never checked. Set " +
            'gateToolCalls to "approve" or "off" to allow unverified calls.',
        };
      }

      if (gateToolCalls === "approve") {
        // Recorded before returning, and keyed per run: the host tells us
        // nothing about the outcome, so without this a single verdict produces
        // one identical modal per tool call and trains reflexive approval.
        if (state.toolGateApprovedRunId === event.runId) return undefined;
        record("approval_requested");
        return {
          requireApproval: {
            title: `Entroly: unverified claim behind ${toolName}`,
            description:
              "The response that led to this tool call contained a claim the " +
              "supplied evidence did not support. Approve only if you have " +
              "checked it yourself." +
              (detail ? `

Recovered evidence (untrusted text): ${detail}` : ""),
            severity: "warning",
            // An unresolved approval always denies, and a headless or
            // unattended host resolves nothing. Bounding the wait turns a
            // silent hang into a denial the operator can see in the log.
            timeoutMs: approvalTimeoutMs,
            allowedDecisions: ["allow-once", "allow-always", "deny"],
            onResolution: (decision) => {
              if (decision === "allow-always") {
                state.toolGateApprovedRunId = event.runId;
              }
              record(`approval_${decision}`);
            },
          },
        };
      }

      record("blocked");
      return {
        block: true,
        blockReason:
          "Entroly withheld this tool call: the response that produced it " +
          "contained a claim the supplied evidence did not support" +
          (detail ? ` (recovered evidence, untrusted text: ${detail})` : "") +
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
        // Saturating maxRounds stops verification but not the run. Marking the
        // verdict stale here stops a frozen result from being attributed to
        // later, unverified outputs in the same run.
        state.verdict = buildVerdict({
          runId: event.runId,
          outputSha256: outputSha,
          result,
          stale: state.attempts >= maxRounds,
        });
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
        state.verdict = buildVerdict({
          runId: event?.runId,
          outputSha256: outputSha,
          result: { status: "verification_error" },
        });
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
        // Positive match required, matching onBeforeToolCall. The permissive
        // form let an event without a runId overwrite this run's reply with a
        // previous run's verified_output -- a wrong answer delivered as a
        // verified one.
        !event?.runId ||
        state.runId !== event.runId ||
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
