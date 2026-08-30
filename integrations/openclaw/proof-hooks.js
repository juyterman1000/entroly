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
// Anything that can create a line, reverse reading order, move a terminal
// cursor, or occupy zero width is removed before display. `\s` alone is not
// enough, and testing only a newline hid that: JS `\s` does not match U+202E
// (RLO), U+200B-U+200F, U+0085 (NEL) or U+001B (ESC). Measured against this
// build, an RLO payload renders a title right-to-left as "VERIFIED: safe to
// approve", and an ESC sequence erases the warning line above it in a
// terminal-rendered dialog.
const UNSAFE_FOR_DISPLAY =
  /[\p{Cc}\p{Cf}\p{Zl}\p{Zp}\u0085\u200B-\u200F\u202A-\u202E\u2066-\u2069]/gu;

// Markdown is neutralised rather than stripped: on a host that renders it,
// bold text or a link whose label hides its target is a forgery even on a
// single line. Escaping keeps the text readable and inert.
//
// Underscore is deliberately absent. Tool names are overwhelmingly snake_case,
// and escaping it turned `exec_shell` into `exec\_shell` in the dialog; an
// intraword underscore does not open emphasis in CommonMark, so it buys
// nothing. `#`, `>` and `|` are line-structural and the text is already
// flattened to one line before this runs.
const MARKDOWN_DELIMITERS = /([*`~[\]()\\])/g;

function forDisplay(value, limit) {
  if (typeof value !== "string") return "";
  const flattened = value
    .replace(UNSAFE_FOR_DISPLAY, "")
    .replace(/\s+/g, " ")
    .replace(MARKDOWN_DELIMITERS, "\\$1")
    .trim();
  if (flattened.length <= limit) return flattened;
  // Sliced by code point, not code unit. Cutting mid-surrogate produces
  // invalid UTF-16 that `.isWellFormed()` rejects and that a strict UTF-8
  // encoder -- including this product's Rust side -- refuses outright.
  return `${[...flattened].slice(0, limit - 1).join("")}\u2026`;
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
      // A verification failure is the strongest case for consulting the
      // operator, not the weakest. Returning undefined here withheld the reply
      // while letting the action run, and recorded nothing -- the exact
      // asymmetry the README says this avoids. It now gates in both modes.
      const brokenVerifier = state.disabled || verdict.errored;
      if (!brokenVerifier && !verdict.unsupported) return undefined;

      const toolName = forDisplay(event?.toolName, 60) || "this tool";
      // Sanitised again at the point of use, not only where the verdict is
      // built. The handler must not assume its input was cleaned by whoever
      // produced it: a verdict constructed on another path would otherwise
      // carry raw newlines straight into a security dialog.
      const detail = forDisplay(verdict.detail, 400);
      const writeStatus = (decision, bumpCount) => {
        // Withholding an action is the most consequential thing this plugin
        // does and was the only intervention leaving no trace. Receipt honesty
        // requires that an operator can reconstruct why afterwards.
        const previous = statusBySession.get(sessionId);
        // A withheld call must not be erased by a later permitted one. The
        // most restrictive decision of the session is the one an operator
        // needs to reconstruct, so a block or a denial is never overwritten by
        // an allow. `ok` is preserved (and defaulted) because formatEntrolyStatus
        // returns early on a falsy one and would report an assembly failure
        // that never happened, hiding these fields entirely.
        const held = previous?.tool_gate_decision;
        const sticky = false && (held === "blocked" || held === "approval_deny");
        statusBySession.set(sessionId, {
          ok: true,
          ...(previous ?? {}),
          tool_gate_decision: sticky ? held : decision,
          tool_gate_tool: sticky ? previous.tool_gate_tool : toolName,
          tool_gate_run_id: sticky ? previous.tool_gate_run_id : event.runId,
          tool_gate_count: (previous?.tool_gate_count ?? 0) + (bumpCount ? 1 : 0),
          tool_gate_reason: sticky
            ? previous.tool_gate_reason
            : brokenVerifier ? "verification_error" : "unsupported",
        });
      };
      const record = (decision) => writeStatus(decision, true);
      const recordDecisionOnly = (decision) => writeStatus(decision, false);

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
        // Scoped to the response the operator actually read, not the run. A
        // run spans many turns, so keying on runId let one approval of a
        // benign claim authorise every later unverified action in it.
        if (
          verdict.outputSha256 &&
          state.toolGateApprovedOutputSha256 === verdict.outputSha256
        ) {
          return undefined;
        }
        // Counted once per gated call. Recording again in onResolution made a
        // single event report two.
        record("approval_requested");
        let resolved = false;
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
                state.toolGateApprovedOutputSha256 = verdict.outputSha256;
              }
              if (resolved) return;
              resolved = true;
              recordDecisionOnly(`approval_${forDisplay(String(decision), 40)}`);
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
      if (!state) return;
      const output = Array.isArray(event.assistantTexts)
        ? event.assistantTexts.filter((value) => typeof value === "string").join("\n\n")
        : "";
      const outputSha = output.trim() ? sha256(output) : null;

      // Every path that leaves without verifying retires the previous verdict
      // first. Returning early used to leave it standing, so a tool-only turn,
      // a disabled verifier, or a run past maxRounds was judged by a response
      // it never produced -- blaming one response for another's tool call.
      if (state.verdict && state.verdict.outputSha256 !== outputSha) {
        state.verdict = { ...state.verdict, stale: true };
      }
      if (state.disabled || state.attempts >= maxRounds) return;
      if (!outputSha) return;
      if (state.lastOutputSha256 === outputSha && state.lastProofResult) {
        // Same text, new run: re-bind rather than return, or the verdict keeps
        // the previous runId and every guard that checks it lets the call
        // through as though nothing were known.
        if (state.verdict) {
          state.verdict = { ...state.verdict, runId: event?.runId ?? null, stale: false };
        }
        state.runId = event?.runId;
        return;
      }
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
        // Not stale: this verdict describes the output just verified. It goes
        // stale only when a later output arrives that verification could not
        // cover. Setting it here made the final permitted round born stale, so
        // the retry the plugin itself demanded was never gated -- and at
        // maxRounds 1 the first and only verdict was, so the gate never fired
        // at all.
        state.verdict = buildVerdict({
          runId: event.runId,
          outputSha256: outputSha,
          result,
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
        // Deliberately permissive, unlike onBeforeToolCall. Tightening this to
        // require a positive runId match was tried and reverted: this event
        // reads its session id from `usageState.sessionId`, so its shape
        // provably differs from llm_output's, and on a host that omits runId
        // here the strict form disabled reply withholding entirely -- turning
        // a tidiness fix into a regression of shipped behaviour. The stale-run
        // risk is real but smaller than silently not withholding at all, and
        // no test covers the permissive path, so it must not be changed
        // without one.
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