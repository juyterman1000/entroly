import { createHash, randomBytes } from "node:crypto";

import { ENTROLY_BRIDGE_SCHEMA } from "./bridge-client.js";

const PROVIDER_MODE = "openclaw_managed";

function positiveInteger(value) {
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : undefined;
}

function boundedString(value, limit = 256) {
  return typeof value === "string" && value.trim()
    ? value.trim().slice(0, limit)
    : null;
}

function safeDiagnostic(value, limit = 400) {
  let diagnostic = String(value?.message ?? value ?? "unknown error")
    .replace(/\s+/g, " ")
    .trim();
  diagnostic = diagnostic
    .replace(
      /\bAuthorization\s*:\s*(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+/gi,
      "Authorization: [REDACTED]",
    )
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer [REDACTED]")
    .replace(
      /\b(api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*["']?[^\s,;"']+/gi,
      "$1=[REDACTED]",
    )
    .replace(/\b(?:sk|gh[opusr])[-_][A-Za-z0-9_-]{8,}\b/gi, "[REDACTED]");
  return diagnostic.slice(0, limit) || "unknown error";
}

function statusSnapshot(status) {
  const snapshot = { ok: status?.ok === true };
  const numericFields = [
    "estimated_tokens",
    "source_tokens",
    "tokens_saved",
    "evidence_pinned",
    "evidence_pin_blocked",
    "context_window",
    "context_output_reserve",
    "context_safety_tokens",
    "proof_guided_attempts",
  ];
  const booleanFields = ["changed", "provider_independent"];
  const stringFields = [
    "schema_version",
    "receipt_id",
    "provider_mode",
    "budget_source",
    "context_discovery_status",
    "context_discovery_trust",
    "context_discovery_model",
    "model",
    "provider_hint",
    "assembly_strategy",
    "proof_guided_status",
    "proof_guided_audit_artifact_id",
    "error",
  ];
  for (const field of numericFields) {
    if (typeof status?.[field] === "number" && Number.isFinite(status[field])) {
      snapshot[field] = status[field];
    }
  }
  for (const field of booleanFields) {
    if (typeof status?.[field] === "boolean") snapshot[field] = status[field];
  }
  for (const field of stringFields) {
    if (status?.[field] === undefined || status?.[field] === null) continue;
    const value =
      field === "error"
        ? safeDiagnostic(status?.[field])
        : boundedString(status?.[field], field === "receipt_id" ? 160 : 256);
    if (value !== null) snapshot[field] = value;
  }
  if (Array.isArray(status?.warnings)) {
    snapshot.warnings = status.warnings
      .slice(0, 8)
      .map((warning) => safeDiagnostic(warning));
  }
  return snapshot;
}

function validateReceiptCommit(result, proposal) {
  if (
    !result ||
    result.ok !== true ||
    result.schema_version !== ENTROLY_BRIDGE_SCHEMA ||
    result.committed !== true ||
    result.receipt_id !== proposal.receiptId ||
    result.proposal_id !== proposal.proposalId ||
    result.proposal_sha256 !== proposal.proposalSha256 ||
    result.receipt_path !== proposal.receiptPath ||
    result.acceptance_commit_sha256 !== proposal.acceptanceCommitSha256
  ) {
    throw new Error("Entroly bridge did not acknowledge the validated receipt");
  }
}

function sha256(value) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function resolveAssemblyRuntime({ tokenBudget, model, runtimeSettings, fallbackTokenBudget }) {
  const explicitBudget = positiveInteger(tokenBudget);
  const runtimeBudget = positiveInteger(runtimeSettings?.limits?.promptTokenBudget);
  const fallbackCandidate = positiveInteger(fallbackTokenBudget);
  const configuredFallback =
    fallbackCandidate !== undefined && fallbackCandidate >= 1024
      ? fallbackCandidate
      : undefined;
  const requestedModel = boundedString(runtimeSettings?.model?.requested);
  const resolvedModel =
    boundedString(runtimeSettings?.model?.resolved) ?? requestedModel ?? boundedString(model);
  const runtimeMetadata = {
    schema_version: runtimeSettings?.schemaVersion === 1 ? 1 : null,
    runtime: {
      host: boundedString(runtimeSettings?.runtime?.host, 64),
      mode: boundedString(runtimeSettings?.runtime?.mode, 64),
      harness_id: boundedString(runtimeSettings?.runtime?.harnessId, 128),
      runtime_id: boundedString(runtimeSettings?.runtime?.runtimeId, 128),
    },
    model: {
      requested: requestedModel ?? boundedString(model),
      resolved: resolvedModel,
      provider: boundedString(runtimeSettings?.model?.provider, 128),
      family: boundedString(runtimeSettings?.model?.family, 128),
    },
    limits: {
      prompt_token_budget: runtimeBudget ?? null,
      max_output_tokens: positiveInteger(runtimeSettings?.limits?.maxOutputTokens) ?? null,
    },
    context_discovery: null,
  };

  if (explicitBudget !== undefined) {
    return {
      tokenBudget: explicitBudget,
      budgetSource: "openclaw_token_budget",
      model: resolvedModel,
      runtimeMetadata,
    };
  }
  if (runtimeBudget !== undefined) {
    return {
      tokenBudget: runtimeBudget,
      budgetSource: "openclaw_runtime_settings",
      model: resolvedModel,
      runtimeMetadata,
    };
  }
  if (configuredFallback !== undefined) {
    return {
      tokenBudget: configuredFallback,
      budgetSource: "operator_fallback",
      model: resolvedModel,
      runtimeMetadata,
    };
  }
  return {
    tokenBudget: undefined,
    budgetSource: "missing",
    model: resolvedModel,
    runtimeMetadata,
  };
}

function validateContextBudgetDiscovery(result) {
  if (!result || result.ok !== true || result.schema_version !== ENTROLY_BRIDGE_SCHEMA) {
    throw new Error("Entroly bridge returned an invalid context-discovery response");
  }
  if (result.status === "unavailable") {
    return {
      available: false,
      metadata: {
        status: "unavailable",
        trust: boundedString(result.trust, 32),
        model_id: boundedString(result.model, 256),
        exact: null,
        context_window: null,
        output_reserve_tokens: null,
        safety_tokens: null,
        registry_digest: boundedString(result.registry_digest, 64),
        source: null,
      },
      warning: safeDiagnostic(result.warning ?? "trusted model metadata was unavailable"),
    };
  }
  if (result.status !== "resolved" || result.budget_source !== "entroly_model_registry") {
    throw new Error("Entroly bridge returned an unknown context-discovery state");
  }
  const tokenBudget = positiveInteger(result.token_budget);
  const contextWindow = positiveInteger(result.context_window);
  const outputReserve = positiveInteger(result.output_reserve_tokens);
  const safetyTokens = positiveInteger(result.safety_tokens);
  const trust = boundedString(result.trust, 32);
  const modelId = boundedString(result.model_id, 256);
  const registryDigest = boundedString(result.registry_digest, 64);
  if (
    tokenBudget === undefined ||
    tokenBudget < 1024 ||
    contextWindow === undefined ||
    outputReserve === undefined ||
    safetyTokens === undefined ||
    !["verified", "user", "discovered"].includes(trust) ||
    modelId === null ||
    registryDigest === null ||
    !/^[0-9a-f]{64}$/.test(registryDigest) ||
    tokenBudget + outputReserve + safetyTokens > contextWindow
  ) {
    throw new Error("Entroly bridge returned unsafe context-discovery limits");
  }
  return {
    available: true,
    tokenBudget,
    metadata: {
      status: "resolved",
      trust,
      model_id: modelId,
      exact: result.exact === true,
      context_window: contextWindow,
      output_reserve_tokens: outputReserve,
      safety_tokens: safetyTokens,
      registry_digest: registryDigest,
      source: boundedString(result.source, 512),
    },
  };
}

function applyDiscoveredBudget(runtime, discovery) {
  return {
    ...runtime,
    tokenBudget: discovery.tokenBudget,
    budgetSource: "entroly_model_registry",
    runtimeMetadata: {
      ...runtime.runtimeMetadata,
      context_discovery: discovery.metadata,
    },
  };
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function isCompressibleTextBlock(block) {
  return Boolean(
    block &&
      typeof block === "object" &&
      !Array.isArray(block) &&
      block.type === "text" &&
      typeof block.text === "string" &&
      Object.keys(block).every((key) => key === "type" || key === "text"),
  );
}

function validateAssemblyResult(result, sourceMessages, preserveLastN) {
  if (
    !result ||
    result.ok !== true ||
    result.schema_version !== ENTROLY_BRIDGE_SCHEMA
  ) {
    throw new Error("Entroly bridge did not return a successful assembly result");
  }
  if (
    !Array.isArray(result.messages) ||
    result.messages.length !== sourceMessages.length ||
    !result.messages.every(
      (message) => message && typeof message === "object" && !Array.isArray(message),
    )
  ) {
    throw new Error("Entroly bridge returned an invalid or incomplete message list");
  }
  if (
    typeof result.estimated_tokens !== "number" ||
    !Number.isFinite(result.estimated_tokens) ||
    result.estimated_tokens < 0
  ) {
    throw new Error("Entroly bridge returned an invalid token estimate");
  }

  const recentStart = Math.max(0, sourceMessages.length - preserveLastN);
  for (let index = 0; index < sourceMessages.length; index += 1) {
    const source = sourceMessages[index];
    const assembled = result.messages[index];
    if (source.role !== assembled.role) {
      throw new Error(`Entroly bridge changed message role at index ${index}`);
    }

    const { content: sourceContent, ...sourceMetadata } = source;
    const { content: assembledContent, ...assembledMetadata } = assembled;
    if (canonicalJson(sourceMetadata) !== canonicalJson(assembledMetadata)) {
      throw new Error(`Entroly bridge changed message metadata at index ${index}`);
    }

    const exactMessage =
      source.role === "system" || source.role === "developer" || index >= recentStart;
    if (exactMessage) {
      if (canonicalJson(sourceContent) !== canonicalJson(assembledContent)) {
        throw new Error(`Entroly bridge changed protected message at index ${index}`);
      }
      continue;
    }

    if (typeof sourceContent === "string") {
      if (typeof assembledContent !== "string") {
        throw new Error(`Entroly bridge changed text content shape at index ${index}`);
      }
      continue;
    }
    if (!Array.isArray(sourceContent) || !Array.isArray(assembledContent)) {
      if (canonicalJson(sourceContent) !== canonicalJson(assembledContent)) {
        throw new Error(`Entroly bridge changed opaque content at index ${index}`);
      }
      continue;
    }
    if (sourceContent.length !== assembledContent.length) {
      throw new Error(`Entroly bridge changed content block count at index ${index}`);
    }
    for (let blockIndex = 0; blockIndex < sourceContent.length; blockIndex += 1) {
      const sourceBlock = sourceContent[blockIndex];
      const assembledBlock = assembledContent[blockIndex];
      if (isCompressibleTextBlock(sourceBlock)) {
        if (!isCompressibleTextBlock(assembledBlock)) {
          throw new Error(
            `Entroly bridge changed text block shape at index ${index}:${blockIndex}`,
          );
        }
      } else if (canonicalJson(sourceBlock) !== canonicalJson(assembledBlock)) {
        throw new Error(
          `Entroly bridge changed opaque content block at index ${index}:${blockIndex}`,
        );
      }
    }
  }
  return result;
}

function estimateTokens(messages) {
  return Math.ceil(JSON.stringify(messages).length / 4);
}

export function formatEntrolyStatus(status) {
  if (!status) {
    return "Entroly: no context assembly has completed for this session yet.";
  }
  if (!status.ok) {
    return [
      "Entroly: last assembly failed open to the original context.",
      "Provider routing remains OpenClaw-managed.",
      `Reason: ${safeDiagnostic(status.error)}`,
    ].join("\n");
  }
  const source = status.source_tokens ?? 0;
  const assembled = status.estimated_tokens ?? source;
  const saved = status.tokens_saved ?? Math.max(0, source - assembled);
  const reduction = source > 0 ? ((saved / source) * 100).toFixed(1) : "0.0";
  const lines = [
    "Entroly protected the last context assembly",
    "Provider routing: OpenClaw-managed (Entroly is provider-independent)",
    `Budget source: ${status.budget_source ?? "unknown"}`,
    `Strategy: ${status.assembly_strategy ?? "budgeted_context"}`,
    `Evidence pinned verbatim: ${status.evidence_pinned ?? 0} message(s)`,
    `Evidence pins blocked by firewall: ${status.evidence_pin_blocked ?? 0}`,
    `Estimated tokens: ${source.toLocaleString()} -> ${assembled.toLocaleString()}`,
    `Estimated reduction: ${reduction}% (${saved.toLocaleString()} tokens)`,
    `Changed: ${status.changed ? "yes" : "no"}`,
  ];
  if (status.context_discovery_status === "resolved") {
    lines.splice(
      3,
      0,
      `Context discovery: ${status.context_discovery_trust ?? "trusted"} metadata for ${status.context_discovery_model ?? "active model"}`,
      `Discovered window: ${(status.context_window ?? 0).toLocaleString()} tokens (${(status.context_output_reserve ?? 0).toLocaleString()} output reserve + ${(status.context_safety_tokens ?? 0).toLocaleString()} safety reserve)`,
    );
  }
  if (status.receipt_id) lines.push(`Receipt: ${boundedString(status.receipt_id, 160)}`);
  if (status.proof_guided_status) {
    lines.push(
      `Proof-guided recovery: ${boundedString(status.proof_guided_status, 160)}`,
      `Proof model attempts: ${Number(status.proof_guided_attempts ?? 0).toLocaleString()}`,
    );
  }
  if (status.proof_guided_audit_artifact_id) {
    lines.push(
      `Proof audit: ${boundedString(status.proof_guided_audit_artifact_id, 160)}`,
    );
  }
  if (status.warnings?.length) {
    lines.push(`Warnings: ${status.warnings.map((warning) => safeDiagnostic(warning)).join(" | ")}`);
  }
  return lines.join("\n");
}

export function formatEntrolyDoctor({ ok, error, pythonCommand = "python" }) {
  if (ok) {
    return [
      "Entroly doctor: ready",
      `Python command: ${pythonCommand}`,
      "Bridge: compatible (v2, two-phase receipts)",
      "Local-only context assembly: available",
      "Provider-neutral bridge: ready; routing and authentication remain OpenClaw-managed",
    ].join("\n");
  }
  const reason = safeDiagnostic(error);
  return [
    "Entroly doctor: not ready",
    `Python command: ${pythonCommand}`,
    `Reason: ${reason}`,
    "Fix: install Entroly into that Python environment with `python -m pip install -U entroly`,",
    "or set plugins.entries.entroly.config.pythonCommand to the correct Python executable.",
  ].join("\n");
}

export function createEntrolyContextEngine({
  bridge,
  delegateCompaction,
  buildMemoryPrompt = () => undefined,
  config = {},
  logger = console,
  statusBySession = new Map(),
  proofStateBySession = new Map(),
  maxStatusSessions = 512,
}) {
  if (typeof delegateCompaction !== "function") {
    throw new TypeError("Entroly requires OpenClaw's compaction delegate");
  }
  const statusLimit = positiveInteger(maxStatusSessions) ?? 512;
  const storeStatus = (sessionId, status) => {
    statusBySession.delete(sessionId);
    statusBySession.set(sessionId, statusSnapshot(status));
    while (statusBySession.size > statusLimit) {
      statusBySession.delete(statusBySession.keys().next().value);
    }
  };

  return {
    info: {
      id: "entroly",
      name: "Entroly Context Engine",
      ownsCompaction: false,
      hostRequirements: {
        "agent-run": {
          requiredCapabilities: ["assemble-before-prompt"],
          unsupportedMessage:
            "Entroly requires a native OpenClaw host that applies assembled context before each model call.",
        },
        "manual-compact": {
          requiredCapabilities: ["compact"],
          unsupportedMessage:
            "Entroly delegates transcript compaction to the native OpenClaw runtime.",
        },
      },
    },

    async ingest() {
      return { ingested: false };
    },

    async ingestBatch() {
      return { ingestedCount: 0 };
    },

    async assemble({
      sessionId,
      sessionKey,
      messages,
      tokenBudget,
      availableTools,
      citationsMode,
      model,
      prompt,
      runtimeSettings,
    }) {
      if (!Array.isArray(messages)) {
        const error = new TypeError(
          "OpenClaw did not supply context messages as an array; Entroly refused to assemble an empty prompt",
        );
        logger.error?.(`entroly: ${safeDiagnostic(error)}`);
        storeStatus(sessionId, { ok: false, error: error.message });
        throw error;
      }
      const sourceMessages = messages;
      let assemblyRuntime = resolveAssemblyRuntime({
        tokenBudget,
        model,
        runtimeSettings,
        fallbackTokenBudget: config.fallbackTokenBudget,
      });
      if (
        assemblyRuntime.tokenBudget === undefined &&
        config.autoDiscoverContextBudget !== false &&
        assemblyRuntime.model
      ) {
        try {
          const discovery = validateContextBudgetDiscovery(
            await bridge.request({
              operation: "resolve_context_budget",
              model: assemblyRuntime.model,
              provider_hint: assemblyRuntime.runtimeMetadata.model.provider,
              requested_output_tokens:
                assemblyRuntime.runtimeMetadata.limits.max_output_tokens,
            }),
          );
          if (discovery.available) {
            assemblyRuntime = applyDiscoveredBudget(assemblyRuntime, discovery);
          } else {
            assemblyRuntime.runtimeMetadata.context_discovery = discovery.metadata;
            logger.warn?.(`entroly: context auto-discovery unavailable: ${discovery.warning}`);
          }
        } catch (error) {
          logger.warn?.(
            `entroly: context auto-discovery failed safely: ${safeDiagnostic(error)}`,
          );
        }
      }
      const preserveLastN = positiveInteger(config.preserveLastN) ?? 4;
      let systemPromptAddition;
      try {
        systemPromptAddition = buildMemoryPrompt({
          availableTools: availableTools ?? new Set(),
          citationsMode,
          agentSessionKey: sessionKey,
        });
      } catch (error) {
        logger.warn?.(
          `entroly: OpenClaw memory guidance could not be added: ${safeDiagnostic(error)}`,
        );
      }
      const failOpen = (error) => {
        const reason = safeDiagnostic(error);
        logger.warn?.(`entroly: assembly failed; passing exact original context: ${reason}`);
        storeStatus(sessionId, {
          ok: false,
          error: reason,
          estimated_tokens: estimateTokens(sourceMessages),
          provider_mode: PROVIDER_MODE,
          provider_independent: true,
          budget_source: assemblyRuntime.budgetSource,
          context_discovery_status:
            assemblyRuntime.runtimeMetadata.context_discovery?.status,
          context_discovery_trust:
            assemblyRuntime.runtimeMetadata.context_discovery?.trust,
          context_discovery_model:
            assemblyRuntime.runtimeMetadata.context_discovery?.model_id,
          context_window:
            assemblyRuntime.runtimeMetadata.context_discovery?.context_window,
          context_output_reserve:
            assemblyRuntime.runtimeMetadata.context_discovery?.output_reserve_tokens,
          context_safety_tokens:
            assemblyRuntime.runtimeMetadata.context_discovery?.safety_tokens,
        });
        proofStateBySession.delete(sessionId);
        return {
          messages: sourceMessages,
          estimatedTokens: estimateTokens(sourceMessages),
          promptAuthority: "preassembly_may_overflow",
          systemPromptAddition,
        };
      };
      if (assemblyRuntime.tokenBudget === undefined) {
        return failOpen(
          new Error(
            "OpenClaw did not provide a positive prompt token budget and Entroly could not resolve trusted model metadata; configure the model context window or plugins.entries.entroly.config.fallbackTokenBudget",
          ),
        );
      }
      try {
        const receiptCommitToken =
          config.writeReceipts === false ? null : randomBytes(32).toString("hex");
        const result = validateAssemblyResult(
          await bridge.request({
            operation: "assemble",
            session_id: sessionId,
            messages: sourceMessages,
            token_budget: assemblyRuntime.tokenBudget,
            budget_source: assemblyRuntime.budgetSource,
            model: assemblyRuntime.model,
            openclaw_runtime: assemblyRuntime.runtimeMetadata,
            prompt,
            workspace_dir: config.workspaceDir,
            preserve_last_n: preserveLastN,
            receipt_dir: config.receiptDir,
            write_receipt: config.writeReceipts !== false,
            receipt_commit_challenge_sha256:
              receiptCommitToken === null ? undefined : sha256(receiptCommitToken),
            receipt_max_files: positiveInteger(config.receiptMaxFiles),
            receipt_max_bytes: positiveInteger(config.receiptMaxBytes),
            distill: config.distill !== false,
            evidence_pinning: config.evidencePinning !== false,
          }),
          sourceMessages,
          preserveLastN,
        );
        if (result.receipt_commit_required === true) {
          const proposal = {
            receiptId: boundedString(result.receipt_id, 160),
            proposalId: boundedString(result.proposal_id, 160),
            proposalSha256: boundedString(result.proposal_sha256, 64),
            receiptPath: boundedString(result.receipt_path, 1024),
            acceptanceCommitSha256:
              receiptCommitToken === null
                ? null
                : sha256(
                    `entroly.openclaw.accept.v1:${result.proposal_sha256}:${receiptCommitToken}`,
                  ),
          };
          if (
            !/^ocr_[0-9a-f]{20}$/.test(proposal.receiptId ?? "") ||
            !/^ocp_[0-9a-f]{32}$/.test(proposal.proposalId ?? "") ||
            !/^[0-9a-f]{64}$/.test(proposal.proposalSha256 ?? "") ||
            !proposal.receiptPath
          ) {
            throw new Error("Entroly bridge returned an invalid receipt proposal");
          }
          validateReceiptCommit(
            await bridge.request({
              operation: "commit_receipt",
              receipt_id: proposal.receiptId,
              proposal_id: proposal.proposalId,
              proposal_sha256: proposal.proposalSha256,
              receipt_path: proposal.receiptPath,
              receipt_commit_token: receiptCommitToken,
              workspace_dir: config.workspaceDir,
            }),
            proposal,
          );
        } else if (result.receipt_path) {
          throw new Error(
            "Entroly bridge wrote a receipt without the required acceptance handshake",
          );
        }
        if (config.proofGuidedRecovery === true) {
          const existing = proofStateBySession.get(sessionId);
          if (!existing || existing.prompt !== prompt || existing.disabled) {
            proofStateBySession.set(sessionId, {
              prompt,
              workspaceDir: config.workspaceDir,
              sourceMessages: structuredClone(sourceMessages),
              assembledMessages: structuredClone(result.messages),
              recoveredMessages: [],
              attempts: 0,
              runId: null,
              lastOutputSha256: null,
              lastProofResult: null,
              retryIssued: false,
              disabled: false,
            });
          }
          while (proofStateBySession.size > statusLimit) {
            proofStateBySession.delete(proofStateBySession.keys().next().value);
          }
        } else {
          proofStateBySession.delete(sessionId);
        }
        storeStatus(sessionId, {
          ...result,
          proof_guided_status:
            config.proofGuidedRecovery === true ? "armed" : "disabled",
          proof_guided_attempts:
            proofStateBySession.get(sessionId)?.attempts ?? 0,
        });
        return {
          messages: result.messages,
          estimatedTokens: result.estimated_tokens,
          promptAuthority:
            result.estimated_tokens > assemblyRuntime.tokenBudget
              ? "preassembly_may_overflow"
              : "assembled",
          systemPromptAddition,
        };
      } catch (error) {
        return failOpen(error);
      }
    },

    async compact(params) {
      return await delegateCompaction(params);
    },

    getStatus(sessionId) {
      return statusBySession.get(sessionId);
    },

    async dispose() {
      await bridge.dispose();
    },
  };
}
