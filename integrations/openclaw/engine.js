const DEFAULT_TOKEN_BUDGET = 50_000;

function estimateTokens(messages) {
  return Math.ceil(JSON.stringify(messages).length / 4);
}

export function formatEntrolyStatus(status) {
  if (!status) {
    return "Entroly: no context assembly has completed for this session yet.";
  }
  if (!status.ok) {
    return `Entroly: last assembly failed open to the original context.\nReason: ${status.error}`;
  }
  const source = status.source_tokens ?? 0;
  const assembled = status.estimated_tokens ?? source;
  const saved = status.tokens_saved ?? Math.max(0, source - assembled);
  const reduction = source > 0 ? ((saved / source) * 100).toFixed(1) : "0.0";
  const lines = [
    "Entroly protected the last context assembly",
    `Strategy: ${status.assembly_strategy ?? "budgeted_context"}`,
    `Evidence pinned verbatim: ${status.evidence_pinned ?? 0} message(s)`,
    `Estimated tokens: ${source.toLocaleString()} -> ${assembled.toLocaleString()}`,
    `Estimated reduction: ${reduction}% (${saved.toLocaleString()} tokens)`,
    `Changed: ${status.changed ? "yes" : "no"}`,
  ];
  if (status.receipt_id) lines.push(`Receipt: ${status.receipt_id}`);
  if (status.warnings?.length) lines.push(`Warnings: ${status.warnings.join(" | ")}`);
  return lines.join("\n");
}

export function createEntrolyContextEngine({
  bridge,
  config = {},
  logger = console,
  statusBySession = new Map(),
}) {

  return {
    info: {
      id: "entroly",
      name: "Entroly Context Engine",
      ownsCompaction: false,
    },

    async ingest() {
      return { ingested: false };
    },

    async ingestBatch() {
      return { ingestedCount: 0 };
    },

    async assemble({
      sessionId,
      messages,
      tokenBudget,
      model,
      prompt,
      runtimeSettings,
    }) {
      const sourceMessages = Array.isArray(messages) ? messages : [];
      const effectiveBudget =
        tokenBudget ?? runtimeSettings?.limits?.promptTokenBudget ?? DEFAULT_TOKEN_BUDGET;
      try {
        const result = await bridge.request({
          operation: "assemble",
          session_id: sessionId,
          messages: sourceMessages,
          token_budget: effectiveBudget,
          model,
          prompt,
          workspace_dir: config.workspaceDir,
          preserve_last_n: config.preserveLastN ?? 4,
          receipt_dir: config.receiptDir,
          write_receipt: config.writeReceipts !== false,
          distill: config.distill !== false,
          evidence_pinning: config.evidencePinning !== false,
        });
        statusBySession.set(sessionId, result);
        return {
          messages: result.messages,
          estimatedTokens: result.estimated_tokens,
          promptAuthority: "assembled",
        };
      } catch (error) {
        logger.warn?.(
          `entroly: assembly failed; passing exact original context: ${error.message}`,
        );
        statusBySession.set(sessionId, {
          ok: false,
          error: error.message,
          estimated_tokens: estimateTokens(sourceMessages),
        });
        return {
          messages: sourceMessages,
          estimatedTokens: estimateTokens(sourceMessages),
          promptAuthority: "preassembly_may_overflow",
        };
      }
    },

    async compact({ currentTokenCount }) {
      return {
        ok: true,
        compacted: false,
        reason:
          "Entroly applies reversible per-turn context assembly and does not rewrite the OpenClaw transcript.",
        result: { tokensBefore: currentTokenCount ?? 0 },
      };
    },

    getStatus(sessionId) {
      return statusBySession.get(sessionId);
    },

    async dispose() {
      await bridge.dispose();
    },
  };
}
