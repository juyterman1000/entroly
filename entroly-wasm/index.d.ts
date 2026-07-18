export * from "./pkg/entroly_wasm";

export type EntrolyProvider = "openai" | "anthropic" | "gemini";

export interface EntrolyAppSdkOptions {
  budget?: number;
  preserveLastN?: number;
  provider?: EntrolyProvider;
}

export interface EntrolyAppDecision {
  provider: EntrolyProvider;
  action: "compress" | "observe";
  tokensBefore: number;
  tokensAfter: number;
  requestFingerprint: string;
}

export interface EntrolyOptimizedParams<T = Record<string, unknown>> {
  params: T;
  decision: EntrolyAppDecision;
}

export function estimateTokens(value: unknown): number;
export function stableStringify(value: unknown): string;
export function optimizeMessages<T extends Array<Record<string, unknown>>>(
  messages: T,
  options?: EntrolyAppSdkOptions,
): { messages: T; tokensBefore: number; tokensAfter: number };
export function optimizeRequestParams<T extends Record<string, unknown>>(
  params: T,
  options?: EntrolyAppSdkOptions,
): EntrolyOptimizedParams<T>;
export function optimizeOpenAIParams<T extends Record<string, unknown>>(
  params: T,
  options?: EntrolyAppSdkOptions,
): EntrolyOptimizedParams<T>;
export function optimizeAnthropicParams<T extends Record<string, unknown>>(
  params: T,
  options?: EntrolyAppSdkOptions,
): EntrolyOptimizedParams<T>;
export function optimizeGeminiParams<T extends Record<string, unknown>>(
  params: T,
  options?: EntrolyAppSdkOptions,
): EntrolyOptimizedParams<T>;

export function createEntrolyMiddleware(options?: EntrolyAppSdkOptions): {
  specificationVersion: "v3";
  transformParams(args: { params: Record<string, unknown> }): Promise<Record<string, unknown>>;
  wrapGenerate(args: { doGenerate: () => Promise<unknown> }): Promise<unknown>;
  wrapStream(args: { doStream: () => Promise<unknown> }): Promise<unknown>;
};

export function wrapOpenAI<TClient>(client: TClient, options?: EntrolyAppSdkOptions): unknown;
export function wrapAnthropic<TClient>(client: TClient, options?: EntrolyAppSdkOptions): unknown;
export function wrapGemini<TClient>(client: TClient, options?: EntrolyAppSdkOptions): unknown;

export interface EntrolyValueReceipt {
  schema_version: "entroly.value-receipt.v1";
  provider_path: Record<string, number | string>;
  local_operations: Record<string, number | string>;
  legacy_unclassified: Record<string, number | string>;
  trust_signals: Record<string, number>;
  pricing: { source: string; as_of: string };
  generated_at_unix: number;
}

export class ValueTracker {
  constructor(dataDir?: string | null);
  record(input?: {
    tokensSaved?: number;
    model?: string;
    duplicates?: number;
    optimized?: boolean;
    source?: "provider" | "proxy" | "gateway" | "sdk" | "npm" | "mcp" | "local" | string;
  }): { tokensSaved: number; costSaved: number };
  getTrends(): Record<string, unknown>;
  getValueReceipt(): EntrolyValueReceipt;
}

export const EVOLUTION_TAX_RATE: number;
export function estimateCost(tokens: number, model?: string): number;

export interface ContextReceiptDocument {
  source_path?: string;
  source?: string;
  path?: string;
  text?: string;
  content?: string;
}

export type ContextReceiptInput =
  | Record<string, string>
  | Array<[string, string] | ContextReceiptDocument | string>;

export interface ContextReceiptOptions {
  query?: string;
  budget?: number;
  tokenBudget?: number;
  token_budget?: number;
  chunkTokens?: number;
  chunk_tokens?: number;
  overlapTokens?: number;
  overlap_tokens?: number;
}

export function ingestReceiptDocuments(
  documents: ContextReceiptInput,
  options?: ContextReceiptOptions,
): Record<string, unknown>;

export function selectReceiptContext(
  index: Record<string, unknown>,
  options: ContextReceiptOptions & { query: string },
): Record<string, unknown>;

export function createContextReceipt(
  documents: ContextReceiptInput,
  options: ContextReceiptOptions & { query: string },
): Record<string, unknown>;

export function renderContextReceipt(receipt: Record<string, unknown>): string;
export function explainReceiptOmission(receipt: Record<string, unknown>, chunkId: string): string;
