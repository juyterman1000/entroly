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
