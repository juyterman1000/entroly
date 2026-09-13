export * from "entroly-wasm";

// --- Shared Memory ---

export interface SharedEntry {
  entry_id: string;
  content_hash: string;
  content: string;
  agent_id: string;
  session_id: string;
  tags: string[];
  simhash: number;
  timestamp: number;
  metadata: Record<string, unknown>;
}

export interface SharedMemoryStats {
  total_entries: number;
  total_tokens: number;
  agents: Record<string, number>;
  top_tags: Record<string, number>;
  oldest: number;
  newest: number;
}

export interface SharedMemoryClient {
  write(content: string, agentId?: string, tags?: string[]): Promise<SharedEntry | null>;
  search(query: string, topK?: number, agentId?: string): Promise<SharedEntry[]>;
  list(agentId?: string, tag?: string, limit?: number): Promise<SharedEntry[]>;
  forget(entryId: string): Promise<boolean>;
  stats(): Promise<SharedMemoryStats>;
}

// --- Output Steering ---

export type EffortLevel = "MINIMAL" | "CONCISE" | "STANDARD" | "DETAILED" | "EXHAUSTIVE";

export interface EffortClassification {
  effort: EffortLevel;
  confidence: number;
  reason: string;
  max_tokens: number;
  directive: string;
}

export interface OutputSteeringClient {
  classify(query: string): Promise<EffortClassification>;
  steer(query: string, effort?: EffortLevel): Promise<EffortClassification>;
}

// --- Context Receipts ---

export interface ContextReceiptFragment {
  id: string;
  source: string;
  byte_start: number;
  byte_end: number;
  token_count: number;
  sha256: string;
}

export interface ContextReceipt {
  receipt_id: string;
  fragments: ContextReceiptFragment[];
  omitted: ContextReceiptFragment[];
  total_tokens: number;
  budget_tokens: number;
  timestamp: number;
}

// --- Verification ---

export interface ClaimAssessment {
  claim: string;
  grounded: boolean;
  confidence: number;
  source_span?: string;
}

export interface VerificationResult {
  claims: ClaimAssessment[];
  overall_grounded: boolean;
  grounding_ratio: number;
}

// --- Chat Messages ---

export interface TextContent {
  type: "text";
  text: string;
}

export interface ImageContent {
  type: "image_url";
  image_url: { url: string };
}

export type ContentPart = TextContent | ImageContent;

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string | ContentPart[];
}

// --- Framework Adapters ---

export function langchainAdapter<T extends { invoke: Function }>(
  llm: T,
  options?: { budget?: number; quality?: string },
): T;

export function llamaindexAdapter<T extends { query: Function }>(
  engine: T,
  options?: { budget?: number },
): T;

export function litellmMiddleware(
  options?: { budget?: number; port?: number },
): (req: any, res: any, next: () => void) => void;

// --- Client Factory ---

export interface EntrolyClientOptions {
  proxyUrl?: string;
  port?: number;
}

export interface EntrolyClient {
  compress(text: string, budget?: number): Promise<string>;
  compressMessages(messages: ChatMessage[], budget?: number): Promise<ChatMessage[]>;
  sharedMemory: SharedMemoryClient;
  outputSteering: OutputSteeringClient;
}

export function createClient(options?: EntrolyClientOptions): EntrolyClient;
