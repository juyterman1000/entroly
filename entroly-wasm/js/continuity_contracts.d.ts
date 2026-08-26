export interface RoutingDecisionInput {
  repository_id: string;
  task_id: string;
  workstream_id: string;
  provider: string;
  model: string;
  runtime: string;
  context_budget_tokens: number;
  policy_version: string;
  reason_codes?: string[];
  feature_commitments?: string[];
  fallback_route_ids?: string[];
  receipt_id?: string;
  evidence_ids?: string[];
  decided_at_ms: number;
}

export type RoutingDecision = RoutingDecisionInput & {
  schema_version: number;
  routing_id: string;
  decision_commitment: string;
};

export interface ModelExecutionOutcomeInput {
  routing_id: string;
  repository_id: string;
  task_id: string;
  workstream_id: string;
  provider: string;
  model: string;
  runtime: string;
  receipt_id?: string;
  request_commitment?: string;
  response_commitment?: string;
  state: "succeeded" | "failed" | "cancelled" | "unknown";
  verification_state: "passed" | "failed" | "skipped" | "unknown" | "stale";
  latency_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  cost_micro_usd?: number;
  error_code?: string;
  evidence_ids?: string[];
  completed_at_ms: number;
}

export type ModelExecutionOutcome = ModelExecutionOutcomeInput & {
  schema_version: number;
  outcome_id: string;
  outcome_commitment: string;
};

export interface VerificationRecordInput {
  repository_id: string;
  subject_id: string;
  subject_commitment: string;
  verified_repository_commitment: string;
  verdict: "passed" | "failed" | "skipped" | "unknown";
  evidence_ids?: string[];
  dependency_commitments?: string[];
  observed_at_ms: number;
  valid_until_ms?: number;
}

export type VerificationRecord = VerificationRecordInput & {
  schema_version: number;
  verification_id: string;
  record_commitment: string;
};

export function createRoutingDecision(input: RoutingDecisionInput): RoutingDecision;
export function createModelExecutionOutcome(
  input: ModelExecutionOutcomeInput,
): ModelExecutionOutcome;
export function createVerificationRecord(input: VerificationRecordInput): VerificationRecord;
export function verificationFreshness(
  record: VerificationRecord | string,
  currentRepositoryCommitment: string,
  nowMs: number,
  invalidatedCommitments?: string[],
): "current" | "stale" | "invalidated" | "unknown";
export function continuationProofState(
  proof: Record<string, unknown> | string,
  repositoryId: string,
  graphRevision: number,
  graphCommitment: string,
): "valid" | "stale" | "invalid";
