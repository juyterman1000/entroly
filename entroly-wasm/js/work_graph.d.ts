import type { RepositoryDiscoveryOptions } from "./work_graph_repo";

export type WorkGraphTrust = "untrusted" | "inferred" | "observed" | "verified";
export type WorkGraphStatus =
  | "unknown"
  | "planned"
  | "in_progress"
  | "blocked"
  | "needs_verification"
  | "completed"
  | "abandoned";

export interface WorkGraphSummary {
  schema_version: number;
  repo_id: string;
  revision: number;
  graph_commitment: string;
  event_count: number;
  node_count: number;
  edge_count: number;
  evidence_count: number;
  unfinished_count: number;
  blocked_count: number;
}

export interface WorkGraphWorkItem {
  node_id: string;
  kind: string;
  label: string;
  status: WorkGraphStatus;
  trust: WorkGraphTrust;
  updated_at_ms: number;
  task_ids: string[];
  agent_ids: string[];
  changed_paths: string[];
  symbol_ids: string[];
  commit_ids: string[];
  decision_ids: string[];
  failure_ids: string[];
  verification_ids: string[];
  evidence_ids: string[];
}

export interface WorkGraphEvidence {
  evidence_id: string;
  kind: string;
  source_ref: string;
  digest: string;
  locator: string;
  trust: WorkGraphTrust;
  observed_at_ms: number;
  attributes: Record<string, unknown>;
}

export interface WorkGraphResumeView {
  repo_id: string;
  graph_revision: number;
  graph_commitment: string;
  selected_workstream: WorkGraphWorkItem;
  task_labels: string[];
  agents: string[];
  decisions: string[];
  failures: string[];
  verification: string[];
  changed_paths: string[];
  commits: string[];
  evidence: WorkGraphEvidence[];
}

export interface WorkGraphContextScope {
  repo_id: string;
  graph_revision: number;
  graph_commitment: string;
  workstream_id: string;
  task_ids: string[];
  task_ids_total: number;
  task_ids_commitment: string;
  agent_ids: string[];
  agent_ids_total: number;
  agent_ids_commitment: string;
  changed_paths: string[];
  changed_paths_total: number;
  changed_paths_commitment: string;
  symbol_ids: string[];
  symbol_ids_total: number;
  symbol_ids_commitment: string;
  commit_ids: string[];
  commit_ids_total: number;
  commit_ids_commitment: string;
  evidence_ids: string[];
  evidence_ids_total: number;
  evidence_ids_commitment: string;
}

export interface WorkGraphCoordinationConflict {
  lease_a: string;
  lease_b: string;
  agent_a: string;
  agent_b: string;
  task_a: string;
  task_b: string;
  overlapping_paths: string[];
  overlapping_symbols: string[];
  reason: string;
}

export interface WorkGraphCoordinationReport {
  as_of_ms: number;
  active_leases: number;
  conflicts: WorkGraphCoordinationConflict[];
}

export interface WorkGraphHandoffReceipt {
  schema_version: number;
  repo_id: string;
  graph_revision: number;
  graph_commitment: string;
  workstream_id: string;
  from_agent: string;
  to_agent: string;
  generated_at_ms: number;
  node_ids: string[];
  edge_ids: string[];
  evidence_ids: string[];
  payload_commitment: string;
}

export interface WorkContinuationManifest {
  context_receipt_commitments?: string[];
  routing_commitments?: string[];
  execution_outcome_commitments?: string[];
  verification_commitments?: string[];
  memory_commitments?: string[];
  outstanding_work_refs?: string[];
  recovery_handle_ids?: string[];
  created_at_ms: number;
}

export interface WorkContinuationProof {
  schema_version: number;
  proof_id: string;
  repository_id: string;
  graph_revision: number;
  graph_commitment: string;
  workstream_id: string;
  from_agent: string;
  to_agent: string;
  handoff_commitment: string;
  context_receipt_commitments: string[];
  routing_commitments: string[];
  execution_outcome_commitments: string[];
  verification_commitments: string[];
  memory_commitments: string[];
  outstanding_work_refs: string[];
  recovery_handle_ids: string[];
  created_at_ms: number;
  proof_commitment: string;
}

export type WorkGraphObservation = Record<string, unknown> & {
  repo_id: string;
  observed_at_ms: number;
};

export class WorkGraph {
  constructor(repoId: string);
  static fromJSON(serialized: string | Record<string, unknown>): WorkGraph;
  static fromRepository(path?: string, options?: RepositoryDiscoveryOptions): WorkGraph;
  static verifyHandoffIntegrity(receipt: string | WorkGraphHandoffReceipt): boolean;
  readonly repoId: string;
  readonly revision: number;
  readonly graphCommitment: string;
  readonly eventCount: number;
  applyEvent(event: string | Record<string, unknown>): string;
  observeRepository(observation: string | WorkGraphObservation): string;
  refreshRepository(path?: string, options?: RepositoryDiscoveryOptions): string;
  merge(other: WorkGraph | string | Record<string, unknown>): number;
  exportJSON(pretty?: boolean): string;
  exportState(): Record<string, unknown>;
  summary(): WorkGraphSummary;
  snapshot(pretty?: boolean): Record<string, unknown>;
  unfinished(pretty?: boolean): WorkGraphWorkItem[];
  resume(workstreamId?: string | null, maxEvidence?: number, pretty?: boolean): WorkGraphResumeView;
  contextScope(workstreamId?: string | null, maxEvidence?: number, pretty?: boolean): WorkGraphContextScope;
  recordContextReceipt(
    receipt: string | Record<string, unknown>,
    agentId?: string,
    sessionId?: string,
  ): string;
  recordMemory(
    memory: string | Record<string, unknown>,
    nowMs?: number,
    supersededIds?: string[],
  ): string;
  recordExecutionChain(
    route: string | Record<string, unknown>,
    outcome: string | Record<string, unknown>,
    verification: string | Record<string, unknown>,
    invalidatedCommitments?: string[],
  ): string;
  continuationProof(
    handoff: string | WorkGraphHandoffReceipt,
    manifest: WorkContinuationManifest,
  ): WorkContinuationProof;
  reconstructedContinuationProof(
    workstreamId: string,
    toAgent: string,
    manifest: WorkContinuationManifest,
  ): WorkContinuationProof;
  coordination(nowMs?: number, pretty?: boolean): WorkGraphCoordinationReport;
  handoff(
    workstreamId: string,
    fromAgent: string,
    toAgent: string,
    generatedAtMs?: number,
    pretty?: boolean,
  ): WorkGraphHandoffReceipt;
  verifyHandoff(receipt: string | WorkGraphHandoffReceipt): boolean;
}

export {
  RepositoryDiscoveryError,
  discoverRepositoryIdentity,
  discoverRepositoryObservation,
} from "./work_graph_repo";
