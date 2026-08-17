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

export { RepositoryDiscoveryError, discoverRepositoryObservation } from "./work_graph_repo";
