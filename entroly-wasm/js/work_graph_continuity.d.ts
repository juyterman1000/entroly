import type {
  WorkContinuationProof,
  WorkGraphHandoffReceipt,
  WorkGraphResumeView,
} from "./work_graph";
import type { RepositoryDiscoveryOptions } from "./work_graph_repo";
import type { WorkGraphStoreOptions } from "./work_graph_store";

export interface ResumeRepositoryOptions {
  workstreamId?: string | null;
  maxEvidence?: number;
  storeOptions?: WorkGraphStoreOptions;
  repositoryOptions?: RepositoryDiscoveryOptions;
}

export interface HandoffRepositoryOptions {
  workstreamId: string;
  fromAgent: string;
  toAgent: string;
  generatedAtMs?: number;
  storeOptions?: WorkGraphStoreOptions;
  repositoryOptions?: RepositoryDiscoveryOptions;
}

export const MAX_RESUME_EVIDENCE: number;
export const MAX_WORK_ID_CHARS: number;

/**
 * Refresh bounded durable repository/checkpoint facts and recover unfinished
 * work from the shared Rust Work Graph. The previous agent does not need to
 * have produced an explicit handoff.
 */
export function resumeRepository(
  repoPath?: string,
  options?: ResumeRepositoryOptions,
): WorkGraphResumeView;

/** Refresh durable facts exactly once, then seal a graph-bound handoff receipt. */
export function handoffRepository(
  repoPath: string | undefined,
  options: HandoffRepositoryOptions,
): WorkGraphHandoffReceipt;

/**
 * Refresh durable facts once and return both the compatibility handoff receipt
 * and the Rust-verified complete continuation proof.
 */
export function handoffRepositoryWithProof(
  repoPath: string | undefined,
  options: HandoffRepositoryOptions,
): {
  handoff: WorkGraphHandoffReceipt;
  continuation_proof: WorkContinuationProof;
};
