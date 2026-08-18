import type { WorkGraphResumeView } from "./work_graph";
import type { RepositoryDiscoveryOptions } from "./work_graph_repo";
import type { WorkGraphStoreOptions } from "./work_graph_store";

export interface ResumeRepositoryOptions {
  workstreamId?: string | null;
  maxEvidence?: number;
  storeOptions?: WorkGraphStoreOptions;
  repositoryOptions?: RepositoryDiscoveryOptions;
}

export const MAX_RESUME_EVIDENCE: number;

/**
 * Refresh bounded durable repository/checkpoint facts and recover unfinished
 * work from the shared Rust Work Graph. The previous agent does not need to
 * have produced an explicit handoff.
 */
export function resumeRepository(
  repoPath?: string,
  options?: ResumeRepositoryOptions,
): WorkGraphResumeView;
