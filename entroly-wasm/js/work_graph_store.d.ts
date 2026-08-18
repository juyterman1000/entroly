import type {
  WorkGraph,
  WorkGraphCoordinationReport,
  WorkGraphHandoffReceipt,
  WorkGraphResumeView,
} from "./work_graph";
import type { RepositoryDiscoveryOptions } from "./work_graph_repo";

export interface WorkGraphStoreOptions {
  root?: string;
  lockTimeoutMs?: number;
  staleLockMs?: number;
  maxStateBytes?: number;
}

export interface WorkClaimOptions {
  agentId: string;
  taskTitle: string;
  taskId?: string;
  sessionId?: string;
  scopePaths?: string[];
  scopeSymbols?: string[];
  ttlMs?: number;
  leaseId?: string;
  observedAtMs?: number;
  sourceKind?: "agent_statement" | "user_statement";
}

export class WorkGraphStoreError extends Error {}
export class WorkGraphLockTimeout extends WorkGraphStoreError {}
export class WorkGraphStateError extends WorkGraphStoreError {}

export class WorkGraphStore {
  constructor(repoId: string, options?: WorkGraphStoreOptions);
  static forRepository(path?: string, options?: WorkGraphStoreOptions): WorkGraphStore;
  readonly repoId: string;
  readonly root: string;
  readonly repoDir: string;
  readonly statePath: string;
  readonly lockPath: string;
  load(): WorkGraph;
  save(graph: WorkGraph): WorkGraph;
  submitObservation(observation: Record<string, unknown>): WorkGraph;
  updateRepository(path?: string, options?: RepositoryDiscoveryOptions): WorkGraph;
  claimWork(path: string, options: WorkClaimOptions): { graph: WorkGraph; leaseId: string };
  coordination(nowMs?: number): WorkGraphCoordinationReport;
  resume(workstreamId?: string | null, maxEvidence?: number): WorkGraphResumeView;
  handoff(
    workstreamId: string,
    fromAgent: string,
    toAgent: string,
    generatedAtMs?: number,
  ): WorkGraphHandoffReceipt;
}
