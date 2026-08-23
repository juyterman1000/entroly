import type {
  WorkGraph,
  WorkGraphCoordinationReport,
  WorkGraphHandoffReceipt,
  WorkGraphResumeView,
  WorkContinuationManifest,
  WorkContinuationProof,
} from "./work_graph";
import type {
  ModelExecutionOutcome,
  RoutingDecision,
  VerificationRecord,
} from "./continuity_contracts";
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
  recordContextReceipt(
    receipt: string | Record<string, unknown>,
    options?: { agentId?: string; sessionId?: string },
  ): { graph: WorkGraph; result: string };
  recordMemory(
    memory: string | Record<string, unknown>,
    nowMs: number,
    supersededIds?: string[],
  ): { graph: WorkGraph; result: string };
  recordExecutionChain(
    route: string | RoutingDecision,
    outcome: string | ModelExecutionOutcome,
    verification: string | VerificationRecord,
    invalidatedCommitments?: string[],
  ): { graph: WorkGraph; result: string };
  continuationProof(
    handoff: string | WorkGraphHandoffReceipt,
    manifest: WorkContinuationManifest,
  ): WorkContinuationProof;
  reconstructedContinuationProof(
    workstreamId: string,
    toAgent: string,
    manifest: WorkContinuationManifest,
  ): WorkContinuationProof;
  handoff(
    workstreamId: string,
    fromAgent: string,
    toAgent: string,
    generatedAtMs?: number,
  ): WorkGraphHandoffReceipt;
}
