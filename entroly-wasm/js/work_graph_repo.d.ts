import type { WorkGraphObservation } from "./work_graph";

export interface RepositoryDiscoveryOptions {
  agentId?: string;
  sessionId?: string;
  taskHint?: Record<string, unknown> | null;
  defaultBranch?: string;
  maxCommits?: number;
  observedAtMs?: number;
  includeCheckpoint?: boolean;
  checkpointDir?: string;
}

export interface RepositoryIdentity {
  repo_id: string;
  root: string;
}

export class RepositoryDiscoveryError extends Error {}

export function discoverRepositoryIdentity(path?: string): RepositoryIdentity;

export function discoverRepositoryObservation(
  path?: string,
  options?: RepositoryDiscoveryOptions,
): WorkGraphObservation;
