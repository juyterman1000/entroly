import { WorkGraphStore } from "./work_graph_store";

export interface WorkContextSnapshotStoreOptions {
  maxContextBytes?: number;
  maxSnapshots?: number;
  maxTotalBytes?: number;
}

export interface VerifiedContextSnapshotBytes {
  payload: Record<string, unknown>;
  commitment: string;
  bytes: Uint8Array;
}

export class WorkContextSnapshotError extends Error {}

export function verifyCanonicalSnapshotBytes(
  value: Uint8Array | string,
  expectedCommitment?: string | null,
  maxBytes?: number,
): VerifiedContextSnapshotBytes;

export class WorkContextSnapshotStore {
  constructor(graphStore: WorkGraphStore, options?: WorkContextSnapshotStoreOptions);

  readonly graphStore: WorkGraphStore;
  readonly maxContextBytes: number;
  readonly maxSnapshots: number;
  readonly maxTotalBytes: number;
  readonly contextDir: string;

  static tokenForCommitment(commitment: string): string;
  static digestFromToken(token: string): string;

  putCanonicalBytes(value: Uint8Array | string, expectedCommitment: string): string;
  getCanonicalBytes(token: string): Uint8Array;
  getJSON(token: string): Record<string, unknown>;
}

export const CONTEXT_SNAPSHOT_TOKEN_PREFIX: "wctx1.";
export const DEFAULT_MAX_CONTEXT_BYTES: number;
export const DEFAULT_MAX_SNAPSHOTS: number;
export const DEFAULT_MAX_TOTAL_BYTES: number;
