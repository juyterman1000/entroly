/**
 * Keeps Entroly's index aware of edits that have not been written to disk.
 *
 * Entroly indexes files from disk and re-indexes them when they change, so a
 * saved edit is picked up. An unsaved buffer is invisible to that: selection
 * ranks and returns the version on disk while the developer is asking about
 * the version in front of them.
 *
 * This tracks dirty documents and pushes their text to the local proxy, which
 * replaces the indexed content for that path until the document is saved or
 * closed. The editor is the only component that can see a dirty buffer, so
 * this has to live here.
 *
 * The transport is injected so the scheduling and lifecycle rules below are
 * testable without an editor host or a running proxy.
 */

export interface OverlayTransport {
  /** Replace the indexed content for `path`. */
  push(path: string, content: string): Promise<void>;
  /** Drop any overlay for `path`, returning the index to the file on disk. */
  clear(path: string): Promise<void>;
}

export interface OverlayOptions {
  /** Documents larger than this are not pushed. */
  maxBytes?: number;
  /** Paths matching any of these are never pushed. */
  excluded?: RegExp[];
}

const DEFAULT_MAX_BYTES = 512 * 1024;

// An unsaved buffer in one of these is not repository context, and some of
// them routinely hold credentials a developer has not committed.
const DEFAULT_EXCLUDED: RegExp[] = [
  /(^|[\\/])\.env(\.|$)/i,
  /(^|[\\/])\.git[\\/]/i,
  /(^|[\\/])node_modules[\\/]/i,
  /\.(pem|key|p12|pfx|keystore)$/i,
];

export class BufferOverlay {
  private readonly pending = new Map<string, string>();
  private readonly active = new Set<string>();
  private inFlight: Promise<void> = Promise.resolve();
  private readonly maxBytes: number;
  private readonly excluded: RegExp[];

  constructor(
    private readonly transport: OverlayTransport,
    options: OverlayOptions = {}
  ) {
    this.maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;
    this.excluded = options.excluded ?? DEFAULT_EXCLUDED;
  }

  /** True when this document may be pushed at all. */
  eligible(path: string, content: string): boolean {
    if (!path) {
      return false;
    }
    if (Buffer.byteLength(content, 'utf8') > this.maxBytes) {
      return false;
    }
    return !this.excluded.some((pattern) => pattern.test(path));
  }

  /**
   * Record an edit. Nothing is sent until `flush`, so a burst of keystrokes
   * collapses into one push of the final text rather than one per character.
   */
  recordChange(path: string, content: string, isDirty: boolean): void {
    if (!isDirty) {
      // The editor reports a document as clean after an undo back to the
      // saved state. Disk is authoritative again, so retract rather than
      // leaving the last pushed text in place.
      this.recordSave(path);
      return;
    }
    if (!this.eligible(path, content)) {
      return;
    }
    this.pending.set(path, content);
  }

  /** The document matches disk again; the index should too. */
  recordSave(path: string): void {
    this.pending.delete(path);
    if (this.active.has(path)) {
      this.pending.set(path, '');
    }
  }

  /** Same as a save from the index's point of view: stop overriding disk. */
  recordClose(path: string): void {
    this.recordSave(path);
  }

  /** Paths whose overlay is currently installed in the index. */
  activePaths(): string[] {
    return [...this.active].sort();
  }

  pendingCount(): number {
    return this.pending.size;
  }

  /**
   * Send everything recorded since the last flush.
   *
   * A failed push is dropped rather than retried: the overlay is an
   * optimisation over disk, and a stuck retry loop against a proxy that is not
   * running would be worse than ranking the saved version of the file.
   */
  flush(): Promise<void> {
    const batch = [...this.pending.entries()];
    this.pending.clear();
    const run = this.inFlight.then(async () => {
      for (const [path, content] of batch) {
        try {
          if (content === '') {
            await this.transport.clear(path);
            this.active.delete(path);
          } else {
            await this.transport.push(path, content);
            this.active.add(path);
          }
        } catch {
          // A failed clear may leave the remote overlay installed.
        }
      }
    });
    this.inFlight = run;
    return run;
  }

  /**
   * Retract every overlay. Called when the extension deactivates so a
   * reloaded window does not leave the index serving text from a buffer that
   * no longer exists anywhere.
   */
  async dispose(): Promise<void> {
    this.pending.clear();
    await this.inFlight;
    for (const path of [...this.active]) {
      try {
        await this.transport.clear(path);
        this.active.delete(path);
      } catch {
        // Best effort on shutdown.
      }
    }
  }
}
