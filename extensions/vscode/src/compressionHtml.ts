// HTML for the context-compression webview. Kept free of the `vscode` module so
// it can be tested under plain Node.
//
// Everything interpolated here comes from the user's workspace: the file name and
// the file's own text. A cloned repository is attacker-controlled input, so a
// file containing `</pre><img src=x onerror=...>` must render as text. The page
// also carries a nonce-only Content-Security-Policy, so markup that got past the
// escaping still could not run script.

/**
 * What `entroly compress --json` reports for one file. Every number the panel
 * shows comes from here; the panel computes none of them itself.
 */
export interface CompressReceipt {
  tokensBefore: number;
  tokensAfter: number;
  reductionPct: number;
  codec: string;
  sourceSha256: string;
  recoveryDigest: string;
  recoveryStore: string;
}

export interface CompressionView {
  fileName: string;
  receipt: CompressReceipt;
  /** Exact text shown in the preview and copied by the Copy button. */
  packet: string;
  /** Per-render random value; the only script allowed to run carries it. */
  nonce: string;
}

const HTML_ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

export function escapeHtml(value: string): string {
  return String(value).replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/**
 * Read the receipt `entroly compress --json` printed, or null when stdout is
 * not one. Without a receipt there is nothing honest to display.
 */
export function parseCompressReceipt(stdout: string): CompressReceipt | null {
  let data: unknown;
  try {
    data = JSON.parse(stdout);
  } catch {
    return null;
  }
  if (typeof data !== 'object' || data === null) {
    return null;
  }
  const raw = data as Record<string, unknown>;
  const before = finiteNumber(raw.tokens_before);
  const after = finiteNumber(raw.tokens_after);
  const pct = finiteNumber(raw.reduction_pct);
  if (before === null || after === null || pct === null) {
    return null;
  }
  return {
    tokensBefore: before,
    tokensAfter: after,
    reductionPct: pct,
    codec: text(raw.codec),
    sourceSha256: text(raw.source_sha256),
    recoveryDigest: text(raw.recovery_digest),
    recoveryStore: text(raw.recovery_store),
  };
}

/** The last non-empty line of CLI output, without terminal colour codes. */
export function lastLine(output: string): string {
  const lines = String(output)
    .replace(/\x1b\[[0-9;]*m/g, '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length > 0 ? lines[lines.length - 1] : '';
}

/**
 * The text the preview shows and the Copy button copies: what
 * `entroly compress` wrote, under one line saying what it is and how to get
 * the omitted parts back.
 */
export function compressionPacket(
  compressed: string,
  fileName: string,
  receipt: CompressReceipt
): string {
  const recover = receipt.recoveryDigest
    ? ` Omitted parts: entroly recover ${receipt.recoveryDigest}` +
      (receipt.recoveryStore ? ` --store "${receipt.recoveryStore}"` : '')
    : '';
  return (
    `[Entroly compressed view of ${fileName}: ${receipt.tokensBefore} -> ` +
    `${receipt.tokensAfter} tokens.${recover}]\n` +
    compressed
  );
}

/** A message the webview may send. Anything else is ignored. */
export function isCopyRequest(message: unknown): boolean {
  return (
    typeof message === 'object' &&
    message !== null &&
    (message as { command?: unknown }).command === 'copy'
  );
}

export function compressionHtml(view: CompressionView): string {
  const nonce = escapeHtml(view.nonce);
  return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}'; base-uri 'none'; form-action 'none';">
      <style>
        body {
          font-family: var(--vscode-font-family, -apple-system, sans-serif);
          color: var(--vscode-editor-foreground);
          background-color: var(--vscode-editor-background);
          padding: 16px;
          line-height: 1.5;
        }
        .header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          border-bottom: 1px solid var(--vscode-widget-border, rgba(255,255,255,0.1));
          padding-bottom: 10px;
          margin-bottom: 16px;
        }
        .title {
          font-size: 16px;
          font-weight: 700;
          color: #818cf8;
        }
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 10px;
          margin-bottom: 16px;
        }
        .stat-card {
          background: var(--vscode-editor-inactiveSelectionBackground, rgba(255,255,255,0.04));
          border: 1px solid var(--vscode-widget-border, rgba(255,255,255,0.1));
          border-radius: 6px;
          padding: 10px;
          text-align: center;
        }
        .stat-val {
          font-size: 18px;
          font-weight: 700;
          color: #34d399;
          font-family: var(--vscode-editor-font-family, monospace);
        }
        .stat-lbl {
          font-size: 11px;
          opacity: 0.7;
          text-transform: uppercase;
        }
        .btn-copy {
          background: #4f46e5;
          color: white;
          border: none;
          border-radius: 4px;
          padding: 8px 14px;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          margin-bottom: 14px;
          display: inline-flex;
          align-items: center;
          gap: 6px;
        }
        .btn-copy:hover {
          background: #4338ca;
        }
        .note {
          font-size: 11px;
          opacity: 0.75;
          margin-bottom: 12px;
          word-break: break-all;
        }
        pre {
          background: var(--vscode-textCodeBlock-background, rgba(0,0,0,0.2));
          border: 1px solid var(--vscode-widget-border, rgba(255,255,255,0.1));
          border-radius: 6px;
          padding: 12px;
          font-family: var(--vscode-editor-font-family, monospace);
          font-size: 12px;
          max-height: 400px;
          overflow-y: auto;
          white-space: pre-wrap;
          word-break: break-all;
        }
      </style>
    </head>
    <body>
      <div class="header">
        <div class="title">⚡ Entroly compress</div>
        <span>${escapeHtml(view.fileName)}</span>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-val">${Number(view.receipt.tokensBefore)}</div>
          <div class="stat-lbl">Tokens before</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">${Number(view.receipt.tokensAfter)}</div>
          <div class="stat-lbl">Tokens after</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">${Number(view.receipt.reductionPct).toFixed(1)}%</div>
          <div class="stat-lbl">Reduction</div>
        </div>
      </div>

      <div class="note">Measured by <code>entroly compress</code>${
        view.receipt.codec ? ` (codec ${escapeHtml(view.receipt.codec)})` : ''
      }. Source ${escapeHtml(view.receipt.sourceSha256 || 'digest not reported')}.</div>

      <button class="btn-copy" id="copyBtn">📋 Copy Context for Prompt</button>

      <pre id="codeView">${escapeHtml(view.packet)}</pre>

      <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        document.getElementById('copyBtn').addEventListener('click', () => {
          vscode.postMessage({ command: 'copy' });
        });
      </script>
    </body>
    </html>`;
}

const STATIC_CSP =
  "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none';";

/**
 * Shown when `entroly compress` could not run or did not return a receipt.
 * It states the reason and shows no numbers and no text to copy: a figure
 * the CLI did not measure is not one the panel may print.
 */
export function compressionErrorHtml(fileName: string, reason: string): string {
  return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="Content-Security-Policy" content="${STATIC_CSP}">
      <style>
        body { font-family: var(--vscode-font-family, sans-serif); color: var(--vscode-editor-foreground);
               background-color: var(--vscode-editor-background); padding: 16px; line-height: 1.5; }
        code { font-family: var(--vscode-editor-font-family, monospace); }
      </style>
    </head>
    <body>
      <h3>Entroly could not compress ${escapeHtml(fileName)}</h3>
      <p>${escapeHtml(reason)}</p>
      <p>This command runs the local <code>entroly compress</code> CLI. Install it with
      <code>pip install -U entroly</code>, make sure <code>entroly</code> is on the PATH VS Code
      starts with, and run the command again.</p>
    </body>
    </html>`;
}

/**
 * The receipt from this window's last `entroly compress` run, or a page that
 * says there is none. It never shows a value the CLI did not report.
 */
export function receiptHtml(receipt: CompressReceipt | undefined, fileName = ''): string {
  const rows = receipt
    ? [
        ['Source', fileName],
        ['Source SHA-256', receipt.sourceSha256 || 'not reported'],
        ['Tokens', `${receipt.tokensBefore} -> ${receipt.tokensAfter} (${receipt.reductionPct.toFixed(1)}% reduction)`],
        ['Codec', receipt.codec || 'not reported'],
        ['Recovery digest', receipt.recoveryDigest || 'not reported'],
        ['Recovery store', receipt.recoveryStore || 'not reported'],
      ]
        .map(
          ([label, value]) =>
            `<div class="row"><div class="label">${escapeHtml(label)}</div><div class="code-box">${escapeHtml(value)}</div></div>`
        )
        .join('')
    : `<p>No receipt yet. Run <b>Entroly: Compress Current File for Prompt</b>; it runs
       <code>entroly compress</code> and this view shows the receipt that run writes.</p>`;
  return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="Content-Security-Policy" content="${STATIC_CSP}">
      <style>
        body { font-family: var(--vscode-editor-font-family, monospace); color: var(--vscode-editor-foreground);
               background-color: var(--vscode-editor-background); padding: 16px; font-size: 12px; line-height: 1.6; }
        .code-box { background: rgba(0,0,0,0.25); padding: 8px; border-radius: 4px; word-break: break-all; margin-top: 4px; }
        .row { margin-bottom: 12px; }
        .label { opacity: 0.7; text-transform: uppercase; font-size: 10px; }
      </style>
    </head>
    <body>
      <h3>Entroly compress receipt</h3>
      ${rows}
      ${receipt ? '<p>Restore the omitted parts with <code>entroly recover &lt;recovery digest&gt; --store &lt;recovery store&gt;</code>.</p>' : ''}
    </body>
    </html>`;
}
