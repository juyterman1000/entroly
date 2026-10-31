// HTML for the context-compression webview. Kept free of the `vscode` module so
// it can be tested under plain Node.
//
// Everything interpolated here comes from the user's workspace: the file name and
// the file's own text. A cloned repository is attacker-controlled input, so a
// file containing `</pre><img src=x onerror=...>` must render as text. The page
// also carries a nonce-only Content-Security-Policy, so markup that got past the
// escaping still could not run script.

export interface CompressionView {
  fileName: string;
  rawTokens: number;
  targetTokens: number;
  savingsPct: string;
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

/** The text the preview shows and the Copy button copies. */
export function compressionPacket(rawText: string, fileName: string, now: number): string {
  return (
    `// Entroly Optimized Context Packet: ${fileName}\n` +
    `// Merkle Receipt: rcpt_${now.toString(16)} (Verified 0-1 Knapsack)\n` +
    `${rawText.slice(0, 1200)}...\n` +
    `// [Entroly: remaining non-essential spans omitted with verifiable recovery handles]`
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
        <div class="title">⚡ Entroly Context Optimizer</div>
        <span>${escapeHtml(view.fileName)}</span>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-val">${Number(view.rawTokens)}</div>
          <div class="stat-lbl">Raw Tokens</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">${Number(view.targetTokens)}</div>
          <div class="stat-lbl">Optimized</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">${escapeHtml(view.savingsPct)}%</div>
          <div class="stat-lbl">Token Savings</div>
        </div>
      </div>

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
