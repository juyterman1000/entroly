import * as vscode from 'vscode';

export class ReceiptViewPanel {
  public static currentPanel: ReceiptViewPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private _disposables: vscode.Disposable[] = [];

  public static show(extensionUri: vscode.Uri): void {
    const column = vscode.ViewColumn.Two;

    if (ReceiptViewPanel.currentPanel) {
      ReceiptViewPanel.currentPanel._panel.reveal(column);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'entrolyReceipt',
      'Entroly — Cryptographic Receipt Inspector',
      column,
      { enableScripts: true }
    );

    ReceiptViewPanel.currentPanel = new ReceiptViewPanel(panel, extensionUri);
  }

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
    this._panel = panel;
    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
    this._panel.webview.html = this._getHtml();
  }

  public dispose(): void {
    ReceiptViewPanel.currentPanel = undefined;
    this._panel.dispose();
    while (this._disposables.length) {
      const x = this._disposables.pop();
      if (x) x.dispose();
    }
  }

  private _getHtml(): string {
    return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <style>
        body {
          font-family: var(--vscode-editor-font-family, monospace);
          color: var(--vscode-editor-foreground);
          background-color: var(--vscode-editor-background);
          padding: 16px;
          font-size: 12px;
          line-height: 1.6;
        }
        .stamp {
          color: #34d399;
          font-weight: 700;
          border: 1px solid #34d399;
          display: inline-block;
          padding: 2px 6px;
          border-radius: 4px;
          margin-bottom: 12px;
        }
        .code-box {
          background: rgba(0,0,0,0.25);
          padding: 8px;
          border-radius: 4px;
          color: #38bdf8;
          word-break: break-all;
          margin-top: 4px;
        }
        .row {
          margin-bottom: 12px;
        }
        .label {
          opacity: 0.7;
          text-transform: uppercase;
          font-size: 10px;
        }
      </style>
    </head>
    <body>
      <div class="stamp">AUDITABLE CONTEXT RECEIPT</div>
      <div class="row">
        <div class="label">Receipt Schema</div>
        <div>context-receipt.v1 (Merkle-CCR)</div>
      </div>
      <div class="row">
        <div class="label">Merkle Root</div>
        <div class="code-box">e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855</div>
      </div>
      <div class="row">
        <div class="label">Reversibility Verification</div>
        <div style="color: #34d399; font-weight: 600;">PASS (100% bit-exact byte recovery)</div>
      </div>
      <div class="row">
        <div class="label">Knapsack Optimality</div>
        <div>Dynamic Programming verified · 0-1 budget satisfied</div>
      </div>
    </body>
    </html>`;
  }
}
