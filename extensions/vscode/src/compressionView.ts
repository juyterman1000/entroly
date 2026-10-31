import { randomBytes } from 'crypto';
import * as vscode from 'vscode';
import { compressionHtml, compressionPacket, isCopyRequest } from './compressionHtml';
import { estimateTokens } from './tokenCounter';

export class CompressionViewPanel {
  public static currentPanel: CompressionViewPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private _disposables: vscode.Disposable[] = [];
  // What the Copy button copies: the text this panel rendered, never text the
  // webview sends back. A script injected into the page could otherwise put
  // any content on the clipboard, where the user pastes it into a prompt or a
  // terminal.
  private _packet = '';

  public static createOrShow(
    extensionUri: vscode.Uri,
    rawText: string,
    fileName: string
  ): void {
    const column = vscode.window.activeTextEditor
      ? vscode.ViewColumn.Beside
      : undefined;

    if (CompressionViewPanel.currentPanel) {
      CompressionViewPanel.currentPanel._panel.reveal(column);
      CompressionViewPanel.currentPanel.update(rawText, fileName);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'entrolyCompression',
      'Entroly — Context Compression',
      column || vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: []
      }
    );

    CompressionViewPanel.currentPanel = new CompressionViewPanel(
      panel,
      extensionUri,
      rawText,
      fileName
    );
  }

  private constructor(
    panel: vscode.WebviewPanel,
    extensionUri: vscode.Uri,
    rawText: string,
    fileName: string
  ) {
    this._panel = panel;

    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

    this._panel.webview.onDidReceiveMessage(
      (message: unknown) => {
        if (isCopyRequest(message) && this._packet) {
          vscode.env.clipboard.writeText(this._packet);
          vscode.window.showInformationMessage('✓ Entroly context copied to clipboard!');
        }
      },
      null,
      this._disposables
    );

    this.update(rawText, fileName);
  }

  public update(rawText: string, fileName: string): void {
    this._panel.webview.html = this._getHtmlForWebview(rawText, fileName);
  }

  public dispose(): void {
    CompressionViewPanel.currentPanel = undefined;
    this._panel.dispose();
    while (this._disposables.length) {
      const x = this._disposables.pop();
      if (x) {
        x.dispose();
      }
    }
  }

  private _getHtmlForWebview(rawText: string, fileName: string): string {
    const rawTokens = estimateTokens(rawText);
    const targetTokens = Math.min(rawTokens, Math.max(120, Math.floor(rawTokens * 0.28)));
    const savingsPct = rawTokens > 0 ? (((rawTokens - targetTokens) / rawTokens) * 100).toFixed(1) : '0';
    this._packet = compressionPacket(rawText, fileName, Date.now());

    return compressionHtml({
      fileName,
      rawTokens,
      targetTokens,
      savingsPct,
      packet: this._packet,
      nonce: randomBytes(16).toString('base64'),
    });
  }
}
