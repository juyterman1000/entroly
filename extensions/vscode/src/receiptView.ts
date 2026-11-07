import * as vscode from 'vscode';
import { receiptHtml } from './compressionHtml';
import { lastCompression } from './compressionView';

export class ReceiptViewPanel {
  public static currentPanel: ReceiptViewPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private _disposables: vscode.Disposable[] = [];

  public static show(extensionUri: vscode.Uri): void {
    const column = vscode.ViewColumn.Two;

    if (ReceiptViewPanel.currentPanel) {
      ReceiptViewPanel.currentPanel._panel.webview.html =
        ReceiptViewPanel.currentPanel._getHtml();
      ReceiptViewPanel.currentPanel._panel.reveal(column);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'entrolyReceipt',
      'Entroly — Compress Receipt',
      column,
      { enableScripts: false }
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

  /**
   * The receipt of this window's last `entroly compress` run, or a page that
   * says there is none. This panel used to show the same hardcoded receipt to
   * everyone: a "Merkle Root" that was the SHA-256 of the empty string, a
   * reversibility "PASS (100% bit-exact byte recovery)" and a "Dynamic
   * Programming verified" knapsack line, none of them computed from anything.
   */
  private _getHtml(): string {
    return receiptHtml(lastCompression?.receipt, lastCompression?.fileName);
  }
}
