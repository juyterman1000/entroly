import * as vscode from 'vscode';
import { TokenStatusBarManager } from './tokenCounter';
import { CompressionViewPanel } from './compressionView';
import { ReceiptViewPanel } from './receiptView';

export function activate(context: vscode.ExtensionContext) {
  // Initialize live token counter status bar
  const tokenManager = new TokenStatusBarManager();
  context.subscriptions.push(tokenManager);

  // Command 1: Compress Selection
  context.subscriptions.push(
    vscode.commands.registerCommand('entroly.compressSelection', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
      }
      const selection = editor.selection;
      const text = selection.isEmpty
        ? editor.document.getText()
        : editor.document.getText(selection);

      CompressionViewPanel.createOrShow(
        context.extensionUri,
        text,
        editor.document.fileName
      );
    })
  );

  // Command 2: Compress File
  context.subscriptions.push(
    vscode.commands.registerCommand('entroly.compressFile', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
      }
      const text = editor.document.getText();
      CompressionViewPanel.createOrShow(
        context.extensionUri,
        text,
        editor.document.fileName
      );
    })
  );

  // Command 3: View Cryptographic Receipt
  context.subscriptions.push(
    vscode.commands.registerCommand('entroly.viewReceipt', () => {
      ReceiptViewPanel.show(context.extensionUri);
    })
  );

  // Command 4: Open Web Playground
  context.subscriptions.push(
    vscode.commands.registerCommand('entroly.openPlayground', () => {
      vscode.env.openExternal(
        vscode.Uri.parse('https://juyterman1000.github.io/entroly/playground/')
      );
    })
  );
}

export function deactivate() {}
