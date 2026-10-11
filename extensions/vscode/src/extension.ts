import * as vscode from 'vscode';
import { TokenStatusBarManager } from './tokenCounter';
import { CompressionViewPanel } from './compressionView';
import { ReceiptViewPanel } from './receiptView';
import { BufferOverlay, OverlayTransport } from './bufferOverlay';

const OVERLAY_FLUSH_MS = 600;

/** Posts overlays to the local proxy. Failures are the caller's to swallow. */
function httpTransport(baseUrl: () => string): OverlayTransport {
  const send = async (path: string, content: string | null) => {
    const response = await fetch(`${baseUrl()}/overlay`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ path, content }),
    });
    if (!response.ok) {
      throw new Error(`overlay ${response.status}`);
    }
  };
  return {
    push: (path, content) => send(path, content),
    clear: (path) => send(path, null),
  };
}

export function activate(context: vscode.ExtensionContext) {
  // Initialize live token counter status bar
  const tokenManager = new TokenStatusBarManager();
  context.subscriptions.push(tokenManager);

  // ── Unsaved buffers ──
  // Entroly indexes files from disk, so an unsaved edit is invisible to
  // selection: it ranks the saved version while the developer is asking about
  // the one on screen. The editor is the only component that can see a dirty
  // buffer, so it pushes them to the local proxy, which drops the override
  // again on save or close.
  const overlay = new BufferOverlay(
    httpTransport(() =>
      vscode.workspace
        .getConfiguration('entroly')
        .get<string>('proxyUrl', 'http://127.0.0.1:9377')
        .replace(/\/+$/, '')
    )
  );
  let flushTimer: NodeJS.Timeout | undefined;
  const scheduleFlush = () => {
    if (flushTimer) {
      clearTimeout(flushTimer);
    }
    flushTimer = setTimeout(() => void overlay.flush(), OVERLAY_FLUSH_MS);
  };
  const overlayEnabled = () =>
    vscode.workspace
      .getConfiguration('entroly')
      .get<boolean>('syncUnsavedBuffers', true);

  context.subscriptions.push(
    // `file` only: an untitled buffer has no path on disk for the index to
    // override, and the endpoint rejects a path that does not resolve.
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (!overlayEnabled() || event.document.uri.scheme !== 'file') {
        return;
      }
      overlay.recordChange(
        event.document.uri.fsPath,
        event.document.getText(),
        event.document.isDirty
      );
      scheduleFlush();
    }),
    vscode.workspace.onDidSaveTextDocument((document) => {
      if (document.uri.scheme !== 'file') {
        return;
      }
      overlay.recordSave(document.uri.fsPath);
      scheduleFlush();
    }),
    vscode.workspace.onDidCloseTextDocument((document) => {
      if (document.uri.scheme !== 'file') {
        return;
      }
      overlay.recordClose(document.uri.fsPath);
      scheduleFlush();
    }),
    {
      dispose: () => {
        if (flushTimer) {
          clearTimeout(flushTimer);
        }
        // Retract on shutdown so a reloaded window does not leave the index
        // serving text from a buffer that no longer exists.
        void overlay.dispose();
      },
    }
  );

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
