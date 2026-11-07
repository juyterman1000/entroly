import { ExecFileException, execFile } from 'child_process';
import { randomBytes } from 'crypto';
import { promises as fs } from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';
import {
  CompressReceipt,
  compressionErrorHtml,
  compressionHtml,
  compressionPacket,
  isCopyRequest,
  lastLine,
  parseCompressReceipt,
} from './compressionHtml';

/** The receipt of this window's last successful run, for the receipt view. */
export let lastCompression: { receipt: CompressReceipt; fileName: string } | undefined;

interface CliResult {
  stdout: string;
  stderr: string;
  error: ExecFileException | null;
}

function runCompress(source: string, out: string, cwd: string | undefined): Promise<CliResult> {
  return new Promise((resolve) => {
    execFile(
      'entroly',
      ['compress', source, '--out', out, '--json'],
      {
        cwd,
        timeout: 120_000,
        windowsHide: true,
        maxBuffer: 16 * 1024 * 1024,
        env: { ...process.env, ENTROLY_NO_DOCKER: '1' },
      },
      (error, stdout, stderr) =>
        resolve({ stdout: String(stdout), stderr: String(stderr), error })
    );
  });
}

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
      void CompressionViewPanel.currentPanel.update(rawText, fileName);
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

    void this.update(rawText, fileName);
  }

  /**
   * Compress what the editor shows with the local CLI and render its receipt.
   *
   * This view used to compute a fixed 72% "saving" from the text length, copy
   * the first 1,200 characters under a made-up "Merkle Receipt ... (Verified
   * 0-1 Knapsack)" line, and claim the rest was "omitted with verifiable
   * recovery handles". None of that was measured or recoverable. The text is
   * now written to a temporary file (so a selection or an unsaved buffer is
   * compressed as shown) and passed to `entroly compress`; everything the
   * panel displays or copies comes from that run.
   */
  public async update(rawText: string, fileName: string): Promise<void> {
    this._packet = '';
    const label = path.basename(fileName) || 'untitled';
    if (!vscode.workspace.getConfiguration('entroly').get<boolean>('useCliIfAvailable', true)) {
      this._panel.webview.html = compressionErrorHtml(
        label,
        'Compression is turned off: the entroly.useCliIfAvailable setting is false.'
      );
      return;
    }

    const workDir = await fs.mkdtemp(path.join(os.tmpdir(), 'entroly-compress-'));
    // Keep the extension: it picks the codec (code skeleton, prose, data...).
    const source = path.join(workDir, `input${path.extname(fileName)}`);
    const out = path.join(workDir, `output${path.extname(fileName)}`);
    try {
      await fs.writeFile(source, rawText, 'utf8');
      const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
      const result = await runCompress(source, out, cwd);
      const receipt = result.error ? null : parseCompressReceipt(result.stdout);
      if (receipt === null) {
        const reason =
          result.error?.code === 'ENOENT'
            ? 'The entroly CLI was not found on PATH.'
            : lastLine(result.stderr) ||
              lastLine(result.stdout) ||
              (result.error ? result.error.message : 'entroly compress returned no receipt.');
        this._panel.webview.html = compressionErrorHtml(label, reason);
        return;
      }
      const compressed = await fs.readFile(out, 'utf8');
      this._packet = compressionPacket(compressed, label, receipt);
      lastCompression = { receipt, fileName: label };
      this._panel.webview.html = compressionHtml({
        fileName: label,
        receipt,
        packet: this._packet,
        nonce: randomBytes(16).toString('base64'),
      });
    } catch (error) {
      this._panel.webview.html = compressionErrorHtml(
        label,
        error instanceof Error ? error.message : String(error)
      );
    } finally {
      await fs.rm(workDir, { recursive: true, force: true });
    }
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
}
