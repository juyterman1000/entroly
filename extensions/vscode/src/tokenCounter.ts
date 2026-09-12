import * as vscode from 'vscode';

export function estimateTokens(text: string): number {
  if (!text) return 0;
  const words = text.match(/[A-Za-z0-9_]+|[^\s\w]/g);
  return words ? Math.max(1, Math.ceil(words.length * 1.15)) : 1;
}

export class TokenStatusBarManager {
  private statusBarItem: vscode.StatusBarItem;
  private disposables: vscode.Disposable[] = [];

  constructor() {
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.statusBarItem.command = 'entroly.compressFile';
    this.update();

    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor(() => this.update()),
      vscode.window.onDidChangeTextEditorSelection(() => this.update()),
      vscode.workspace.onDidChangeTextDocument((e) => {
        if (vscode.window.activeTextEditor?.document === e.document) {
          this.update();
        }
      })
    );
  }

  public update(): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      this.statusBarItem.hide();
      return;
    }

    const config = vscode.workspace.getConfiguration('entroly');
    if (!config.get<boolean>('showStatusBarTokens', true)) {
      this.statusBarItem.hide();
      return;
    }

    const selection = editor.selection;
    let text = '';
    let isSelection = false;

    if (!selection.isEmpty) {
      text = editor.document.getText(selection);
      isSelection = true;
    } else {
      text = editor.document.getText();
    }

    const tokens = estimateTokens(text);
    const budget = config.get<number>('tokenBudget', 450);

    if (isSelection) {
      this.statusBarItem.text = `$(zap) ${tokens.toLocaleString()} sel tok | Entroly`;
      this.statusBarItem.tooltip = `Selected: ${tokens.toLocaleString()} tokens (~${text.length} chars).\nClick to compress selection into Entroly context.`;
    } else {
      this.statusBarItem.text = `$(zap) ${tokens.toLocaleString()} tok | Entroly`;
      this.statusBarItem.tooltip = `File: ${tokens.toLocaleString()} tokens (Target Budget: ${budget}).\nClick to compress into prompt-ready context.`;
    }

    this.statusBarItem.show();
  }

  public dispose(): void {
    this.statusBarItem.dispose();
    this.disposables.forEach((d) => d.dispose());
  }
}
