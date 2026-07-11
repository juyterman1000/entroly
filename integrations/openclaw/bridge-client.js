import { spawn } from "node:child_process";
import readline from "node:readline";

export class EntrolyBridgeClient {
  constructor({
    pythonCommand = "python",
    timeoutMs = 5000,
    logger = console,
    spawnProcess = spawn,
  } = {}) {
    this.pythonCommand = pythonCommand;
    this.timeoutMs = timeoutMs;
    this.logger = logger;
    this.spawnProcess = spawnProcess;
    this.nextId = 1;
    this.pending = new Map();
    this.process = undefined;
  }

  start() {
    if (this.process && !this.process.killed) {
      return this.process;
    }
    const child = this.spawnProcess(
      this.pythonCommand,
      ["-m", "entroly.openclaw_bridge", "--jsonl"],
      { stdio: ["pipe", "pipe", "pipe"], windowsHide: true },
    );
    this.process = child;
    const lines = readline.createInterface({ input: child.stdout });
    lines.on("line", (line) => this.#onLine(line));
    child.stderr.on("data", (chunk) => {
      const message = String(chunk).trim();
      if (message) this.logger.warn?.(`entroly bridge: ${message}`);
    });
    child.on("error", (error) => this.#handleChildFailure(child, error));
    child.on("exit", (code, signal) => {
      this.#handleChildFailure(
        child,
        new Error(`Entroly bridge exited (${code ?? signal ?? "unknown"})`),
      );
    });
    return child;
  }

  request(payload) {
    const child = this.start();
    const requestId = String(this.nextId++);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#terminate(
          child,
          new Error(`Entroly bridge timed out after ${this.timeoutMs}ms`),
        );
      }, this.timeoutMs);
      this.pending.set(requestId, { resolve, reject, timer });
      child.stdin.write(`${JSON.stringify({ ...payload, request_id: requestId })}\n`, (error) => {
        if (!error) return;
        this.#terminate(child, error);
      });
    });
  }

  health() {
    return this.request({ operation: "health" });
  }

  #onLine(line) {
    let response;
    try {
      response = JSON.parse(line);
    } catch (error) {
      this.logger.warn?.(`entroly bridge emitted invalid JSON: ${error.message}`);
      return;
    }
    const pending = this.pending.get(String(response.request_id));
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pending.delete(String(response.request_id));
    if (response.ok) pending.resolve(response);
    else pending.reject(new Error(response.error || "Entroly bridge request failed"));
  }

  #failAll(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  #handleChildFailure(child, error) {
    if (this.process !== child) return;
    this.process = undefined;
    this.#failAll(error);
  }

  #terminate(child, error) {
    if (this.process !== child) return;
    this.process = undefined;
    this.#failAll(error);
    child.stdin.destroy();
    if (!child.killed) child.kill();
  }

  async dispose() {
    const child = this.process;
    this.process = undefined;
    if (!child || child.killed) return;
    this.#failAll(new Error("Entroly bridge disposed"));
    child.stdin.end();
    // Brief grace for the process to exit on stdin EOF before SIGTERM.
    await new Promise((resolve) => {
      const timer = setTimeout(() => resolve(), 200);
      child.on("exit", () => { clearTimeout(timer); resolve(); });
    });
    if (!child.killed) child.kill();
  }
}
