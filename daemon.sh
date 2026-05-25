#!/usr/bin/env bash
# Start a gated Codex/tmux self-improvement loop for Entroly's own
# daemon/self-optimization behavior.

set -euo pipefail

SESSION="${SESSION:-daemon}"
WORKERS="${WORKERS:-3}"
CODEX_BOOT_SLEEP="${CODEX_BOOT_SLEEP:-5}"
CONTINUOUS_WORKERS="${CONTINUOUS_WORKERS:-0}"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
TASK_ROOT="$ROOT/tasks"
LOG_ROOT="$ROOT/logs"
TELEMETRY_ROOT="$ROOT/telemetry"
PUSH_GATE_ROOT="$TASK_ROOT/push-gate"
PROMPT_ROOT="$TASK_ROOT/prompts"
CODEX_CMD="${CODEX_CMD:-codex}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.4-mini}"
CODEX_AGENT_HOME="${CODEX_AGENT_HOME:-}"
if [ -z "$CODEX_AGENT_HOME" ]; then
  if [ -f "$HOME/.codex/auth.json" ]; then
    CODEX_AGENT_HOME="$HOME"
  elif [[ "$ROOT" =~ ^(/mnt/[^/]+/Users/[^/]+)(/|$) ]]; then
    CODEX_AGENT_HOME="${BASH_REMATCH[1]}"
  else
    CODEX_AGENT_HOME="$HOME"
  fi
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

resolve_codex() {
  if command -v "$CODEX_CMD" >/dev/null 2>&1 && HOME="$CODEX_AGENT_HOME" "$CODEX_CMD" --version >/dev/null 2>&1; then
    return
  fi

  local local_codex="$ROOT/.tmp/codex-cli-linux/node_modules/.bin/codex"
  if [ -x "$local_codex" ] && HOME="$CODEX_AGENT_HOME" "$local_codex" --version >/dev/null 2>&1; then
    CODEX_CMD="$local_codex"
    return
  fi

  die "codex command is not runnable: $CODEX_CMD. Install a Linux Codex CLI or run: npm install --prefix .tmp/codex-cli-linux @openai/codex"
}

ensure_runtime_dirs() {
  mkdir -p \
    "$TASK_ROOT/proposed" \
    "$TASK_ROOT/approved" \
    "$TASK_ROOT/claimed" \
    "$TASK_ROOT/done" \
    "$TASK_ROOT/rejected" \
    "$TASK_ROOT/failed" \
    "$PUSH_GATE_ROOT/pending" \
    "$PUSH_GATE_ROOT/approved" \
    "$PUSH_GATE_ROOT/rejected" \
    "$PROMPT_ROOT" \
    "$LOG_ROOT" \
    "$TELEMETRY_ROOT"
}

install_wsl_git_eol_guard() {
  if [[ "$ROOT" != /mnt/* ]] || [ ! -d "$ROOT/.git/info" ]; then
    return
  fi

  local attributes="$ROOT/.git/info/attributes"
  local marker="# codex-daemon: WSL CRLF checkout guard"
  if [ -f "$attributes" ] && grep -Fq "$marker" "$attributes"; then
    return
  fi

  {
    printf '\n%s\n' "$marker"
    printf '* text=auto eol=crlf\n'
  } >>"$attributes"
}

start_codex_pane() {
  local pane="$1"
  local title="$2"
  local prompt="$3"
  local prompt_file="$PROMPT_ROOT/$title.md"
  local startup_log="$LOG_ROOT/startup-$title.log"
  local quoted_root
  local quoted_codex
  local quoted_home
  local quoted_model
  local quoted_prompt_file

  tmux select-pane -t "$pane" -T "$title" >/dev/null 2>&1 || true
  printf '%s\n' "$prompt" >"$prompt_file"
  quoted_root="$(printf "%q" "$ROOT")"
  quoted_codex="$(printf "%q" "$CODEX_CMD")"
  quoted_home="$(printf "%q" "$CODEX_AGENT_HOME")"
  quoted_model="$(printf "%q" "$CODEX_MODEL")"
  quoted_prompt_file="$(printf "%q" "$prompt_file")"
  tmux send-keys -t "$pane" "cd $quoted_root && HOME=$quoted_home $quoted_codex -m $quoted_model \"\$(cat $quoted_prompt_file)\"; echo '[codex exited]'; sleep 3600" C-m
  sleep "$CODEX_BOOT_SLEEP"
  tmux capture-pane -pt "$pane" -S -80 >"$startup_log" 2>&1 || true
  if ! grep -Eq "OpenAI Codex|model:|gpt-" "$startup_log"; then
    die "Codex did not start cleanly in pane '$title'. See $startup_log"
  fi
}

planner_prompt() {
  cat <<PROMPT
You are the SYSTEM PLANNER for Entroly self-improvement.

Repository root: $ROOT
Task proposal queue: $TASK_ROOT/proposed

Your job:
- Analyze the repository for small improvements to Entroly's daemon self-optimization behavior.
- Focus only on daemon learning cadence, feedback journaling, dreaming-loop observability, control API learning stats, worker lifecycle safety, or focused tests for those behaviors.
- Write proposed tasks as Markdown files into $TASK_ROOT/proposed.

Rules:
- Do not modify code.
- Do not approve your own tasks.
- Never propose deployment, security, auth, packaging, release, or broad refactor work.
- Each task must be independently executable in one small PR.
- Each task must name target files/modules, expected behavior change, focused tests, max changed files, and rollback risk.
- Each task must include a security/legal/compliance impact section, even when the answer is "none expected".
- Avoid duplicate tasks already present in proposed, approved, claimed, done, rejected, or failed.
- If no meaningful task exists, write nothing and sleep before checking again.
PROMPT
}

architect_prompt() {
  cat <<PROMPT
You are the SENIOR ARCHITECT GATE for Entroly self-improvement.

Repository root: $ROOT
Proposed queue: $TASK_ROOT/proposed
Approved queue: $TASK_ROOT/approved
Rejected queue: $TASK_ROOT/rejected
Pre-push gate pending: $PUSH_GATE_ROOT/pending
Pre-push gate approved: $PUSH_GATE_ROOT/approved
Pre-push gate rejected: $PUSH_GATE_ROOT/rejected

Your job:
- Review tasks in $TASK_ROOT/proposed.
- Approve only small, isolated, measurable improvements to daemon self-optimization behavior.
- Move approved tasks to $TASK_ROOT/approved.
- Move rejected tasks to $TASK_ROOT/rejected and append a short rejection reason.
- Review every pre-push request in $PUSH_GATE_ROOT/pending.
- Move pre-push requests to $PUSH_GATE_ROOT/approved only when security, legal, compliance, tests, and scope checks are complete.
- Move unsafe or incomplete pre-push requests to $PUSH_GATE_ROOT/rejected and append a short rejection reason.

Approval criteria:
- Scope is limited to daemon learning cadence, feedback journaling, dreaming-loop observability, control API learning stats, worker lifecycle safety, or focused tests for those behaviors.
- Task names exact target files/modules.
- Task states the expected behavior change.
- Task lists focused tests to run.
- Task caps changed files at 3 or fewer.
- Task includes rollback risk.
- Task includes a security/legal/compliance impact section.

Reject:
- Vague tasks, broad refactors, no-op cleanup, dependency churn, deployment/security/auth/infra changes, release work, or anything likely to touch unrelated dirty files.

Mandatory pre-push gate checklist:
- Confirm the branch is not main and is based on current origin/main.
- Inspect changed files with git diff --name-only origin/main...HEAD and reject unrelated files.
- Confirm there are no secret, credential, token, key, certificate, or private-key additions.
- Confirm there are no dependency, license, NOTICE, SECURITY.md, PRIVACY.md, legal, compliance, data-retention, telemetry-consent, auth, or deployment changes unless the approved task explicitly required them.
- Confirm focused tests and lint/check commands passed and are listed in the request.
- Confirm git diff --check passes.
- Confirm PR text will disclose any user-visible behavior, data handling, or telemetry impact.

Do not modify code. Workers must consume only approved tasks.
Workers must not push until their matching pre-push request is in $PUSH_GATE_ROOT/approved.
PROMPT
}

worker_prompt() {
  local worker_id="$1"
  cat <<PROMPT
You are WORKER AGENT $worker_id for Entroly self-improvement.

Repository root: $ROOT
Approved queue: $TASK_ROOT/approved
Claimed queue: $TASK_ROOT/claimed
Done queue: $TASK_ROOT/done
Failed queue: $TASK_ROOT/failed
Pre-push gate pending: $PUSH_GATE_ROOT/pending
Pre-push gate approved: $PUSH_GATE_ROOT/approved
Pre-push gate rejected: $PUSH_GATE_ROOT/rejected
Log path: $LOG_ROOT/worker-$worker_id.log
Continuous mode: $CONTINUOUS_WORKERS

Loop:
1. Pick exactly one task from $TASK_ROOT/approved.
2. Atomically claim it by moving it into $TASK_ROOT/claimed with your worker id in the filename.
3. Before editing, run git status and git fetch origin main. If the worktree is dirty, do not edit; write a blocker note to $LOG_ROOT/worker-$worker_id.log and wait.
4. Create a feature branch from current origin/main named agent/self-improvement-$worker_id-<short-slug>.
5. Implement only the approved task. Do not touch unrelated files.
6. Run the focused tests named by the task, plus any directly relevant lint/check command.
7. Commit with a concise message only if tests pass.
8. Before pushing, create a Markdown pre-push request in $PUSH_GATE_ROOT/pending named <branch-slug>.md.
9. The pre-push request must include branch name, task file, changed files, git diff --stat, exact tests run, git diff --check result, secret-scan result, and legal/compliance/security notes.
10. Wait until the senior architect moves the matching request to $PUSH_GATE_ROOT/approved. If it appears in $PUSH_GATE_ROOT/rejected, fix the issue or move the task to failed with the reason.
11. Only after pre-push approval, push the branch and open a draft PR with gh if available. If gh is unavailable, push the branch and write the manual PR note to the task file.
12. Move successful tasks to $TASK_ROOT/done. Move failed tasks to $TASK_ROOT/failed with the failing command and short reason.
13. If Continuous mode is 0, stop after one completed or failed task. If no approved task exists, sleep and check again.

Required pre-push checks before writing the gate request:
- git status --short --branch
- git fetch origin main
- git diff --check
- git diff --name-only origin/main...HEAD
- git diff --stat origin/main...HEAD
- Search changed files for secrets or credentials, including api_key, secret, token, password, private key, sk-, ghp_, and certificate material.
- Verify no dependency, license, NOTICE, SECURITY.md, PRIVACY.md, legal, compliance, data-retention, telemetry-consent, auth, deployment, or release files changed unless the approved task explicitly required it.

Never:
- Push directly to main.
- Push without an approved pre-push gate request.
- Disable or skip tests to make a change pass.
- Modify deployment, security, auth, packaging, release, or broad architecture files.
- Work from $TASK_ROOT/proposed.
- Create artificial/no-op commits.
PROMPT
}

evaluator_prompt() {
  cat <<PROMPT
You are the EVALUATOR AGENT for Entroly self-improvement.

Repository root: $ROOT
Task root: $TASK_ROOT
Log root: $LOG_ROOT

Responsibilities:
- Review open PRs from branches named agent/self-improvement-*.
- Detect regressions, excessive scope, missing tests, and architectural drift.
- Check that every pushed branch had an approved pre-push gate request under $PUSH_GATE_ROOT/approved.
- Approve only measurable, small improvements to daemon self-optimization behavior.
- Reject unsafe changes and write concise findings to $LOG_ROOT/evaluator.log.

Focus:
- Stability.
- Simplicity.
- Maintainability.
- Test coverage.

Do not modify code.
PROMPT
}

telemetry_prompt() {
  cat <<PROMPT
You are the TELEMETRY AGENT for Entroly self-improvement.

Repository root: $ROOT
Log root: $LOG_ROOT
Telemetry root: $TELEMETRY_ROOT

Tasks:
- Monitor $LOG_ROOT and task queues for repeated failures, flaky tests, stuck tasks, or performance regressions.
- Write concise findings into $TELEMETRY_ROOT/*.md.
- Prefer concrete evidence: command, task file, branch, timestamp, and failure pattern.

Do not modify code.
PROMPT
}

start_session() {
  need tmux
  resolve_codex
  need git
  ensure_runtime_dirs
  install_wsl_git_eol_guard

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    die "tmux session '$SESSION' already exists. Use '$0 restart' or '$0 attach'."
  fi

  local created=0
  cleanup_on_error() {
    if [ "$created" -eq 1 ]; then
      tmux kill-session -t "$SESSION" 2>/dev/null || true
    fi
  }
  trap cleanup_on_error ERR EXIT

  tmux new-session -d -s "$SESSION" -n planner -c "$ROOT"
  created=1
  local planner
  planner="$(tmux display-message -p -t "$SESSION:planner.0" '#{pane_id}')"

  local architect
  architect="$(tmux new-window -P -F '#{pane_id}' -t "$SESSION:" -n architect -c "$ROOT")"

  local worker_panes=()
  local i
  for i in $(seq 1 "$WORKERS"); do
    worker_panes+=("$(tmux new-window -P -F '#{pane_id}' -t "$SESSION:" -n "worker-$i" -c "$ROOT")")
  done

  local evaluator
  evaluator="$(tmux new-window -P -F '#{pane_id}' -t "$SESSION:" -n evaluator -c "$ROOT")"

  local telemetry
  telemetry="$(tmux new-window -P -F '#{pane_id}' -t "$SESSION:" -n telemetry -c "$ROOT")"

  start_codex_pane "$planner" "planner" "$(planner_prompt)"
  start_codex_pane "$architect" "architect-gate" "$(architect_prompt)"

  for i in "${!worker_panes[@]}"; do
    start_codex_pane "${worker_panes[$i]}" "worker-$((i + 1))" "$(worker_prompt "$((i + 1))")"
  done

  start_codex_pane "$evaluator" "evaluator" "$(evaluator_prompt)"
  start_codex_pane "$telemetry" "telemetry" "$(telemetry_prompt)"
  trap - ERR EXIT
  created=0

  echo ""
  echo "=================================="
  echo " Gated self-improvement daemon started"
  echo "=================================="
  echo ""
  echo "Session:   $SESSION"
  echo "Root:      $ROOT"
  echo "Tasks:     $TASK_ROOT"
  echo "Logs:      $LOG_ROOT"
  echo "Telemetry: $TELEMETRY_ROOT"
  echo ""
  echo "Attach:    tmux attach -t $SESSION"
  echo "Stop:      $0 stop"
}

stop_session() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Stopped tmux session '$SESSION'."
  else
    echo "No tmux session '$SESSION' is running."
  fi
}

status_session() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' is running."
    tmux list-panes -a -F '#{window_index}:#{pane_index} #{window_name} #{pane_current_command}'
  else
    echo "tmux session '$SESSION' is not running."
  fi
}

case "${1:-start}" in
  start)
    start_session
    ;;
  stop)
    need tmux
    stop_session
    ;;
  restart)
    need tmux
    stop_session
    start_session
    ;;
  status)
    need tmux
    status_session
    ;;
  attach)
    need tmux
    exec tmux attach -t "$SESSION"
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status|attach]" >&2
    exit 2
    ;;
esac
