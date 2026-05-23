#!/usr/bin/env bash
# Smoke test all entroly CLI commands.
# Short-running ones are executed; long-running ones (proxy/serve/go/wrap/dashboard)
# are verified via --help only.

set -u
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export ENTROLY_TELEMETRY=0
TMPROOT=$(mktemp -d)
OUTDIR=$TMPROOT/out
mkdir -p "$OUTDIR"

pass=0; fail=0; skip=0
declare -a failed_cmds

run() {
    local name="$1"; shift
    local log="$OUTDIR/$name.log"
    timeout 30 "$@" >"$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "PASS  $name"
        pass=$((pass+1))
    elif [ $rc -eq 124 ]; then
        echo "TIMEOUT $name (>30s)"
        fail=$((fail+1))
        failed_cmds+=("$name (timeout)")
    else
        echo "FAIL  $name  (rc=$rc)"
        echo "      --- last 5 lines of stderr ---"
        tail -5 "$log" | sed 's/^/      /'
        fail=$((fail+1))
        failed_cmds+=("$name")
    fi
}

help_only() {
    local name="$1"
    run "${name}--help" entroly "$name" --help
}

# Short commands: actually run
run "init--dry-run" entroly init --dry-run
run "health" entroly health
run "status" entroly status
run "config" entroly config
run "doctor" entroly doctor
run "telemetry-status" entroly telemetry status
run "drift" entroly drift
run "profile-list" entroly profile list
run "role-list" entroly role list
run "migrate" entroly migrate
run "benchmark" entroly benchmark --budget 4096
run "demo" entroly demo
run "share" entroly share -o "$OUTDIR/report.html"
run "optimize" entroly optimize --task "auth" --budget 4096 --quiet
run "feedback" entroly feedback --outcome success
run "export" entroly export "$OUTDIR/export.json"
run "import" entroly import "$OUTDIR/export.json"
run "digest" entroly digest
run "completions-bash" entroly completions bash
run "completions-zsh" entroly completions zsh
run "completions-fish" entroly completions fish
run "search" entroly search foo
run "compile" entroly compile --max-files 20
run "verify" entroly verify
run "sync" entroly sync --max-files 20
run "docs" entroly docs --max-files 5
run "finetune" entroly finetune -o "$OUTDIR/train.jsonl"
echo "test" | run "batch" entroly batch --budget 4096

# Long-running: --help only
help_only serve
help_only proxy
help_only dashboard
help_only go
help_only wrap
help_only clean
help_only learn
help_only autotune

echo
echo "==== RESULTS ===="
echo "PASS:    $pass"
echo "FAIL:    $fail"
if [ $fail -gt 0 ]; then
    echo "Failed commands:"
    for c in "${failed_cmds[@]}"; do echo "  - $c"; done
fi
echo "Logs:    $OUTDIR"
