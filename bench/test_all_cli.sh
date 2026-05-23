#!/usr/bin/env bash
# Smoke test every entroly CLI subcommand. Two passes:
#   1. --help must exit 0
#   2. a safe functional invocation (read-only or bounded) must exit 0 or a
#      known "expected failure" (missing input, etc).
set -u
PASS=0; FAIL=0; SKIP=0
FAILED_CMDS=()
SKIPPED_CMDS=()

run() {
  local label="$1"; shift
  local expect="$1"; shift  # "pass" or "skip"
  local out rc
  out=$("$@" 2>&1)
  rc=$?
  if [[ "$expect" == "skip" ]]; then
    SKIP=$((SKIP+1))
    SKIPPED_CMDS+=("$label")
    printf "  SKIP  %-40s (by design)\n" "$label"
    return
  fi
  if [[ $rc -eq 0 ]]; then
    PASS=$((PASS+1))
    printf "  PASS  %-40s\n" "$label"
  else
    FAIL=$((FAIL+1))
    FAILED_CMDS+=("$label (rc=$rc)")
    printf "  FAIL  %-40s rc=%d\n    %s\n" "$label" "$rc" "$(echo "$out" | tail -3 | tr '\n' ' ')"
  fi
}

CMDS=(init serve dashboard health autotune go proxy optimize feedback benchmark status config telemetry clean export import drift profile batch demo wrap learn doctor digest migrate role completions compile verify sync search docs share finetune)

echo "=== PASS 1: --help for every subcommand ==="
for c in "${CMDS[@]}"; do
  run "$c --help" pass entroly "$c" --help >/dev/null 2>&1
  rc=$?  # real rc isn't captured — rerun properly
done
# above is broken; redo with proper capture
PASS=0; FAIL=0; FAILED_CMDS=()
for c in "${CMDS[@]}"; do
  if entroly "$c" --help >/dev/null 2>&1; then
    PASS=$((PASS+1))
    printf "  PASS  %s --help\n" "$c"
  else
    FAIL=$((FAIL+1))
    FAILED_CMDS+=("$c --help")
    printf "  FAIL  %s --help\n" "$c"
  fi
done

echo
echo "=== PASS 2: safe functional smoke ==="

# Read-only / safe invocations
run "status"              pass entroly status
run "config"              pass entroly config
run "telemetry --status"  pass entroly telemetry --status
run "doctor"              pass entroly doctor
run "drift"               pass entroly drift
run "profile list"        pass entroly profile list
run "role list"           pass entroly role list
run "completions bash"    pass entroly completions bash
run "digest"              pass entroly digest
run "share"               pass entroly share
run "search jaccard"      pass entroly search jaccard
run "verify"              pass entroly verify
run "health --quick"      pass entroly health --quick
run "docs"                pass entroly docs
run "sync"                pass entroly sync

# Stateful but reversible / quick
run "optimize --task q --budget 1024 --quiet"  pass entroly optimize --task "test" --budget 1024 --quiet
run "feedback --outcome success"               pass entroly feedback --outcome success
run "demo"                                      pass entroly demo

# Skipped: long-running / interactive / network
run "serve"      skip entroly serve
run "dashboard"  skip entroly dashboard
run "go"         skip entroly go
run "proxy"      skip entroly proxy
run "wrap"       skip entroly wrap
run "autotune"   skip entroly autotune
run "benchmark"  skip entroly benchmark
run "batch"      skip entroly batch
run "learn"      skip entroly learn
run "compile"    skip entroly compile
run "init"       skip entroly init
run "clean"      skip entroly clean
run "export"     skip entroly export
run "import"     skip entroly import
run "migrate"    skip entroly migrate
run "finetune"   skip entroly finetune

echo
echo "=== RESULTS ==="
echo "PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"
if [[ $FAIL -gt 0 ]]; then
  echo "Failed:"
  for f in "${FAILED_CMDS[@]}"; do echo "  - $f"; done
fi
exit $FAIL
