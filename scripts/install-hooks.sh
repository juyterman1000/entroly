#!/usr/bin/env bash
# install-hooks.sh — symlink (or copy on Windows) repo-tracked git hooks
# into .git/hooks/. Run once per fresh clone:
#
#     bash scripts/install-hooks.sh
#
# Idempotent. Safe to re-run after `git pull` brings new hooks.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/scripts/git-hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

if [ ! -d "$HOOKS_SRC" ]; then
  echo "ERROR: $HOOKS_SRC not found" >&2
  exit 1
fi

mkdir -p "$HOOKS_DST"

installed=0
for src in "$HOOKS_SRC"/*; do
  [ -f "$src" ] || continue
  name="$(basename "$src")"
  dst="$HOOKS_DST/$name"

  # Windows / Git Bash: symlinks unreliable, copy + chmod instead
  if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "cygwin" ] || [ "$OSTYPE" = "win32" ]; then
    cp -f "$src" "$dst"
  else
    ln -sf "$src" "$dst"
  fi
  chmod +x "$dst" 2>/dev/null || true
  echo "  installed: $name"
  installed=$((installed + 1))
done

echo ""
echo "✓ $installed git hook(s) installed into $HOOKS_DST"
echo ""
echo "Hooks will now run automatically on the matching git operation."
echo "Bypass any hook (when you really need to) with --no-verify."
