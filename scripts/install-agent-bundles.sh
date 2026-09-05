#!/usr/bin/env sh
set -eu

action="install"
agent="all"
force="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    install|status|uninstall) action="$1" ;;
    --agent) shift; agent="${1:?missing agent}" ;;
    --force) force="true" ;;
    *) echo "usage: $0 [install|status|uninstall] [--agent all|codex|claude|gemini] [--force]" >&2; exit 2 ;;
  esac
  shift
done

case "$agent" in all|codex|claude|gemini) ;; *) echo "invalid agent: $agent" >&2; exit 2 ;; esac
command -v entroly >/dev/null 2>&1 || { echo "entroly is not on PATH" >&2; exit 1; }

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
user_root=${HOME:?HOME is required}
codex_root=${CODEX_HOME:-"$user_root/.codex"}

is_bundle() {
  [ -f "$1/entroly-bundle.json" ] && grep -q '"id"[[:space:]]*:[[:space:]]*"entroly"' "$1/entroly-bundle.json"
}

operate() {
  target_agent=$1
  source=$2
  destination=$3
  case "$destination" in "$user_root"/*) ;; *) echo "refusing destination outside user profile: $destination" >&2; exit 1 ;; esac

  if [ "$action" = "status" ]; then
    if is_bundle "$destination"; then state="installed"; elif [ -e "$destination" ]; then state="occupied-by-other-content"; else state="not-installed"; fi
    echo "$target_agent: $state ($destination)"
    return
  fi

  stamp=$(date -u +%Y%m%d%H%M%S)
  if [ "$action" = "uninstall" ]; then
    [ -e "$destination" ] || { echo "$target_agent: not installed"; return; }
    is_bundle "$destination" || { echo "refusing to move unrecognized content at $destination" >&2; exit 1; }
    disabled="$destination.entroly-disabled-$stamp"
    mv -- "$destination" "$disabled"
    echo "$target_agent: disabled; recoverable at $disabled"
    return
  fi

  is_bundle "$source" || { echo "invalid Entroly bundle source: $source" >&2; exit 1; }
  if [ -e "$destination" ]; then
    [ "$force" = "true" ] || { echo "destination exists: $destination; re-run with --force" >&2; exit 1; }
    backup="$destination.entroly-backup-$stamp"
    mv -- "$destination" "$backup"
    echo "$target_agent: backed up existing directory to $backup"
  fi
  mkdir -p -- "$(dirname -- "$destination")"
  cp -R -- "$source" "$destination"
  echo "$target_agent: installed at $destination"
}

if [ "$agent" = "all" ] || [ "$agent" = "codex" ]; then operate codex "$repo_root/integrations/codex/entroly/skills/entroly-evidence-operations" "$codex_root/skills/entroly-evidence-operations"; fi
if [ "$agent" = "all" ] || [ "$agent" = "claude" ]; then operate claude "$repo_root/skills/entroly-evidence-operations" "$user_root/.claude/skills/entroly-evidence-operations"; fi
if [ "$agent" = "all" ] || [ "$agent" = "gemini" ]; then operate gemini "$repo_root/integrations/gemini/entroly" "$user_root/.gemini/extensions/entroly"; fi

if [ "$action" = "install" ]; then echo "Restart the selected agent if it does not reload skills or extensions live."; fi
