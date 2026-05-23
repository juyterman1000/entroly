#!/bin/sh
# Entroly universal installer.
#
# Usage:
#   curl -fsSL https://entroly.dev/install.sh | sh
#   curl -fsSL https://entroly.dev/install.sh | sh -s -- --extras full
#
# Detects Python 3.10+, picks pip or pipx, installs entroly, verifies, and
# prints next steps. POSIX sh — no bashisms.

set -eu

ENTROLY_PACKAGE="entroly"
ENTROLY_EXTRAS="full"
QUIET=0

# ── colors ────────────────────────────────────────────────────────────
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
    BOLD=$(tput bold)
    DIM=$(tput dim)
    GREEN=$(tput setaf 2)
    RED=$(tput setaf 1)
    YELLOW=$(tput setaf 3)
    CYAN=$(tput setaf 6)
    RESET=$(tput sgr0)
else
    BOLD=""; DIM=""; GREEN=""; RED=""; YELLOW=""; CYAN=""; RESET=""
fi

say()   { [ "$QUIET" -eq 1 ] || printf '%s\n' "$*"; }
ok()    { say "  ${GREEN}✓${RESET} $*"; }
warn()  { printf '%s\n' "  ${YELLOW}!${RESET} $*" >&2; }
fail()  { printf '%s\n' "  ${RED}✗${RESET} $*" >&2; exit 1; }

# ── parse args ────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --extras)         ENTROLY_EXTRAS="${2:-full}"; shift 2 ;;
        --extras=*)       ENTROLY_EXTRAS="${1#*=}"; shift ;;
        --no-extras)      ENTROLY_EXTRAS=""; shift ;;
        --package)        ENTROLY_PACKAGE="${2:-entroly}"; shift 2 ;;
        --quiet|-q)       QUIET=1; shift ;;
        -h|--help)
            cat <<EOF
Entroly installer

Usage:
  install.sh [--extras NAME] [--no-extras] [--package NAME] [--quiet]

Options:
  --extras NAME    pip extras to install (default: full)
  --no-extras      install entroly only, no extras
  --package NAME   pypi package name (default: entroly)
  --quiet, -q      suppress non-error output
EOF
            exit 0 ;;
        *) warn "unknown argument: $1"; shift ;;
    esac
done

# ── banner ────────────────────────────────────────────────────────────
say ""
say "  ${BOLD}${CYAN}Entroly${RESET}  ${DIM}— context compression + cost-cutting proxy${RESET}"
say "  ${DIM}─────────────────────────────────────────────────${RESET}"
say ""

# ── detect platform ───────────────────────────────────────────────────
OS=$(uname -s 2>/dev/null || echo unknown)
ARCH=$(uname -m 2>/dev/null || echo unknown)

case "$OS" in
    Linux*)  PLATFORM="linux" ;;
    Darwin*) PLATFORM="macos" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
    *)       PLATFORM="unknown" ;;
esac

ok "Platform: ${PLATFORM}/${ARCH}"

# ── detect python ─────────────────────────────────────────────────────
find_python() {
    for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
            case "$ver" in
                3.1[0-9]*|3.[2-9][0-9]*|[4-9].*) echo "$cand"; return 0 ;;
            esac
        fi
    done
    return 1
}

PYTHON=$(find_python || echo "")
if [ -z "$PYTHON" ]; then
    say ""
    fail "Python 3.10+ not found. Install one of:
        macOS:   brew install python@3.12
        Ubuntu:  sudo apt install python3.12 python3.12-venv
        Fedora:  sudo dnf install python3.12
        Windows: https://www.python.org/downloads/"
fi

PYVER=$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
ok "Python: ${PYTHON} (${PYVER})"

# ── pick installer (pipx > pip --user > pip) ──────────────────────────
INSTALLER=""
INSTALL_CMD=""

if command -v pipx >/dev/null 2>&1; then
    INSTALLER="pipx"
    if [ -n "$ENTROLY_EXTRAS" ]; then
        INSTALL_CMD="pipx install '${ENTROLY_PACKAGE}[${ENTROLY_EXTRAS}]'"
    else
        INSTALL_CMD="pipx install '${ENTROLY_PACKAGE}'"
    fi
elif "$PYTHON" -m pip --version >/dev/null 2>&1; then
    INSTALLER="pip"
    USER_FLAG="--user"
    # If we're in a venv, drop --user
    if "$PYTHON" -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' 2>/dev/null; then
        USER_FLAG=""
    fi
    if [ -n "$ENTROLY_EXTRAS" ]; then
        INSTALL_CMD="$PYTHON -m pip install $USER_FLAG --upgrade '${ENTROLY_PACKAGE}[${ENTROLY_EXTRAS}]'"
    else
        INSTALL_CMD="$PYTHON -m pip install $USER_FLAG --upgrade '${ENTROLY_PACKAGE}'"
    fi
else
    fail "Neither pipx nor pip is available. Install pip:
        $PYTHON -m ensurepip --upgrade"
fi

ok "Installer: ${INSTALLER}"
say ""
say "  ${DIM}Running: ${INSTALL_CMD}${RESET}"
say ""

# ── run install ───────────────────────────────────────────────────────
if ! sh -c "$INSTALL_CMD"; then
    say ""
    fail "Installation failed. Try with --no-extras for a minimal install:
        curl -fsSL https://entroly.dev/install.sh | sh -s -- --no-extras"
fi

# ── verify ────────────────────────────────────────────────────────────
say ""
if command -v entroly >/dev/null 2>&1; then
    INSTALLED_VER=$(entroly --version 2>/dev/null | head -n1 || echo "unknown")
    ok "Installed: ${BOLD}${INSTALLED_VER}${RESET}"
else
    warn "entroly is installed but not on PATH."
    case "$INSTALLER" in
        pipx)
            warn "Run: ${BOLD}pipx ensurepath${RESET} and restart your shell."
            ;;
        pip)
            BIN=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts"))' 2>/dev/null || echo "")
            warn "Add to PATH: ${BOLD}${BIN}${RESET}"
            ;;
    esac
fi

# ── next steps ────────────────────────────────────────────────────────
say ""
say "  ${BOLD}Next steps:${RESET}"
say ""
say "    ${CYAN}cd /path/to/your/repo${RESET}"
say "    ${CYAN}entroly go${RESET}                ${DIM}# auto-detect, start proxy + dashboard${RESET}"
say ""
say "  ${DIM}Or wrap your AI tool directly:${RESET}"
say "    ${CYAN}entroly wrap claude${RESET}       ${DIM}# Claude Code${RESET}"
say "    ${CYAN}entroly wrap cursor${RESET}       ${DIM}# Cursor${RESET}"
say "    ${CYAN}entroly wrap codex${RESET}        ${DIM}# Codex CLI${RESET}"
say ""
say "  ${DIM}Docs:    https://entroly.dev/docs${RESET}"
say "  ${DIM}Discord: https://discord.gg/entroly${RESET}"
say "  ${DIM}Issues:  https://github.com/juyterman1000/entroly/issues${RESET}"
say ""
