#!/usr/bin/env sh
# entroly-rs installer — downloads the prebuilt single-binary proxy.
#
#   curl -fsSL https://raw.githubusercontent.com/juyterman1000/entroly/main/install.sh | sh
#
# Zero dependencies, no Python. For Windows, download the .zip from the
# GitHub Releases page. For platforms without a prebuilt binary, build from
# source:  cargo install --git https://github.com/juyterman1000/entroly \
#            --bin entroly-rs --features proxy
set -eu

REPO="juyterman1000/entroly"
BIN="entroly-rs"

os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
  Linux)
    case "$arch" in
      x86_64 | amd64) target="x86_64-unknown-linux-gnu" ;;
      *) echo "entroly-rs: no prebuilt binary for Linux/$arch yet — build from source (see header)."; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$arch" in
      x86_64) target="x86_64-apple-darwin" ;;
      arm64 | aarch64) target="aarch64-apple-darwin" ;;
      *) echo "entroly-rs: unsupported macOS arch $arch"; exit 1 ;;
    esac
    ;;
  *)
    echo "entroly-rs: unsupported OS '$os' (Windows: download the .zip from the Releases page)."
    exit 1
    ;;
esac

asset="${BIN}-${target}.tar.gz"
url="https://github.com/${REPO}/releases/latest/download/${asset}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading ${asset} ..."
if ! curl -fsSL "$url" -o "${tmp}/${asset}"; then
  echo "entroly-rs: download failed ($url). Check the latest release has this asset."
  exit 1
fi
tar -xzf "${tmp}/${asset}" -C "$tmp"

# Prefer a user-writable bin dir; fall back to /usr/local/bin via sudo.
dir="${HOME}/.local/bin"
if mkdir -p "$dir" 2>/dev/null && install -m 0755 "${tmp}/${BIN}" "${dir}/${BIN}" 2>/dev/null; then
  :
else
  dir="/usr/local/bin"
  sudo install -m 0755 "${tmp}/${BIN}" "${dir}/${BIN}"
fi

echo "Installed ${BIN} -> ${dir}/${BIN}"
case ":${PATH}:" in
  *":${dir}:"*) ;;
  *) echo "Add to PATH:  export PATH=\"${dir}:\$PATH\"" ;;
esac
"${dir}/${BIN}" --version || true

echo
echo "Run the compressing proxy:"
echo "  ${BIN} proxy --upstream https://api.anthropic.com    # then point ANTHROPIC_BASE_URL at it"
