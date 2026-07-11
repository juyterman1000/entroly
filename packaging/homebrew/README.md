# Homebrew Tap for Entroly

The formula in this directory is the canonical source. To make `brew tap juyterman1000/entroly && brew install entroly` work, the formula must live in a separate repo named `juyterman1000/homebrew-entroly`.

Current release example version: `1.0.47`.

## Release checklist

1. Copy `packaging/homebrew/entroly.rb` into `juyterman1000/homebrew-entroly/Formula/entroly.rb`.
2. Set `VER=1.0.47` when preparing this release.
3. Download the matching PyPI sdist: `entroly-1.0.47.tar.gz`.
4. Recalculate the `sha256` for that tarball before publishing the Homebrew tap update.
5. Verify locally before pushing with `brew install --build-from-source ./Formula/entroly.rb`, `brew test entroly`, and `brew audit --new --strict entroly`.

You can also automate the tap update with a GitHub Action that watches `juyterman1000/entroly` releases and opens a PR in `homebrew-entroly` with the bumped formula.
