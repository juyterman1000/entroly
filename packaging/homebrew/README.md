# Homebrew Tap for Entroly

The formula in this directory is the canonical source. To make
`brew tap juyterman1000/entroly && brew install entroly` work, the formula
must live in a separate repo named `juyterman1000/homebrew-entroly`.

## One-time setup

```bash
# 1. Create the tap repo on GitHub:
gh repo create juyterman1000/homebrew-entroly --public \
  --description "Homebrew tap for Entroly"

# 2. Clone it locally and seed it:
gh repo clone juyterman1000/homebrew-entroly /tmp/homebrew-entroly
mkdir -p /tmp/homebrew-entroly/Formula
cp packaging/homebrew/entroly.rb /tmp/homebrew-entroly/Formula/

# 3. Fill in the sha256 (after publishing to PyPI):
VER=0.19.3
curl -sLO "https://files.pythonhosted.org/packages/source/e/entroly/entroly-${VER}.tar.gz"
SHA=$(shasum -a 256 "entroly-${VER}.tar.gz" | awk '{print $1}')
sed -i.bak "s/REPLACE_WITH_SDIST_SHA256_AT_RELEASE_TIME/${SHA}/" \
  /tmp/homebrew-entroly/Formula/entroly.rb

# 4. Push:
cd /tmp/homebrew-entroly
git add Formula/entroly.rb
git commit -m "entroly ${VER}"
git push origin main
```

## On every release

Either bump manually:

```bash
VER=0.19.3
sed -i.bak "s|entroly-[0-9.]*\.tar\.gz|entroly-${VER}.tar.gz|" Formula/entroly.rb
curl -sLO "https://files.pythonhosted.org/packages/source/e/entroly/entroly-${VER}.tar.gz"
SHA=$(shasum -a 256 "entroly-${VER}.tar.gz" | awk '{print $1}')
sed -i.bak "s|sha256 \".*\"|sha256 \"${SHA}\"|" Formula/entroly.rb
git commit -am "entroly ${VER}" && git push
```

Or automate with a GitHub Action that watches `juyterman1000/entroly`
releases and opens a PR in `homebrew-entroly` with the bumped formula.
[`mislav/bump-homebrew-formula-action`](https://github.com/mislav/bump-homebrew-formula-action)
does this in ~10 lines.

## Verify locally before pushing

```bash
brew install --build-from-source ./Formula/entroly.rb
brew test entroly
brew audit --new --strict entroly
```
