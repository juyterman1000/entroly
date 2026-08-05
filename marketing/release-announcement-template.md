# Entroly release announcement template

Status: template, not a published release announcement.

Use this only after the coordinated release workflow verifies every package and
artifact referenced below. Remove sections that do not apply instead of filling
them with generic promotion.

## Headline

Entroly `<version>`: `<one concrete user-facing outcome>`

## What changed

Entroly `<version>` adds or changes:

- `<user-facing capability>` — `<why it matters>`
- `<integration or operational improvement>` — `<who benefits>`
- `<trust, recovery, or verification improvement>` — `<failure prevented>`

Avoid listing internal refactors unless they change reliability, compatibility,
security, or maintenance for users.

## Who should upgrade

Upgrade when you:

- `<affected workflow or host>`;
- `<benefit from new feature or fixed defect>`;
- `<need the new compatibility or security behavior>`.

Stay on the prior version temporarily when:

- `<known migration blocker>`;
- `<unsupported platform or dependency>`.

## Install or upgrade

```bash
pip install -U entroly==<version>
```

Optional verified channels:

```bash
npm install -g entroly@<version>
brew upgrade entroly
docker pull ghcr.io/juyterman1000/entroly:<version>
```

Include a channel only after its exact version is publicly available and a clean
install or pull has been verified.

## Verify locally

```bash
entroly doctor
entroly verify-claims
```

Add a feature-specific verification command when one exists.

## Evidence

For every quantitative statement, include:

- workload or fixture;
- baseline and treatment;
- model/provider when applicable;
- token or context budget;
- sample size and uncertainty when applicable;
- raw artifact or reproduction command;
- limitations and negative results.

Do not reuse a result from an older version without proving the current release
still reproduces it.

## Compatibility and migration

- Python: `<supported versions>`
- Rust/native: `<artifact and ABI notes>`
- npm/WASM: `<package compatibility>`
- MCP/plugin: `<manifest and host compatibility>`
- OpenClaw/other integrations: `<minimum host versions>`
- Configuration migration: `<exact action or none>`
- Rollback: `<tested rollback command>`

## Security and privacy

State whether the release changes:

- network behavior;
- provider-bound data;
- local persistence;
- ports or authentication;
- credentials or environment variables;
- MCP permissions;
- recovery-store format;
- verification fail-open/fail-closed behavior.

Write `No change` only after checking the final diff.

## Known limitations

- `<limitation or workload where feature passes through>`
- `<known issue or unsupported path>`
- `<measurement boundary>`

Do not hide a known regression because it weakens the launch message.

## Channel summaries

### GitHub release

Use the complete version with migration, evidence, checksums, and known limits.

### PyPI and npm

Use one paragraph focused on package users and link the complete release notes.

### Community post

Explain one meaningful technical problem and invite reproducible feedback.

### Newsletter pitch

Offer a technical angle and disclose maintainer affiliation. Do not request the
editor to repeat unverified metrics.

### Social post

Use one factual outcome, one verification command, and one link. Avoid threads
that fragment caveats away from the claim.

## Publication gate

- [ ] Main contains the exact release commit.
- [ ] Required CI passed on the unchanged release head.
- [ ] GitHub release and tag resolve to the same commit.
- [ ] PyPI package is public and installs cleanly.
- [ ] npm packages are public and load correctly.
- [ ] MCP, plugin, OpenClaw, Docker, binary, and Homebrew channels referenced in
      the announcement are public and verified.
- [ ] Version is synchronized in citation metadata.
- [ ] Documentation and limitations describe the released behavior.
- [ ] Every quantitative claim links current evidence.
- [ ] Rollback and uninstall instructions were checked.
- [ ] Public URLs are recorded in the distribution registry where applicable.
