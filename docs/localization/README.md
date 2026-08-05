# Entroly localization and multilingual discovery

Entroly should be discoverable beyond English, but inaccurate machine-translated
security, installation, or benchmark claims would be worse than no translation.
This directory defines the publication contract for localized documentation.

## Initial languages

Prioritize:

1. Simplified Chinese (`zh-CN`) for MCP and AI-agent developer communities.
2. Japanese (`ja`) for local tooling and coding-agent users.
3. Spanish (`es`) for broad developer accessibility.
4. Brazilian Portuguese (`pt-BR`) for the Latin American open-source ecosystem.

Additional languages should follow demonstrated user demand and available expert
review rather than list-count goals.

## Pages to localize first

Localize short, high-value pages before translating the full documentation:

- product definition and best-fit/weak-fit guidance;
- installation and clean uninstall;
- local-versus-provider data boundary;
- MCP and Claude Code setup;
- exact recovery and Context Receipts;
- limitations and non-guarantees;
- how to report an integration or reproduction.

Do not begin with benchmark marketing. Readers must understand the protocol and
limitations before workload-specific numbers.

## Publication states

- **draft:** translation exists but has not received technical and language review;
- **reviewed:** one language reviewer and one Entroly technical reviewer approved
  the exact revision;
- **published:** the reviewed file is linked from the documentation and sitemap;
- **stale:** the English source changed materially after the translation review;
- **retired:** no maintainer can keep the translation safe and current.

Draft pages must not be indexed or advertised as current documentation.

## Required metadata

Every localized page must state:

- language and locale;
- translated source path and source commit;
- Entroly version reviewed;
- translation status;
- language reviewer;
- technical reviewer;
- review date;
- link to the current English source;
- warning when the English source is newer.

## Translation rules

Keep these product terms in English on first use, followed by a concise local
explanation where useful:

- Entroly
- Context Assurance
- Context Receipt
- MCP / Model Context Protocol
- WITNESS
- content-addressed recovery

Do not translate package names, commands, environment variables, file paths,
model IDs, JSON keys, API fields, error messages, or cryptographic handles.

Do not strengthen claims during translation. In particular, preserve:

- workload and token-budget caveats;
- distinction between provider-observed usage and local estimates;
- proxy provider boundary;
- pass-through behavior;
- limitations of evidence verification;
- absence of universal savings or quality guarantees.

## Review checklist

- [ ] All commands match the current release.
- [ ] Package names and URLs are unchanged.
- [ ] Privacy and provider boundaries match `PRIVACY.md`.
- [ ] Limitations match `docs/limitations.md`.
- [ ] Benchmark language names the workload, version, model, and caveats.
- [ ] No sentence implies universal superiority or guaranteed savings.
- [ ] A fluent reviewer confirms natural technical language.
- [ ] A technical reviewer confirms product accuracy.
- [ ] The source commit and reviewed Entroly version are recorded.
- [ ] The page is added to the sitemap only after review.

## Contribution process

Open an issue using the integration request form and select documentation or
curated list. State the locale, pages, reviewer availability, and maintenance
plan. A translation PR should contain one locale and one coherent page set so
review remains bounded.

Automated translation may be used as a draft aid, but cannot satisfy the review
requirement by itself.
