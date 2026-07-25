# Entroly Simple Mode

**Status:** product requirement; not yet a shipped capability.

## Product promise

A person should be able to reduce avoidable AI context usage without learning about tokens, MCP, proxies, JSON configuration, or agent architecture.

The desired experience is:

```text
Install Entroly
      ↓
Choose an AI app
      ↓
Turn on Context Assurance
      ↓
Use the app normally
      ↓
See measured usage, recoverable evidence, and honest savings status
```

Entroly must not claim that this experience already exists until the installer, supported-app connectors, and end-to-end verification are shipped.

## Intended users

Simple Mode is for people who use AI as a tool rather than build AI infrastructure, including:

- librarians researching, summarizing, and organizing documents;
- healthcare staff using approved AI tools for administrative or research workflows;
- cleaners and field-service workers using AI for scheduling, translation, customer messages, and instructions;
- students, teachers, small-business owners, writers, analysts, and office staff;
- individual developers who want savings without maintaining an agent stack.

Entroly does not provide medical advice or replace organizational privacy, security, or compliance policy. Sensitive information must remain governed by the connected application's rules and the user's organization.

## Simple Mode principles

### 1. No architecture rewrite

The user connects a supported application. Entroly operates through a documented proxy, wrapper, plugin, or MCP path without requiring changes to the user's codebase or agent design.

### 2. One-time guided setup

The setup assistant must:

1. detect supported local applications;
2. explain what data stays local and what the connected AI provider still receives;
3. configure the safest supported integration path;
4. run a no-key local verification;
5. offer an optional provider-bound pilot;
6. show how to disable or remove the integration.

### 3. Plain-language controls

Default controls:

- **Savings mode:** Off / Balanced / Maximum
- **Keep originals recoverable:** On by default
- **Verify important answers:** On by default for supported workflows
- **Private folders:** user-selectable exclusions
- **Pause Entroly:** one click

Advanced token budgets, routing weights, receipt schemas, and engine settings stay behind an Advanced panel.

### 4. Honest savings

The dashboard must separate:

- measured provider-bound input-token reduction;
- local-only context reduction where provider delivery is unobserved;
- modeled cost avoidance with pricing source and date;
- unknown or unpriced usage;
- quality and verification results.

No fixed compression percentage, universal cost reduction, or provider discount may be displayed without observed evidence.

### 5. Context Assurance receipt

After a supported session, the user can open a plain-language receipt:

```text
Entroly used 12 relevant items from 84 available items.
7 duplicate or low-value items were omitted.
3 omitted originals remain recoverable.
1 evidence risk needs review.
Provider-bound input reduction: measured / not observed.
```

A technical view may expose hashes, paths, token counts, provenance, and verification details.

### 6. Fail-open and reversible

If Entroly cannot safely optimize a request, the original request continues unchanged. The user can disable Entroly without losing the host application's history or configuration.

## Required product surfaces

### Desktop application

- Windows and macOS first; Linux package after the desktop contract stabilizes.
- System-tray status: Active, Paused, Needs attention.
- Local service lifecycle management.
- Automatic updates with signed releases and rollback.

### Supported-app connector catalog

Each connector shows:

- support level: Verified / Experimental / Community;
- integration type: Proxy / Wrapper / Plugin / MCP;
- setup steps;
- data boundary;
- exact features available;
- removal instructions;
- last verified application version.

### Everyday dashboard

The default dashboard answers:

1. Is Entroly active?
2. Which AI apps are connected?
3. Was provider-bound usage observed?
4. How much context was reduced?
5. Were important originals recoverable?
6. Did verification find an evidence problem?

### Support and diagnostics

- one-click health check;
- redacted diagnostic bundle preview;
- clear error messages without stack traces by default;
- guided repair;
- never upload diagnostics without explicit consent.

## Initial connector priorities

1. OpenAI- and Anthropic-compatible local proxy applications
2. Claude Code and Codex wrappers
3. OpenClaw context engine
4. Hermes Agent context engine
5. OpenCode MCP and compaction integration
6. local Ollama and LM Studio workflows
7. common desktop AI clients where a stable, permitted integration contract exists

## Acceptance criteria

A nontechnical pilot user can:

- install Entroly without a terminal;
- connect one supported AI application in under five guided screens;
- complete a local verification without an API key;
- understand whether savings are measured, modeled, or unknown;
- recover one omitted original from a receipt;
- pause and uninstall Entroly without damaging the connected application;
- complete the flow without encountering the terms MCP, JSON, token budget, or process environment unless they open Advanced settings.

## Release gate

Simple Mode may be advertised as "one-click" only after:

- clean installation passes on supported operating systems;
- connector setup and removal are tested against the listed application versions;
- no provider credential is persisted outside the host's approved storage;
- fail-open behavior is verified;
- receipt recovery is verified;
- provider-bound savings labels are evidence-classified;
- usability testing succeeds with nontechnical participants;
- accessibility checks cover keyboard navigation, readable contrast, screen readers, and plain-language errors.
