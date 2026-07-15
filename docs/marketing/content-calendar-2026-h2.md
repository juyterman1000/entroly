# Entroly evidence-led content calendar

Period: August 2026 through January 2027. Review monthly against the metrics in
the [open-source growth audit](oss-growth-audit-2026-07.md).

The goal is to teach context engineering and make Entroly's value reproducible.
One technical source asset should be adapted for each channel; do not post the
same promotional copy everywhere. Every quantitative post links to raw evidence
and a release tag.

## Monthly plan

| Month | Primary technical asset | Tutorial / case study | Short-form adaptations | Demo video | Newsletter / community |
|---|---|---|---|---|---|
| Aug 2026 | “What did the model actually see? A practical context-receipt specification” | Five-minute Claude Code, Codex and OpenClaw setup matrix | X thread on receipt anatomy; LinkedIn architecture note; Dev.to tutorial | From install to first verified receipt | Roadmap vote and contributor office hour |
| Sep 2026 | Reproducible context-compression scorecard with raw fixtures | How to benchmark your own repository without leaking code | Reddit methodology discussion; benchmark chart with caveats; release post | Run the benchmark locally and inspect failures | Benchmark reproduction challenge, no prize-for-stars |
| Oct 2026 | Recoverable compression versus irreversible summarization | Incident case study: finding evidence omitted from an agent request | X failure-analysis thread; LinkedIn data-integrity post; Dev.to case study | Omitted-evidence explorer walkthrough | Maintainer retrospective and good-first-issue batch |
| Nov 2026 | Provider-independent context control for OpenClaw | OpenClaw custom-provider tutorial using one Entroly engine | ClawHub feature spotlight; community clip; integration release post | OpenClaw provider auto-discovery demo | Integration partner Q&A |
| Dec 2026 | Cache-aware context: when fewer tokens cost more | Measure cache-adjusted cost across Anthropic/OpenAI-compatible routes | X cost-math thread; LinkedIn engineering note; practical Reddit post | Cache hit and receipt comparison | Year-in-review metrics with failures included |
| Jan 2027 | “Trustworthy agent context” technical report v1 | LangChain, LlamaIndex and LiteLLM interoperability recipes | HN launch; Dev.to overview; conference abstract; LinkedIn report | End-to-end multi-agent evidence handoff | 2027 roadmap and contributor recognition |

## Channel standards

### Technical blog and Dev.to

- Lead with the problem and runnable repository, not the product name.
- Include environment, versions, commands, raw artifacts and limitations.
- Provide a “when Entroly is not the right tool” section.
- Use canonical links so search engines attribute the original guide.

### X and LinkedIn

- Use one result, one diagram, and one reproducible command per post.
- Explain what failed or did not generalize.
- Invite corrections and workload submissions, not stars as payment.
- Turn useful replies into documented FAQ entries with permission.

### Reddit

- Post only in communities where the technical lesson is directly relevant.
- Disclose maintainer affiliation in the opening paragraph.
- Avoid synchronized voting, cross-post spam and competitive dunking.
- Remain available to answer technical questions after posting.

### YouTube

Planned tutorials:

1. First verified context receipt in under five minutes.
2. Compare raw versus Entroly context on your own repository.
3. Diagnose an omitted-evidence failure and recover the source.
4. Install the OpenClaw plugin with any provider route.
5. Read benchmark JSON and reproduce the published scorecard.
6. Build a custom MCP client around Entroly's hardened tool allowlist.

Record terminal commands at readable speed. Add chapters, captions, transcript,
exact versions and pinned commands. Never replace evidence with an animation.

## Hacker News launch checklist

- Ship a stable release and public postmortem for any launch blocker first.
- Use a descriptive title, not superlatives.
- Prepare a one-command no-key proof and a clean-machine recording.
- Link source, license, architecture, limitations and benchmark artifacts.
- Have maintainers available for technical questions for the first day.
- Answer criticism with data; log valid issues publicly.
- Do not ask communities or employees to coordinate votes.
- Publish a 72-hour follow-up with traffic, failures and fixes.

## Conference and podcast ideas

- “Context receipts: provenance for agent prompts.”
- “Recoverable compression under a hard token budget.”
- “The cache-adjusted economics of agent context.”
- “Testing what an AI coding agent did not see.”
- “Local-first verification without another model call.”

Each proposal should include a working open-source demo and general lessons that
remain useful without adopting Entroly.

## Monthly operating rhythm

Week 1: publish the technical source and raw artifacts. Week 2: publish the
tutorial and office hour. Week 3: adapt the result for two relevant channels.
Week 4: report metrics, failures, corrections and next experiments. Stop or
change a channel if it creates low-quality traffic or maintainer overload.
