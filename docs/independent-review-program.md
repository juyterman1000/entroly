# Entroly independent review program

Entroly needs independent, adversarial, reproducible evaluation—not copied
README claims. This program defines the minimum evidence for a review that the
project may cite as independent.

## Independence rule

A review is independent only when the reviewer controls the workload, scoring,
interpretation, and publication. Entroly maintainers may answer setup questions
or fix a clearly documented bug, but may not tune the workload, remove negative
cases, select only favorable results, or approve conclusions before publication.

Reviews sponsored with money, hardware, services, or other consideration must be
clearly labeled. Sponsored does not mean invalid, but it is not the same as
unsolicited independent coverage.

## Recommended review tracks

### 1. Coding-agent debugging

- Choose a real repository and a bug not created for Entroly.
- Freeze the repository revision and task description.
- Compare raw context with Entroly under the same model, provider, temperature,
  tools, permissions, and effective context cap.
- Measure task outcome, input/output usage, latency, retries, and recovery calls.
- Include cases where Entroly passes through or misses evidence.

### 2. Structured tool output

- Use real JSON, logs, shell output, API results, or database output.
- Preserve identical user questions and model settings across arms.
- Report whether the answer required information removed from active context.
- Test exact recovery after a process restart when recovery is part of the claim.

### 3. Conversation compaction

- Use a multi-turn session with decisions, rejected hypotheses, commands, errors,
  paths, and verification state.
- Score whether the agent preserves and applies those details after compaction.
- Report active-context reduction and total provider-observed usage separately.

### 4. Repository intelligence

- Evaluate symbol discovery, change impact, test localization, and relevant-file
  retrieval on repositories selected before results are viewed.
- Keep the benchmark's own index, oracle, and scoring rules unchanged.
- Distinguish file-level retrieval from line- or symbol-level precision.

### 5. Recovery and receipt integrity

- Attempt wrong-handle, stale-source, corrupted-store, restart, concurrency, and
  path-boundary cases.
- Verify exact bytes, digest, length, provenance, and failure behavior.
- Check that receipts do not claim provider-bound savings when only a local
  estimate was observed.

## Required report fields

A citable review must include:

- reviewer identity or publication;
- disclosure of relationship, sponsorship, and provided support;
- Entroly version and installation source;
- operating system, runtime versions, and hardware where relevant;
- repository or dataset revision;
- model and provider;
- token/context budget and cache settings;
- baseline and treatment commands;
- task-level outcomes;
- input/output token observations and their source;
- latency measurement method;
- recovery, receipt, and verification behavior;
- failures, regressions, exclusions, and pass-through cases;
- raw artifacts or enough detail for reproduction.

## Controls

Use the controls required by the question:

- **No-Entroly baseline:** proves the task and model work without the treatment.
- **Null-context control:** detects tasks solvable from the prompt or oracle alone.
- **Equal-token control:** separates context selection from merely granting one arm
  more tokens.
- **No-recovery arm:** measures whether recoverability changes task success.
- **Pass-through arm:** detects overhead on inputs that already fit the budget.
- **Base-model control:** separates context-treatment effects from model changes.

Not every review needs every control. The report must justify the chosen arms.

## Interpretation rules

- A ceiling where all arms pass does not establish non-inferiority.
- Fewer active tokens do not prove lower total provider cost.
- A larger reduction with worse task quality is not a win.
- Provider-observed usage is distinguished from tokenizer estimates.
- Estimated tokenizer counts must not be labeled provider-billed usage.
- One repository, model, or content type cannot support a universal claim.
- Tuning on the test set invalidates held-out status.
- Retrieval of a known answer passage is not generated-answer quality.

## Reviewer quickstart

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U entroly
entroly doctor
entroly verify-claims
entroly simulate
```

Then freeze the workload and comparison protocol before connecting a paid model.

## How Entroly may cite a review

The project may quote only conclusions supported by the published report and
must preserve the review's date, version, workload, limitations, and disclosure.
Negative results must not be omitted from the summary when they materially
change interpretation.

A review URL and status should be recorded in
`docs/distribution/targets.json` or a dedicated evidence registry. A private
email, unpublished draft, or maintainer-run reproduction is not an independent
review.

## Reporting a reproduction

Open a GitHub issue using the independent-review template and include public
artifact links. Security-sensitive findings should follow `SECURITY.md` instead
of being disclosed through a public benchmark issue.
