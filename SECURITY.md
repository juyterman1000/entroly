# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Entroly, please report it responsibly.

**Email:** fastrunner10090@gmail.com

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We aim to acknowledge reports within 48 hours and provide a fix or mitigation plan within 7 days.

**Please do not open a public GitHub issue for security vulnerabilities.**

## Hallucination Detection Accuracy Disclosure

Entroly ships a hallucination detector (EICV / WITNESS) that runs locally
and can optionally rewrite or suppress LLM responses. Users relying on
this output for compliance-sensitive decisions should understand the
following:

- **False-positive rate is non-zero.** A truthful claim may be wrongly
  flagged as hallucinated and removed (in `strict` mode) or annotated
  with `[unverified]` (in `annotate` mode).
- **False-negative rate is non-zero.** A hallucinated claim may pass
  through undetected.
- **Operating points on public benchmarks** are recorded in
  `benchmarks/results/` (FEVER, SQuAD v2, HaluEval-QA, TruthfulQA,
  RAGAS). Review these before deploying.
- **Performance varies by domain.** Profiles (`rag`, `qa`,
  `summarization`, `dialogue`, `fact_check`) tune the decision band but
  do not eliminate either error class.
- **`audit` mode is the safe default** for telemetry, dashboards, and
  any application where modifying the AI's output is undesirable. It
  emits observability headers without changing the response body.
- **Not certified for medical, legal, financial, or safety-critical
  applications.** EICV is a research-grade tool. Independent validation
  is required before use in regulated sectors.
- **Suppression should be disclosed to end users** when the detector is
  used in a service that returns modified AI output to third parties.

