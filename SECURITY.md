# Security Policy

Entroly handles source code, prompts, model responses, credentials passed to
configured providers, and recoverable context artifacts. Security reports are
treated as product-trust incidents.

## Supported versions

Security fixes are released on the latest stable `1.0.x` line. Users should run
the newest published patch because Entroly does not maintain parallel patch
branches for older releases.

| Release line | Security support |
| --- | --- |
| Latest published `1.0.x` | Supported |
| Earlier `1.0.x` patches | Upgrade required |
| `0.x` and older | Unsupported |

Runtime and platform details are maintained in
[SUPPORTED_VERSIONS.md](SUPPORTED_VERSIONS.md).

## Report a vulnerability privately

Do **not** open a public issue for a suspected vulnerability.

Email **fastrunner10090@gmail.com** with the subject
`[Entroly security] <short description>`. Include, when available:

- the affected version, installation method, and operating system;
- a minimal reproduction or proof of concept;
- expected and observed behavior;
- potential impact and whether credentials or user data may be exposed;
- any suggested mitigation;
- whether you want public credit in the advisory.

Never include live provider keys, private source code, or other people's data.
Use synthetic fixtures and redact logs whenever possible.

The project aims to acknowledge a complete report within 48 hours and provide
an initial assessment within seven days. Resolution timing depends on severity,
reproduction quality, and coordinated-disclosure needs. Reporters will receive
updates when the assessment, mitigation, release, or disclosure status changes.

## Disclosure and release process

For a confirmed vulnerability, maintainers will:

1. reproduce and scope the issue privately;
2. prepare a minimal regression test and fix;
3. review affected Python, Rust, WASM, npm, Docker, MCP, and integration
   surfaces;
4. publish patched artifacts before publicizing exploit details;
5. publish a GitHub Security Advisory with affected and fixed versions;
6. credit the reporter when requested and appropriate.

## Security boundaries

- Entroly is local-first, but configured proxy/provider paths necessarily send
  selected request data to the provider chosen by the operator.
- Context Receipts and Context Commits may contain source material or exact
  recovery data. Protect them with the same access and retention policy as the
  source repository.
- Content addressing detects mutation; it does not prove operator identity
  unless an authenticated signing or attestation path is used.
- Remote model-registry discovery is opt-in. Local ranking, verification, and
  diagnostics must not introduce surprise remote calls.
- Fail-open optimization means an internal optimization error should preserve
  the original request. Security, authorization, path-safety, and explicit
  compliance gates remain fail-closed.

## Hallucination-detection disclosure

WITNESS/EICV output is advisory evidence, not a certification. False positives
and false negatives are possible, performance varies by domain, and independent
validation is required for medical, legal, financial, regulated, or
safety-critical uses. `audit` mode is the safest default when modifying an AI
response would be undesirable. Current benchmark artifacts and limitations live
under [`benchmarks/results/`](benchmarks/results/) and
[`docs/limitations.md`](docs/limitations.md).

