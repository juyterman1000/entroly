# Support

Choose the channel that matches the problem so maintainers and other users can
respond efficiently.

| Need | Channel |
| --- | --- |
| Setup question, design discussion, or usage advice | [GitHub Discussions](https://github.com/juyterman1000/entroly/discussions) |
| Reproducible bug or compatibility regression | [Bug report](https://github.com/juyterman1000/entroly/issues/new?template=bug_report.yml) |
| Feature proposal with a concrete workflow | [Feature request](https://github.com/juyterman1000/entroly/issues/new?template=feature_request.yml) |
| Benchmark result or public-claim discrepancy | [Evidence report](https://github.com/juyterman1000/entroly/issues/new?template=evidence_report.yml) |
| Vulnerability or credential/data exposure | Follow [SECURITY.md](SECURITY.md); do not file publicly |
| Community conduct concern | Email `fastrunner10090@gmail.com` with subject `[Entroly conduct]` |

## Before asking for help

```bash
entroly --version
entroly doctor
entroly verify-claims
```

Include the installation method, Python or Node version, operating system,
integration path, exact command, expected behavior, actual behavior, and the
smallest redacted log that reproduces the issue. Never post API keys, bearer
tokens, private source code, raw Context Commits, or recovery bundles.

Maintainers prioritize data loss, security, silent corruption, broken installs,
and regressions in supported paths. Response times are goals, not paid support
SLAs. Community members are encouraged to answer questions and confirm
reproductions.
