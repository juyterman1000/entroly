# PR 264 trust-hotfix execution note

This branch begins at `b15bc231e8fe39221979a46224fcfbafa64dadec`, the merged result of PR #264.

The temporary review workflow applies a deterministic patch only if its source anchors still match, runs focused Python/Rust/WASM trust tests, commits the validated result, and deletes the temporary delivery workflows. The product PR must remain unmerged until the repository-wide CI passes on the resulting unchanged head.

Targeted contracts:

- exact original-byte recovery for lossy codec representations;
- critical numeric log distinctions are never templated away;
- source-derived protected evidence, with fail-closed codec fallback;
- no production `sufficient` verdict from uncalibrated thresholds;
- token-boundary rather than substring stem matching;
- specified, versioned wide-SimHash hashing with golden vectors;
- regex-lite QCCR parity exercised in the WASM wrapper crate;
- release surfaces bumped to 1.0.73.
