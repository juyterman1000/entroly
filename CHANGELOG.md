# Entroly changelog

This file is the stable entry point for user-facing release history. Published
version-specific notes live in [GitHub Releases](https://github.com/juyterman1000/entroly/releases).

## Unreleased

- Dollars: the headline is now the conservative, cache-aware net saving, and
  OpenAI and Gemini traffic is priced.
  - **The headline dollar figure changed.** Traffic Value, `/stats`
    (`headline`) and `entroly value` now lead with **Net saving
    (conservative)**: what Entroly removed, priced at the cheapest rate the
    provider could have billed it, minus what it added and any cached prefix
    it broke at the dearest, summed (signed) over requests that have provider
    usage and a verified price, and shown with that coverage ("priced N of M
    requests"). With no priced request there is no dollar figure; the
    list-price model is never substituted. The former lead figures stay in
    the API under their existing keys and are relabelled as what they are:
    "List-price model (ignores caching)" (`estimated_value_avoided_usd`),
    "Provider cache discount (measured)" (`measured_cache_benefit_usd`) and
    "Modeled total (list price + cache discount)"
    (`total_ai_value_protected_usd`). New fields: `headline`,
    `net_saving_conservative_usd`, `net_saving_conservative_micro_usd`,
    `net_saving_priced_requests`, `net_saving_coverage_pct`, `headline_metric`,
    and Prometheus `entroly_net_saving_conservative_usd` /
    `entroly_net_saving_priced_requests_total`.
  - Bundled, dated, source-linked price tables for OpenAI
    (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-6-luna`, `gpt-4.1`,
    `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`) and Gemini (`gemini-3.5-flash`,
    `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`,
    `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`,
    `gemini-2.5-flash-lite`) join the Anthropic one. Until now an OpenAI or
    Gemini receipt carried provider usage and no dollar figure. Models and
    request sizes whose price could not be read from the provider's own page
    stay unpriced; `ENTROLY_PRICING_CATALOG` still overrides everything.
  - Measured on a replayed 25-turn agent session per provider against a
    provider stub that bills cache reads and writes (64 requests): stub-billed
    saving $0.760, headline $0.554, list-price model $1.62. Per provider:
    Anthropic $0.253 measured / $0.159 reported, OpenAI $0.507 / $0.396.
    One workload and one stub, not a bound.
  - The governor now also sees OpenAI and Gemini conversations, since it acts
    only on priced requests.

- Proxy: agent sessions cost less time and keep the code the agent read.
  Measured by replaying one 25-turn tool session and a set of plain chat
  requests through `entroly proxy` against a provider stub that counts tokens
  and bills prefix-cache reads and writes.
  - **Repository context is now opt-in** (`ENTROLY_CONTEXT_INJECTION=off|chat|always`,
    default `off`). Every new question gained ~740 tokens and 5-6 s; the
    tokens were an alphabetical file listing, identical for a haiku and for a
    question about `handle_proxy`, because the hierarchy was seeded with
    `qccr::<source>` ids the fragment store does not hold. A query-conditioned
    selection is now rendered flat, as the ranker produced it, and a listing
    with no cluster and no code is never injected. **Behaviour change:** set
    `ENTROLY_CONTEXT_INJECTION=chat` or `always` to keep context injection.
  - Selection runs once per new user question. On OpenAI-format requests it
    ran on every tool turn, for the question asked many turns earlier: ~5.8 s
    per turn, now ~0.1 s.
  - File reads and source code are forwarded as sent. 36% of the lines of the
    files an agent read reached the model, and the request carried no tool or
    instruction for recovering the rest. `ENTROLY_PROTECT_FILE_READS=0`
    restores the old behaviour. `ENTROLY_TOOL_RESULT_EXCLUDE` is now honoured
    on the default path; it was read and then applied only with session
    rescue off.
  - A tool result byte-identical to an earlier one in the same request is
    replaced by a one-line pointer to it (`ENTROLY_TOOL_RESULT_DEDUP`).
  - A tool result is rewritten only when that removes at least 20% of it.
  - Net-saving ledger: an added block the provider read from cache is charged
    as a cache read. It was charged against whatever the request wrote, so a
    turn that appended the client's own new content billed old rewrites as
    fresh 1-hour writes; four such turns held the conversation, and the hold
    forwarded raw history and broke the cache. A block's digest no longer
    includes `cache_control`, which clients move every turn: a removed block
    was priced as never-sent on two consecutive turns, which put the ledger
    above the measured saving ($0.418 against $0.395 on the same session).
  - `/stats` and `entroly value` count requests whose only change was
    compression. They were booked only when context was injected, so a session
    cut by half reported `saved_total: 0` and `input_tokens_reduced: 0`.

- Claude Code integration fixes.
  - The RAVS capture hook recorded no outcome from Claude Code. Claude Code puts
    no exit code in a Bash hook payload: it fires `PostToolUse` only after exit
    0 and `PostToolUseFailure` after a non-zero exit, whose `error` opens with
    `Exit code N`. Capture now reads both, inferring exit 0 from `PostToolUse`
    only under `--host claude-code`, and the session journal records a failing
    command as `exit:N` rather than "ok".
  - `entroly ravs hook install --claude-code` now registers `PostToolUseFailure`
    too, with `--host claude-code` on the command. Re-run it to upgrade an
    existing registration in place; the cookbook snippet and this repository's
    `.claude/settings.json` are updated the same way.
  - `entroly uninstall` removes those hooks before running pip. They name this
    installation's `entroly` by absolute path, so once pip deleted it every Bash
    call in Claude Code showed a hook error.
  - When every plugin runner failed, the plugin said "no runner found. Install
    uv ... or Node" even when `entroly` was installed and broken, and dropped the
    real error. It now names the failing runner, its exit status and the last
    line of its error, with a separate message when nothing is installed.
    A broken install that exits before reading the hook payload is reported
    by its exit status, not by the incidental `EPIPE`.
  - The capture hook's `entroly` path is written as one word for bash, which
    Claude Code runs hooks with (Git Bash on Windows). A path containing a
    space split into words, and on Windows bash removed every backslash, so
    the hook failed on every Bash tool call. Re-run
    `entroly ravs hook install --claude-code` to rewrite an existing hook.
- `entroly hook install` no longer edits shell startup files. The block it
  appended compressed nothing (its functions set a variable nothing read),
  replaced any existing bash `DEBUG` trap, and could not be removed:
  `entroly hook uninstall` failed with "takes 0 positional arguments". Install
  now changes nothing and points at `entroly shrink -- <command>`, which does
  compress command output; uninstall removes an existing block byte for byte;
  status lists each shell config file.
- First-run fixes from a clean-room run of `pip install entroly`, then
  `verify-claims`, `simulate` and `entroly go`, on pallets/click.
  - `.entroly/` now ignores itself with its own `.gitignore` (as `.pytest_cache`
    does). `entroly go` fills it with one generated summary per source file, and
    the first `git add -A` afterwards committed all 172 of them.
  - `verify-claims` says when relevance ranking is off. Without `entroly-core`
    every check read green while selection ignored the query; the result now
    states that the savings figure is budget arithmetic and how to fix it.
  - `verify-claims`, `simulate`, `perf` and `demo` share one file cap (120), so
    two local checks on the same repository no longer report 114 and 94 files.
  - `simulate` prints its limitations one per line.
  - `entroly go` gives each client family its own base URL:
    `ANTHROPIC_BASE_URL=http://localhost:9377` for the Anthropic SDK and Claude
    Code, and `.../v1` only for OpenAI-compatible SDKs. The single `/v1` URL it
    printed made Anthropic clients call `/v1/v1/messages`, which returned 404.
  - A default start no longer warns that auto-recovery will not cover
    streaming. Both settings involved are defaults; `/stats` still reports it,
    and the warning remains for an explicit `ENTROLY_AUTO_RECOVERY`.
  - `entroly daemon` honours `ENTROLY_PROXY_PORT` and `ENTROLY_DASHBOARD_PORT`,
    as `entroly proxy` documents. It hard-coded 9377 and 9378, so with the
    proxy moved to another port it still tried to bind the old one.
  - `entroly learn --json` prints JSON. Without `--history` it printed its
    human report, which no JSON consumer can parse.
- Surfaces that printed figures nobody measured now show measured values or
  say they have none.
  - VS Code: the compress commands computed a fixed 72% saving, copied the first
    1,200 characters under a made-up "Merkle Receipt ... (Verified 0-1
    Knapsack)" line, and "View Cryptographic Receipt" showed every user the same
    hardcoded receipt, whose "Merkle Root" was the SHA-256 of the empty string.
    The commands now run `entroly compress` on what the editor shows and display
    that receipt; without the CLI they say so and show no figures.
  - Entroly Desktop (`ui/`): counters started at 1,428,500 tokens and $14.28 and
    never changed, the header said "DAEMON :9377 ACTIVE" while offline, and the
    compression preview showed ~78% saved for any text. Counters now show "—"
    until `/api/stats` reports a value, offline says offline, and the preview is
    labelled an illustration and reports what its rules removed.
  - The "Add Entroly to Cursor" links installed `entroly-mcp@1.0.84`: the pin
    sits base64-encoded inside the link, where no version check could see it.
    They now install the current release; the staleness checker decodes these
    links and the bump script re-pins them.
  - Browser extension: "Compress Selection with Entroly" stored the selection
    where the playground could never read it, and the GitHub button's `?repo=`
    was ignored. The selection now travels in the URL fragment (never sent to a
    server) and the playground loads both.
- A fail-closed `assure_context` now says what it could not verify. It raised
  only "context assurance failed closed: uncertain"; the error now lists each
  check that did not pass and, for the token budget, whether tiktoken or only
  its cached `o200k_base` encoding is missing, with the command that caches it.
- Documentation that failed when copied: the README's first SDK example raised
  `NameError`; the Copilot CLI setup used a `--client copilot` the CLI rejects
  (it now shows Copilot's own `/mcp add`); `entroly attach create` was missing
  its required `--client`; and `entroly audit --taint` lacked the session
  chain it needs.
- Responses through `entroly proxy` keep the provider's own metadata.
  - Request ids (`request-id`, `x-request-id`) and rate-limit and retry headers
    (`anthropic-ratelimit-*`, `x-ratelimit-*`, `retry-after`) reach the client;
    every one was dropped, so the SDKs' `_request_id` was `None`.
  - A 429 or a final 5xx on the buffered path is the provider's body and
    headers, not the proxy's own JSON, so clients can tell `overloaded_error`
    from `rate_limit_error` again. The retry policy is unchanged.
  - Other provider endpoints (`count_tokens`, the model list, files, audio)
    pass through with their headers, and non-JSON bodies arrive byte for byte
    instead of wrapped as `{"data": ...}` text.
- Requests the proxy made larger are reported as larger. Appending retrieved
  context turned a 15-token "Say hello" into 674 tokens, yet `entroly value`
  counted it as optimized and the `X-Entroly-Outcome` header said
  `optimized`. Value now reports `requests_enlarged`, `input_tokens_added` and
  `net_input_tokens_reduced` beside the saving, and the header says
  `context_added`, `modified` or `observed` unless the request was sent smaller.
- Response distillation no longer leaves broken text: "I hope this helps!"
  ended an answer in a bare "I", and removing a hedge at the start of a
  sentence left it in lower case.
- The npm MCP server (`entroly serve` after `npm install -g entroly`, and
  `npx entroly-wasm --serve`) now answers newline-delimited requests with
  newline-delimited JSON, as the MCP stdio transport specifies. It always
  replied with `Content-Length` frames, which the official MCP clients cannot
  parse ("Unexpected token 'C'"), so an editor configured by `entroly init`
  never completed the handshake. Clients that send `Content-Length` frames
  still get them back. Its `scan_for_vulnerabilities` tool now scans the code
  it is given; it looked up a stored fragment named by `source` instead and
  answered "Fragment not found" to every call.
- The Python MCP tools `shared_memory_list` and `learn_from_failures` no longer
  fail on every call. Each passed its backend an argument the backend did not
  accept (`tag`, `sources`); the store now filters by tag and the miner reads
  the sources asked for.
- `entroly serve` now stops on Ctrl-C and SIGTERM while its stdin is still
  open. In a terminal, Ctrl-C was ignored until Enter was pressed and then
  printed three tracebacks; when a host kept stdin open, SIGTERM was logged but
  the server kept running, holding its dashboard port and file watchers. State
  is still persisted before it exits.
- With a HOME it cannot write, Entroly now keeps its data in a directory only
  the current account can open (`<tmp>/entroly-<uid>`, mode 0700) instead of
  the shared `<tmp>/entroly`. `verify-claims` and `simulate` failed with
  "Permission denied: '/tmp/entroly/checkpoints'" once another account had
  used the shared name, and an account that created it world-writable decided
  where everyone else's checkpoints and telemetry went. A per-account path
  that is not a private directory owned by the current account is refused with
  a message naming `ENTROLY_DIR`. Data previously kept under `/tmp/entroly` is
  not carried over.
- Two Entroly processes in one project (proxy, MCP server, CLI) no longer
  corrupt its persisted index. Every writer staged in the same
  `index.json.gz.tmp`, so concurrent saves produced index files that would not
  load ("Failed to load persistent index") and forced a full re-index.

- With session rescue off, shell tool output of at most 4,000 characters is
  forwarded verbatim (`ENTROLY_SHELL_VERBATIM_CHARS`, `ENTROLY_SHELL_TOOLS`);
  a 30-file `ls -l` had been rewritten with sizes rounded to "1KB" and modes
  and timestamps dropped. Tool results are now matched to their tool's name
  through the call id, so `ENTROLY_TOOL_RESULT_EXCLUDE` also protects
  Anthropic tool results, which it never matched before.
- Security: closed four paths by which untrusted content or another website
  could act through Entroly's local surfaces.
  - The proxy's sidecar endpoints (`/retrieve`, `/context` and others) now
    require a loopback `Host` header. A DNS-rebinding page could otherwise read
    recovered context, because its same-origin GET carried no `Origin` or
    `Referer` for the existing checks to reject.
  - The VS Code compression view escaped nothing it rendered: a file containing
    markup ran script in the webview, which could put any text on the
    clipboard. It now escapes the file name and text, runs under a nonce-only
    CSP, and copies only the text it rendered.
  - The public playground wrote GitHub file names into the page as HTML; a
    repository file named `<img src=x onerror=...>.py` ran script there.
  - The streaming example accepted WebSockets from any site, listened on every
    interface, and rendered model output as HTML. It now checks Origin and Host,
    binds to 127.0.0.1, and escapes output.
- Claude Opus 4.8, Opus 4.7 and Sonnet 4.6 now resolve to their verified
  1M-token window, 128K output and published prices (Bedrock ids included);
  they had fallen back to 128K, so session rescue compacted history early.
- The templated log form no longer factors out values that contain its `|`
  column separator. A value like `tags=a|b` split into two column entries,
  shifting every later entry so each line's values sat beside another line's
  (which user had which tags was wrong, though every value was present). Those
  lines now stay verbatim. New tests check that every compressed form keeps
  paired facts -- test and outcome, file:line and error code, row id and
  value -- together.
- Traffic Receipts now carry a conservative net saving
  (`net_saving_conservative_micro_usd`) for priced Claude models, on by
  default. It prices only what Entroly changed: removals at the cheapest
  possible rate, additions and any cache prefix Entroly broke at the most
  expensive, scaled by the provider's own token count. Unchanged requests are
  zero, the provider's cache discount is never credited, values can be
  negative, and it is never added to the headline. Uses a bundled, dated
  Anthropic price table unless `ENTROLY_PRICING_CATALOG` is set
  (`ENTROLY_BUNDLED_PRICING=0` disables it). Receipts now share the
  `X-Entroly-Request-Id` the client sees when the client sends no request id.
- Net-saving governor, on by default: when a conversation's conservative net
  saving falls below -$0.02 over at least 3 priced requests, with losses on
  at least 2 of them, the proxy stops
  applying optional optimizations to that conversation and forwards the
  client's own request. Visible as `X-Entroly-Net-Saving-Governor: hold` and
  in the receipt; `ENTROLY_NET_SAVING_GOVERNOR=0` disables it
  (`ENTROLY_NET_SAVING_FLOOR_USD`, `ENTROLY_NET_SAVING_MIN_REQUESTS` tune it).
- The conservative net saving now prices cache writes at the 5-minute/1-hour
  mix the provider reports (`cache_creation`), instead of assuming every write
  is a 1-hour write, and charges added content from uncached input at the input
  rate. Without a reported split it keeps the 1-hour assumption.
- A tool result cut to a partial codec view (for example a code skeleton)
  now opens with a one-line notice naming the view and how to get the exact
  content; bodies were previously dropped with no marker on the
  `ENTROLY_SESSION_RESCUE=0` and OpenClaw paths.
- Provider TLS now trusts the operating-system certificate store plus the
  bundled roots, in the Python proxy and `entroly-rs`, so Entroly works behind
  TLS-inspecting corporate gateways. `NODE_EXTRA_CA_CERTS` adds roots instead
  of replacing the store; `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` still name the
  whole trust set. `ENTROLY_TLS_TRUST=bundled` restores bundled roots. Adds the
  `truststore` dependency. See
  [Troubleshooting](docs/TROUBLESHOOTING.md#certificate_verify_failed-or-unknownissuer-behind-a-corporate-network).
- Claude Fable 5.1, Opus 5.5, Opus 5, Sonnet 5.5, Sonnet 5 and Haiku 5.5 now
  resolve to their verified 1M-token window and published prices, and Opus 4.6
  to its 1M window; they had fallen back to 128K, so session rescue compacted
  history at ~113K tokens. Savings are priced at Opus 4.5-4.8's $5/$25 and
  Haiku 4.5's $1/$5, with Bedrock and router model names recognised.
  Injected repository context is budgeted against at most a 200K window
  (`ENTROLY_INJECTION_REFERENCE_WINDOW`, 0 disables), which lowers GPT-5.6
  and Gemini 3.6 injection ceilings from 256K to 60K.
- Conversation compression (`ENTROLY_SESSION_RESCUE=0`) no longer rewrites
  Anthropic `thinking`/`redacted_thinking` blocks, which the API rejects when
  modified, and measures utilisation against the request model's window
  instead of a fixed 128K.
- Corrected optional WITNESS calibration's lower-tail rank and empty-evidence
  handling. Legacy or invalid stored profiles now return all-SUPPRESS thresholds
  rather than reactivating permissive learned values; recalibrate explicitly.
  Missing profiles retain uncalibrated defaults. See the
  [assumptions, scope and compatibility notes](docs/witness-calibration.md).
- Newly prepared contexts sign a full Context Commit anchor so exact recovery
  and proof-guided planning can reach committed omissions beyond the 20-item
  receipt preview. Older preparations remain listed-only; re-prepare to opt in.
  Audit loading now verifies the exact payload it consumes. See
  [recovery compatibility and rollback](docs/verified-efficiency-layer.md#recovery-compatibility-and-rollback).
- Proof-guided runtime v2 rejects replies that do not carry the issued request's
  `revision` and `request_sha256`, and serializes cooperating local writers.
  CLI, MCP and HTTP clients must retain that binding beside each model call.
  Existing v1 sessions are inspect-only: re-prepare with a new idempotency key
  and upgrade all writers together. See the
  [protocol and rollback limits](docs/proof-guided-context-fixed-point.md#durable-host-driven-protocol).
- Runtime native-engine repair now requires explicit consent. Install
  `entroly-core` yourself, call `entroly.repair()`, or opt in to startup repair
  with `ENTROLY_ENABLE_SELF_HEAL=1`. `ENTROLY_NO_SELF_HEAL=1` and
  `ENTROLY_AIR_GAP=1` override consent. Missing-native diagnostics remain visible.
- Corrected the PEP 668 marker path so repair respects externally managed Python.
- Context Receipt and Context Commit file writes use atomic replacement and
  private POSIX file modes. Windows inherits directory ACLs; existing directory
  permissions and unencrypted storage remain operator responsibilities.
- Corrected privacy/telemetry disclosures and removed unsupported accuracy and
  unnamed-product comparison claims. Privacy diagnostics no longer certify the
  absence of network traffic. Historical loss cases remain documented.

These changes are not part of the published releases below until a patch release
is completed. See [recovery-data security](docs/recovery-data-security.md).

## Current release

- [`v1.0.84`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.84)

## Recent releases

- [`v1.0.83`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.83)
- [`v1.0.82`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.82)

## Changelog contract

A release note must describe user-visible behavior rather than only internal
implementation work. Include, where applicable:

- capabilities added, changed, or removed;
- installation and upgrade impact;
- compatibility and migration requirements;
- security, privacy, persistence, or network-boundary changes;
- fixes for user-visible failures;
- verification commands and linked evidence;
- known limitations and regressions;
- rollback or uninstall guidance.

Quantitative statements must identify the exact version, workload, model or
provider when applicable, token or context budget, sample, protocol, and raw
artifact. A result from an earlier version is not automatically evidence for a
new release.

## Publication rule

Do not add a release as current until its tag and coordinated artifacts are
public and resolve to the intended commit. The canonical release procedure is
[`docs/RELEASE.md`](docs/RELEASE.md), and the coordinated publication workflow is
[`.github/workflows/entroly-publish.yml`](.github/workflows/entroly-publish.yml).
