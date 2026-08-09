# Code, shell, framework, and vision integration boundaries

These surfaces are opt-in or bounded additions to Entroly's evidence-first
compression. They do not change exact-recovery guarantees or authorize remote
downloads.

## Parser-backed code structure

Install `entroly[code-intelligence]` to enable parser-backed exact source spans
for common programming languages. Entroly recognizes more than 27 language
grammars by extension. The parser is optional: missing, malformed, oversized,
or unsupported input uses the existing deterministic extractor.

Entroly does not download a parser as a side effect of reading code. With
language-pack releases that use on-demand grammar downloads, only already
cached grammars are used. An operator may explicitly allow acquisition with
`ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD=1`.

Limitations: parser-backed spans improve structural boundaries; they do not
claim semantic call resolution for every grammar. Python keeps its deeper AST
call graph, while other languages retain conservative dependency inference.

## Command-aware shell evidence

The Entropic Shell Codec combines its generic entropy scorer with small outcome
profiles for test runners, Rust and Node builds, Git, containers, Kubernetes,
Terraform, and general build tools. Profiles preserve the invocation, failure
evidence, and terminal outcome under tight budgets. Unknown commands stay on
the generic path. Every compressed representation retains the complete original
in its content-addressed recovery store.

## Framework request adapters

- `EntrolyCompressor` preserves concrete LangChain message types, tool calls,
  IDs, and provider metadata. It supports sync, async, batch, and stream use.
- `EntrolyDocumentCompressor` implements the retriever-compressor protocol and
  preserves document IDs and metadata.
- `EntrolyLiteLLMCallback` implements the proxy pre-call hook without making
  LiteLLM a required dependency.
- `EntrolyASGIMiddleware` compresses bounded JSON POST/PUT message bodies and
  replays the original bytes on parse or compression failure.

All request adapters preserve tools and generation controls. Applications remain
responsible for choosing budgets appropriate for their model and workflow.

## Embedded image optimization

The proxy preserves images by default. Set `ENTROLY_IMAGE_OPTIMIZATION=1` and
install `entroly[images]` to permit provider-aware resizing of embedded base64
images. OpenAI-, Anthropic-, and Gemini-shaped payloads are recognized. External
URLs are never fetched; malformed images, unknown formats, missing Pillow, and
quality-gate failures keep the original bytes. Audit headers report counts and
estimated before/after image tokens without exposing image data.

Provider token counts are estimates, not billing records. Provider-reported
usage remains authoritative.
