# Entroly Examples

All examples are local-first unless they explicitly configure a provider. Start
with the no-key examples before running an agent or proxy.

## Five-minute examples

| Goal | Command | Network or key required? |
| --- | --- | --- |
| Verify the installed package | `entroly verify-claims` | No |
| Estimate reduction on the current repository | `entroly simulate` | No |
| Run the Python SDK quickstart | `python examples/sdk_quickstart.py` | No |
| Demonstrate receipt proof primitives | `python examples/demo_receipt_proof.py` | No |
| Exercise Memory OS | `python examples/memory_os_e2e_demo.py --json` | No |
| Exercise Memory Fabric | `python examples/memory_fabric_e2e_demo.py --json` | No |

## Python SDK

```python
from entroly import compress, optimize

log = "INFO poll complete\n" * 100 + "ERROR database unavailable\n"
smaller = compress(log, budget=100, content_type="log")

fragments = [
    {"source": "auth.py", "content": "def rotate_token(): ..."},
    {"source": "billing.py", "content": "def create_invoice(): ..."},
]
selected = optimize(
    fragments,
    budget=500,
    query="Where is token rotation implemented?",
)
print(selected["context_text"])
```

Use `compress` for query-agnostic structural compaction. Use `optimize` when a
real task query is available.

## Context Receipt

```python
from entroly import create_context_receipt, render_context_receipt

documents = {
    "master.md": "The agreement renews annually.",
    "addendum.md": "The addendum allows termination on change of control.",
}
receipt = create_context_receipt(
    documents,
    query="Can the customer terminate after a change of control?",
    budget=800,
    recoverable=True,
)
print(render_context_receipt(receipt))
```

The recovery bundle can contain source text. Keep it under the same access and
retention policy as the documents.

## Context Commit

```python
from entroly import create_context_commit, replay_context, verify_context_commit

commit = create_context_commit(
    [("policy.md", "Production credentials rotate every 30 days.")],
    query="How often do credentials rotate?",
    token_budget=500,
)
assert verify_context_commit(commit).valid
assert replay_context(commit)
```

## MCP and scoped attachment

```bash
# Local stdio MCP setup
entroly init

# Existing client session with a scoped, expiring grant
entroly attach create --client codex --project . --ttl 4h --install
entroly attach list
```

Restart or reload the client, list its tools, and test against a synthetic file
before enabling Entroly on a private repository.

## Proxy

```bash
python -m pip install "entroly[proxy]"
entroly proxy
```

Point only the test process at the local base URL. Provider-specific examples
are maintained in [cookbook/README.md](cookbook/README.md) and
[docs/compression-proxy.md](docs/compression-proxy.md).

## OpenClaw

```bash
openclaw plugins install clawhub:entroly-openclaw
openclaw plugins list
```

The plugin README documents configuration, compatibility, and removal:
[`integrations/openclaw/README.md`](integrations/openclaw/README.md).

## More examples

- [Cookbook](cookbook/README.md): agent wrappers, proxy setup, SDK, RAVS,
  health, CI budgets, export/import, and daemon recipes.
- [`examples/README.md`](examples/README.md): runnable example inventory and
  prerequisites.
- [Context Receipt examples](docs/examples/context_receipt.md): rendered and raw
  audit artifacts.
- [Public evidence](docs/public-evidence.md): commands that reproduce prominent
  repository claims.
