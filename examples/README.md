# Runnable examples

Run examples from the repository root in an activated environment:

```bash
python -m pip install -e ".[test]"
python examples/sdk_quickstart.py
```

| File | Purpose | Extra requirements |
| --- | --- | --- |
| `sdk_quickstart.py` | Minimal base-package compression and task selection | None |
| `demo_receipt_proof.py` | Auditable receipt log and witness primitives | None |
| `memory_os_e2e_demo.py` | Memory OS end-to-end contract | None |
| `memory_fabric_e2e_demo.py` | Memory Fabric end-to-end contract | None |
| `demo_full_experience.py` | Broad interactive product demo | Optional native engine and terminal support |
| `demo_value.py` | Native-engine value demonstration | `entroly-core` |
| `stream_claude_server.py` | Development-only streaming UI prototype | Local customization; do not use as a production starter |

Examples must use synthetic data, fail with actionable dependency messages,
and avoid machine-specific paths or credentials. Public examples are smoke
paths, not benchmark evidence unless linked to a versioned protocol and raw
result artifact.

See [EXAMPLES.md](../EXAMPLES.md) for copy-paste SDK, receipt, MCP, proxy, and
OpenClaw snippets.
