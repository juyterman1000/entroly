# Entroly Memory Fabric

Entroly Memory Fabric is the unifying product layer for Entroly's memory stack.

It exists because Entroly now has several memory technologies:

- `MemoryOS` for local public runtime memory,
- optional `hippocampus-sharp-memory` bridge for cross-session retention,
- Rust `MemoryManager` for native high-scale cognitive memory,
- SCHIPC for inter-agent memory traffic control,
- ComplianceGate for unsafe memory traffic,
- Pollination for learned agent lesson sharing,
- Federation for privacy-preserving shared learning,
- Context Receipts and WITNESS for audit and verification.

The Fabric turns those into one product story:

```text
remember safely -> recall under budget -> emit receipt -> persist -> verify -> learn/share when useful
```

## Public API

Use the first-party stable path:

```python
from entroly.memory_fabric import MemoryFabric

fabric = MemoryFabric()
fabric.remember(
    "Auth timeout was fixed in auth/session.py",
    agent_id="coder",
    importance=0.9,
    source="incident/auth-timeout",
    tags=["auth", "timeout"],
)

result = fabric.recall(
    "why is login timing out again?",
    agent_id="coder",
    budget=1200,
)

print(result.as_text())
print(result.receipt())
fabric.save(".entroly/memory.json")
```

Load it later:

```python
from entroly.memory_fabric import MemoryFabric

fabric = MemoryFabric.load(".entroly/memory.json")
```

## Layer model

| Layer | Status today | Role |
|---|---|---|
| MemoryOS | Active | Public runtime memory: local safety, budget-aware recall, decay, persistence, receipts |
| Hippocampus bridge | Optional | Cross-session long-term retention when `hippocampus-sharp-memory` is installed |
| Rust MemoryManager | Native adapter ready; wheel export pending | Native high-scale memory with hippocampal buffer, Kanerva SDM, LSH, sleep replay |
| SCHIPC | Native adapter ready; wheel export pending | Suppresses redundant inter-agent messages before token explosion |
| ComplianceGate | Native adapter ready; wheel export pending | Kernel memory-safety layer for PII, injection, and rate limiting |
| Pollination | Native adapter ready; wheel export pending | TD(0)-learned agent lesson sharing and experience packs |
| Federation | Optional / explicit | Differentially private archetype-weight sharing |
| Context Receipts + WITNESS | Active | Audit what was selected/omitted and verify generated answers against evidence |

## Why this is different from a memory store

Most memory products answer:

```text
What facts should I retrieve?
```

Memory Fabric answers:

```text
What memory is safe, fresh, useful, under budget, auditable, and worth sharing?
```

That lets Entroly position itself as a memory-control runtime, not a vector store wrapper.

## Capability introspection

The Fabric exposes runtime capabilities explicitly:

```python
fabric = MemoryFabric()
for layer in fabric.capabilities():
    print(layer.name, layer.status, layer.role)
```

This matters for production because optional memory engines should not silently change behavior. Applications can log exactly which memory stack is active.

## Native kernel adapters

`MemoryFabric` now detects and uses native kernels when the installed `entroly_core` wheel exports them:

- `MemoryManager`,
- `IpcBus`,
- `ComplianceGate`,
- `PollinationEngine`.

The adapter path is fail-closed:

- if `ComplianceGate` blocks a message, Fabric returns `delivered=False`,
- if `IpcBus` is unavailable, Fabric returns `native_ipc_unavailable`,
- if `PollinationEngine` is unavailable, lesson recording/sharing returns a structured disabled result,
- `capabilities()` reports `available`, `internal`, or `disabled` for each layer.

Example:

```python
from entroly.memory_fabric import MemoryFabric

fabric = MemoryFabric()

sent = fabric.send_agent_message(1, 2, "new auth retry lesson")
blocked = fabric.send_agent_message(1, 2, "sk-secret-token-should-not-pass")

fabric.record_agent_lesson("coder", "auth retry strategy worked", success=True, surprise=0.2)
fabric.share_agent_lessons("coder", "reviewer")
fabric.reward_agent_share("coder", "reviewer", 1.0)

print(sent)
print(blocked)
print(fabric.stats())
```

To activate the native kernels in wheels, the Rust PyO3 module must export the existing classes from `entroly-core/src/lib.rs`:

```rust
m.add_class::<memory::MemoryManager>()?;
m.add_class::<ipc::IpcBus>()?;
m.add_class::<compliance::ComplianceGate>()?;
m.add_class::<pollination::PollinationEngine>()?;
```

The Fabric Python side is ready for those exports now.

## End-to-end demo

Run the deterministic offline demo:

```bash
python examples/memory_fabric_e2e_demo.py
python examples/memory_fabric_e2e_demo.py --json
```

It demonstrates:

1. storing high-value evidence through the Fabric,
2. blocking unsafe memory before storage,
3. recalling under budget,
4. emitting one combined receipt,
5. reporting the active/internal/optional memory layers,
6. saving and loading durable local memory.

## Stress benchmark

Run the Fabric benchmark:

```bash
python benchmarks/memory_fabric_stress_test.py
python benchmarks/memory_fabric_stress_test.py --json
```

It gates:

1. public orchestrator recall,
2. capability contract completeness,
3. safety contract,
4. receipt contract,
5. persistence contract.

The benchmark is offline and intentionally does not depend on an LLM, vector DB, or optional memory engine.

## CI gate

The MemoryOS Production Gate now includes Fabric checks:

```bash
pytest tests/test_memory_fabric.py tests/test_memory_fabric_benchmark.py tests/test_memory_fabric_e2e_demo.py
python benchmarks/memory_fabric_stress_test.py --json
python examples/memory_fabric_e2e_demo.py --json
```

## Maturity contract

Use this claim publicly now:

```text
Entroly Memory Fabric unifies MemoryOS with optional long-term and native memory layers through one safe public orchestration API. Native SCHIPC, ComplianceGate, and Pollination adapters are ready and activate automatically when exported by entroly_core.
```

Avoid this claim until public PyO3 exports and external benchmarks are fully green:

```text
Every Entroly memory layer is fully productized and benchmarked against every peer.
```

## Next integration milestones

1. Add the four PyO3 export lines in `entroly-core/src/lib.rs`.
2. Add public examples for SCHIPC, ComplianceGate, Pollination, and Federation using real native classes.
3. Run an external benchmark comparing Memory Fabric against simple vector memory, Mem0/Zep-style memory, and graph-memory systems.
4. Add a real model-output benchmark: recall -> context receipt -> answer -> WITNESS verification.
