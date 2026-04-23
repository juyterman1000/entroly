---
claim_id: e3baa0a6-33d8-4b21-b6b2-ac1af264e616
entity: cognitive_bus
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\cognitive_bus.rs:132
  - entroly-wasm\src\cognitive_bus.rs:436
  - entroly-wasm\src\cognitive_bus.rs:59
last_checked: 2026-04-23T03:07:07.891497+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: cognitive_bus

**Language:** rust
**Lines of code:** 884

## Types
- `pub struct BusEvent` — An event published on the cognitive bus.
- `pub struct CognitiveBus` — - Entroly dedup (SimHash novelty scoring) - Hippocampus memory (salience-based remember/recall bridge) - NKBE allocator (events influence budget reallocation)  Memory-aware routing: - Events with sali
- `pub enum EventType` — Event types routable on the cognitive bus. Maps to agentOS 25 event types, grouped into 4 zones.

## Dependencies
- `crate::dedup::`
- `std::cmp::Ordering`
- `std::collections::`
