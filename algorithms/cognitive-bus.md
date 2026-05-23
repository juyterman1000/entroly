# Cognitive Bus — ISA Event Routing

**Implementation:** [`entroly-core/src/cognitive_bus.rs`](../entroly-core/src/cognitive_bus.rs)
**Math reference:** [`../RESEARCH.md`](../RESEARCH.md)

## One-paragraph spec

A principled message bus for routing events between AI agents:
observations, tool results, beliefs, memory recalls. Each subscriber
maintains a **Poisson rate model** over its inbound events. Priority of
a new event is **KL-divergence between observed and expected rate**
times **recency** times **novelty** times **mutual information**. High-
salience events get flagged for the hippocampus bridge (cross-session
memory). **Welford online spike detection** bypasses the queue for
critical events.

> `priority(e, s) = KL(λ_obs ‖ λ_exp) · recency(e) · novelty(e) · mi(e, s)`

## When you need it

Multi-agent workflows where agents need to coordinate (notice each
other's findings, share tool results, react to changes). Most projects
do this with ad-hoc queues or a flat event log. The Cognitive Bus
applies information-theoretic priority so the *most surprising,
relevant, and recent* events reach each agent first — the right
asymptotic primitive for coordination.

## Memory-aware features

- High-salience events automatically flag the hippocampus for `remember()`
- Recalled long-term memories can be re-published as `memory_recall` events
- Emotional tags propagate (critical events → 3× priority boost)
- Consolidated memories (neocortex) get lower priority than fresh episodic memory

## Why it's novel

The combination of **per-subscriber Poisson rate estimation** with
**KL-divergence as priority** is, to our knowledge, unique in the AI
agent space. Most pub/sub systems use FIFO or fan-out; entroly's bus
makes priority a first-class information-theoretic quantity.

## References

Welford 1962 (online statistics); Kullback-Leibler 1951 (relative
entropy); Tulving 1972 (episodic / semantic memory distinction).
