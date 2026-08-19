//! # Entroly Engine — the single source of truth for compute
//!
//! Every algorithm Entroly ships lives here exactly once. The binding crates
//! are thin shells over this one:
//!
//! ```text
//!   entroly-engine  ── algorithms, no binding code ──┐
//!                                                    ├─> entroly-core (PyO3)
//!                                                    │      -> pip, cargo, MCP, SDK, Homebrew
//!                                                    └─> entroly-wasm (wasm-bindgen)
//!                                                           -> npm
//! ```
//!
//! ## Why this crate exists
//!
//! `entroly-core` and `entroly-wasm` previously carried *copies* of these 32
//! modules. Nothing linked them, so nothing failed when one was fixed and the
//! other was not, and they drifted by 4,065 lines — including a similarity
//! estimator that was corrected in the core while the WebAssembly build kept
//! shipping the broken form.
//!
//! With the algorithms here, that failure mode is unrepresentable: a change
//! reaches every distribution channel or it does not compile.
//!
//! ## The `python` feature
//!
//! PyO3 attributes on types that cross the Python boundary are gated behind
//! `feature = "python"`, which is **off by default**. `entroly-core` enables it;
//! `entroly-wasm` must not, since linking PyO3 into WebAssembly is not viable.
//! Prefer `#[cfg_attr(feature = "python", ...)]` over duplicating a type.

pub mod anomaly;
pub mod bm25;
pub mod cache;
pub mod causal;
pub mod channel;
pub mod cognitive_bus;
pub mod conversation_pruner;
pub mod dedup;
pub mod depgraph;
pub mod eicv;
pub mod eicv_suppressor;
pub mod entropy;
pub mod fragment;
pub mod guardrails;
pub mod health;
pub mod hierarchical;
pub mod knapsack;
pub mod knapsack_sds;
pub mod learning;
pub mod lsh;
pub mod nkbe;
pub mod prism;
pub mod query;
pub mod query_persona;
pub mod resonance;
pub mod rnr;
pub mod sast;
pub mod semantic_dedup;
pub mod simhash_wide;
pub mod skeleton;
pub mod trajectory;
pub mod utilization;

pub mod work_graph;

#[cfg(test)]
mod coordination_index;
