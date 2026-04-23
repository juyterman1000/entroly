---
claim_id: 37ad6133-7b4f-4827-99fa-1fd8524ff8f2
entity: conversation_pruner
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\conversation_pruner.rs:188
  - entroly-wasm\src\conversation_pruner.rs:212
  - entroly-wasm\src\conversation_pruner.rs:96
  - entroly-wasm\src\conversation_pruner.rs:134
  - entroly-wasm\src\conversation_pruner.rs:232
  - entroly-wasm\src\conversation_pruner.rs:264
  - entroly-wasm\src\conversation_pruner.rs:323
  - entroly-wasm\src\conversation_pruner.rs:344
  - entroly-wasm\src\conversation_pruner.rs:399
  - entroly-wasm\src\conversation_pruner.rs:466
last_checked: 2026-04-23T03:07:07.892340+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: conversation_pruner

**Language:** rust
**Lines of code:** 1467

## Types
- `pub struct ConvBlock` — A single conversation block with causal metadata.
- `pub struct PruneResult` — Result of conversation pruning.
- `pub enum BlockKind` — The kind of conversation block.
- `pub enum Resolution` — Resolution level for a conversation block (LOD tier).

## Functions
- `fn classify_block(role: &str, content: &str, tool_name: Option<&str>) -> BlockKind` — Classify a message block from its role and content.
- `fn infer_dependencies(blocks: &[ConvBlock]) -> Vec<Vec<usize>>` — Infer causal dependencies between blocks.  Heuristics: 1. ToolResult[i] depends on ToolCall[i-1] (if adjacent) 2. AssistantMessage[i] depends on the most recent ToolResult before it 3. Every block dep
- `fn build_forward_refs(deps: &[Vec<usize>]) -> HashMap<usize, usize>` — Build forward reference counts from dependency edges.
- `fn score_block(
    block: &ConvBlock,
    block_pos: usize,
    forward_value: f64,
    n_blocks: usize,
    forward_refs: &HashMap<usize, usize>,
    now: f64,
    decay_lambda: f64,
) -> f64` — Score a block's information value for pruning decisions.  w(v) = α·forward_value + β·ref_density + γ·recency + δ·kind_shield - ε·noise_penalty  Weights: α=0.25, β=0.20, γ=0.25, δ=0.25, ε=0.05  Returns
- `fn compute_all_forward_overlaps(blocks: &[ConvBlock]) -> Vec<f64>` — Pre-compute forward bigram overlap for ALL blocks in O(N·W) total.  Processes blocks right-to-left, accumulating a running union of "future" bigrams.  Each block's overlap = fraction of its bigrams fo
- `fn kkt_multichoice_bisect(
    items: &[McItem],
    token_budget: u32,
) -> Vec<Resolution>` — For each block i and resolution level l, define: tokens_after(i,l) = tokens(i) × level.token_fraction() info_cost(i,l)    = value(i) × level.info_loss() efficiency(i,l)   = info_cost(i,l) / tokens_sav
- `fn enforce_dag_coherence(
    deps: &[Vec<usize>],
    assignments: &mut [Resolution],
)` — Enforce causal coherence on the resolution assignments.  Rule: if block B depends on block A, then B.resolution ≥ A.resolution. A response that references a tool result can't be at higher fidelity tha
- `fn protect_recent(
    blocks: &[ConvBlock],
    assignments: &mut [Resolution],
    protect_last_n: usize,
)` — Protect recent blocks: last N non-user blocks stay at Skeleton or better.
- `fn compress_block(block: &ConvBlock, resolution: Resolution) -> String` — Generate compressed content for a block at a given resolution.
- `fn prune_conversation(
    blocks: &[ConvBlock],
    token_budget: u32,
    decay_lambda: f64,
    protect_last: usize,
) -> PruneResult` — Uses multi-choice knapsack via KKT dual bisection with causal DAG coherence enforcement.  Protected blocks (user/system messages) are never compressed.  Recent blocks are kept at Skeleton or better.  
- `fn progressive_thresholds(
    blocks: &[ConvBlock],
    utilization: f64,
    recency_cutoff: usize,
) -> Vec<(usize, Resolution)>` — Returns recommended resolution per block at the current pressure level.  | Utilization | Action                                 | |-------------|----------------------------------------| | < 70%      

## Dependencies
- `crate::entropy::`
- `serde::`
- `std::collections::`
