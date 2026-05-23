# Entroly Repo File Map

Canonical ownership map across the Python product shell, Rust core, and WASM/JS surface.

## root

| Path | Role | Category |
|---|---|---|
| `.dockerignore` | top-level support artifact | `root-support` |
| `.gitignore` | top-level support artifact | `root-support` |
| `all_files_list.txt` | generated repo inventory artifact | `root-artifact` |
| `arch_summary.txt` | generated architecture summary artifact | `root-artifact` |
| `build_errors.txt` | generated build log artifact | `root-artifact` |
| `build_output.txt` | generated build log artifact | `root-artifact` |
| `CONTRIBUTING.md` | contribution workflow | `root-doc` |
| `docker-compose.yml` | local multi-service runtime | `root-ops` |
| `Dockerfile.entroly` | container runtime packaging | `root-ops` |
| `extractor.py` | one-off code extraction utility | `root-tool` |
| `extractor_cogops.py` | one-off CogOps extraction utility | `root-tool` |
| `job_logs.txt` | generated job log artifact | `root-artifact` |
| `LICENSE` | license | `root-doc` |
| `pyproject.toml` | top-level Python packaging and workspace metadata | `root-meta` |
| `README.md` | product positioning and install surface | `root-doc` |
| `ruff.toml` | top-level support artifact | `root-support` |
| `ruff_errors.txt` | generated lint artifact | `root-artifact` |
| `SECURITY.md` | security policy | `root-doc` |
| `super_dump.txt` | generated repository dump artifact | `root-artifact` |
| `super_extractor.py` | one-off repo summarization utility | `root-tool` |
| `test_auth.py` | auth-focused regression test | `root-test` |
| `test_cogops_smoke.py` | Python CogOps integration smoke test | `root-test` |
| `test_output.txt` | generated test artifact | `root-artifact` |
| `test_output2.txt` | generated test artifact | `root-artifact` |
| `test_rust_cogops.py` | Rust CogOps integration smoke test | `root-test` |
| `tuning_config.json` | shared tuning defaults | `root-config` |

## python

| Path | Role | Category |
|---|---|---|
| `entroly/__init__.py` | package entry metadata | `python-meta` |
| `entroly/_docker_launcher.py` | Docker launcher shim | `python-runtime` |
| `entroly/adaptive_pruner.py` | conversation/context pruning policy | `python-support` |
| `entroly/auto_index.py` | workspace discovery and raw ingest indexing | `python-support` |
| `entroly/autotune.py` | parameter tuning and feedback journal | `python-support` |
| `entroly/belief_compiler.py` | Truth to Belief compiler | `python-cogops` |
| `entroly/benchmark_harness.py` | benchmark execution harness | `python-support` |
| `entroly/change_listener.py` | workspace change sync and listener glue | `python-cogops` |
| `entroly/change_pipeline.py` | change-driven PR and review pipeline | `python-cogops` |
| `entroly/checkpoint.py` | checkpoint persistence and recovery | `python-support` |
| `entroly/cli.py` | primary CLI and operator surface | `python-runtime` |
| `entroly/config.py` | project configuration and paths | `python-support` |
| `entroly/context_bridge.py` | multi-agent context and orchestration surface | `python-runtime` |
| `entroly/dashboard.py` | developer-facing runtime dashboard | `python-runtime` |
| `entroly/entroly_mcp_client.py` | example MCP client | `python-example` |
| `entroly/epistemic_router.py` | ingress routing policy engine | `python-cogops` |
| `entroly/evolution_logger.py` | miss tracking and capability-gap logging | `python-cogops` |
| `entroly/flow_orchestrator.py` | canonical flow executor | `python-cogops` |
| `entroly/integrate_entroly_mcp.py` | example MCP integration | `python-example` |
| `entroly/long_term_memory.py` | cross-session memory adapter | `python-support` |
| `entroly/multimodal.py` | image, diff, diagram, and voice ingestion | `python-support` |
| `entroly/npm/index.js` | Python package support file | `python-support` |
| `entroly/npm/package.json` | Python package support file | `python-support` |
| `entroly/npm/README.md` | package-level README | `python-doc` |
| `entroly/prefetch.py` | predictive dependency and file prefetch | `python-support` |
| `entroly/provenance.py` | provenance graph and context trace builder | `python-support` |
| `entroly/proxy.py` | HTTP proxy prompt compiler runtime | `python-runtime` |
| `entroly/proxy_config.py` | proxy quality and model budget config | `python-support` |
| `entroly/proxy_transform.py` | provider-specific context injection | `python-support` |
| `entroly/pyproject.toml` | package metadata | `python-meta` |
| `entroly/query_refiner.py` | query shaping and refinement | `python-support` |
| `entroly/README.md` | package-level README | `python-doc` |
| `entroly/repo_map.py` | canonical repo inventory and ownership map | `python-cogops` |
| `entroly/server.py` | primary Python MCP server and product shell | `python-runtime` |
| `entroly/skill_engine.py` | dynamic skill synthesis and lifecycle | `python-cogops` |
| `entroly/tuning_config.json` | package tuning defaults | `python-config` |
| `entroly/value_tracker.py` | cost and value accounting | `python-support` |
| `entroly/vault.py` | vault persistence and artifact schema | `python-cogops` |
| `entroly/verification_engine.py` | belief verification and confidence engine | `python-cogops` |

## rust-core

| Path | Role | Category |
|---|---|---|
| `entroly-core/Cargo.lock` | Rust core package metadata | `rust-meta` |
| `entroly-core/Cargo.toml` | Rust core package metadata | `rust-meta` |
| `entroly-core/pyproject.toml` | Rust core package metadata | `rust-meta` |
| `entroly-core/README.md` | Rust core package metadata | `rust-meta` |
| `entroly-core/src/anomaly.rs` | anomaly detection | `rust-verification` |
| `entroly-core/src/cache.rs` | EGSC cache and retrieval economics | `rust-core` |
| `entroly-core/src/causal.rs` | causal context graph and intervention logic | `rust-belief` |
| `entroly-core/src/channel.rs` | channel-coding reward and contradiction logic | `rust-learning` |
| `entroly-core/src/cognitive_bus.rs` | inter-agent event bus | `rust-action` |
| `entroly-core/src/cogops.rs` | Rust epistemic engine and CogOps data plane | `rust-cogops` |
| `entroly-core/src/conversation_pruner.rs` | conversation compression runtime | `rust-action` |
| `entroly-core/src/dedup.rs` | SimHash deduplication | `rust-core` |
| `entroly-core/src/depgraph.rs` | dependency graph extraction | `rust-core` |
| `entroly-core/src/entropy.rs` | information density scoring | `rust-core` |
| `entroly-core/src/fragment.rs` | context fragment model and scoring helpers | `rust-core` |
| `entroly-core/src/guardrails.rs` | criticality, safety, and ordering policy | `rust-verification` |
| `entroly-core/src/health.rs` | codebase health analysis | `rust-verification` |
| `entroly-core/src/hierarchical.rs` | hierarchical compression | `rust-belief` |
| `entroly-core/src/knapsack.rs` | budgeted context selection optimizer | `rust-core` |
| `entroly-core/src/knapsack_sds.rs` | streaming/diverse selection and IOS logic | `rust-core` |
| `entroly-core/src/lib.rs` | core Rust engine and PyO3 export surface | `rust-runtime` |
| `entroly-core/src/lsh.rs` | approximate recall index | `rust-core` |
| `entroly-core/src/nkbe.rs` | multi-agent token budget allocator | `rust-action` |
| `entroly-core/src/prism.rs` | reinforcement and spectral optimizer | `rust-learning` |
| `entroly-core/src/query.rs` | query analysis and refinement heuristics | `rust-action` |
| `entroly-core/src/query_persona.rs` | query manifold and archetype modeling | `rust-learning` |
| `entroly-core/src/resonance.rs` | pairwise fragment resonance modeling | `rust-learning` |
| `entroly-core/src/sast.rs` | static security analysis engine | `rust-verification` |
| `entroly-core/src/semantic_dedup.rs` | semantic deduplication refinement | `rust-core` |
| `entroly-core/src/skeleton.rs` | multi-language structure extraction | `rust-truth` |
| `entroly-core/src/utilization.rs` | fragment utilization scoring | `rust-verification` |
| `entroly-core/tests/test_brutal.py` | Rust core integration test driver | `rust-test` |
| `entroly-core/tests/test_integration.py` | Rust core integration test driver | `rust-test` |

## wasm

| Path | Role | Category |
|---|---|---|
| `entroly-wasm/build_output.txt` | WASM package support file | `wasm-support` |
| `entroly-wasm/Cargo.lock` | WASM package support file | `wasm-support` |
| `entroly-wasm/Cargo.toml` | WASM package support file | `wasm-support` |
| `entroly-wasm/index.js` | WASM package support file | `wasm-support` |
| `entroly-wasm/js/auto_index.js` | Node workspace indexing wrapper | `wasm-support` |
| `entroly-wasm/js/autotune.js` | Node autotune wrapper | `wasm-support` |
| `entroly-wasm/js/checkpoint.js` | Node checkpoint wrapper | `wasm-support` |
| `entroly-wasm/js/cli.js` | Node CLI over WASM engine | `wasm-runtime` |
| `entroly-wasm/js/config.js` | Node configuration wrapper | `wasm-support` |
| `entroly-wasm/js/server.js` | Node MCP server over WASM engine | `wasm-runtime` |
| `entroly-wasm/package.json` | WASM package support file | `wasm-support` |
| `entroly-wasm/pkg/.gitignore` | WASM package support file | `wasm-support` |
| `entroly-wasm/pkg/entroly_wasm.d.ts` | WASM package support file | `wasm-support` |
| `entroly-wasm/pkg/entroly_wasm.js` | WASM package support file | `wasm-support` |
| `entroly-wasm/pkg/entroly_wasm_bg.wasm` | WASM package support file | `wasm-support` |
| `entroly-wasm/pkg/entroly_wasm_bg.wasm.d.ts` | WASM package support file | `wasm-support` |
| `entroly-wasm/pkg/package.json` | WASM package support file | `wasm-support` |
| `entroly-wasm/src/anomaly.rs` | anomaly detection | `rust-verification` |
| `entroly-wasm/src/cache.rs` | EGSC cache and retrieval economics | `rust-core` |
| `entroly-wasm/src/causal.rs` | causal context graph and intervention logic | `rust-belief` |
| `entroly-wasm/src/channel.rs` | channel-coding reward and contradiction logic | `rust-learning` |
| `entroly-wasm/src/cognitive_bus.rs` | inter-agent event bus | `rust-action` |
| `entroly-wasm/src/conversation_pruner.rs` | conversation compression runtime | `rust-action` |
| `entroly-wasm/src/dedup.rs` | SimHash deduplication | `rust-core` |
| `entroly-wasm/src/depgraph.rs` | dependency graph extraction | `rust-core` |
| `entroly-wasm/src/entropy.rs` | information density scoring | `rust-core` |
| `entroly-wasm/src/fragment.rs` | context fragment model and scoring helpers | `rust-core` |
| `entroly-wasm/src/guardrails.rs` | criticality, safety, and ordering policy | `rust-verification` |
| `entroly-wasm/src/health.rs` | codebase health analysis | `rust-verification` |
| `entroly-wasm/src/hierarchical.rs` | hierarchical compression | `rust-belief` |
| `entroly-wasm/src/knapsack.rs` | budgeted context selection optimizer | `rust-core` |
| `entroly-wasm/src/knapsack_sds.rs` | streaming/diverse selection and IOS logic | `rust-core` |
| `entroly-wasm/src/lib.rs` | WASM export surface for the Rust engine | `wasm-runtime` |
| `entroly-wasm/src/lsh.rs` | approximate recall index | `rust-core` |
| `entroly-wasm/src/nkbe.rs` | multi-agent token budget allocator | `rust-action` |
| `entroly-wasm/src/prism.rs` | reinforcement and spectral optimizer | `rust-learning` |
| `entroly-wasm/src/query.rs` | query analysis and refinement heuristics | `rust-action` |
| `entroly-wasm/src/query_persona.rs` | query manifold and archetype modeling | `rust-learning` |
| `entroly-wasm/src/resonance.rs` | pairwise fragment resonance modeling | `rust-learning` |
| `entroly-wasm/src/sast.rs` | static security analysis engine | `rust-verification` |
| `entroly-wasm/src/semantic_dedup.rs` | semantic deduplication refinement | `rust-core` |
| `entroly-wasm/src/skeleton.rs` | multi-language structure extraction | `rust-truth` |
| `entroly-wasm/src/utilization.rs` | fragment utilization scoring | `rust-verification` |
| `entroly-wasm/test_autotune.js` | WASM package support file | `wasm-support` |
| `entroly-wasm/test_output.txt` | WASM package support file | `wasm-support` |
| `entroly-wasm/test_wasm_e2e.js` | WASM package support file | `wasm-support` |

## tests

| Path | Role | Category |
|---|---|---|
| `tests/functional_test.py` | Python integration or functional test | `python-test` |
| `tests/test_apa.py` | Python integration or functional test | `python-test` |
| `tests/test_change_listener.py` | Python integration or functional test | `python-test` |
| `tests/test_context_bridge.py` | Python integration or functional test | `python-test` |
| `tests/test_deep_functional.py` | Python integration or functional test | `python-test` |
| `tests/test_e2e.py` | Python integration or functional test | `python-test` |
| `tests/test_ecc.py` | Python integration or functional test | `python-test` |
| `tests/test_egtc_v2.py` | Python integration or functional test | `python-test` |
| `tests/test_functional.py` | Python integration or functional test | `python-test` |
| `tests/test_intensive_functional.py` | Python integration or functional test | `python-test` |
| `tests/test_ios.py` | Python integration or functional test | `python-test` |
| `tests/test_mcp_protocol.py` | Python integration or functional test | `python-test` |
| `tests/test_production_e2e.py` | Python integration or functional test | `python-test` |
| `tests/test_proxy_providers.py` | Python integration or functional test | `python-test` |
| `tests/test_real_user.py` | Python integration or functional test | `python-test` |
| `tests/test_repo_map.py` | Python integration or functional test | `python-test` |

## other

| Path | Role | Category |
|---|---|---|
| `.claude/settings.json` | support file | `support` |
| `.claude/settings.local.json` | support file | `support` |
| `.devcontainer/devcontainer.json` | support file | `support` |
| `.entroly/vault/beliefs/_docker_launcher_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/adaptive_pruner_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/anomaly_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/auto_index_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/autotune_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/belief_compiler_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/benchmark_harness_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/cache_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/causal_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/change_pipeline_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/channel_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/checkpoint_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/cli_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/cognitive_bus_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/cogops_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/compare_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/config_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/context_bridge_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/conversation_pruner_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/dashboard_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/dedup_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/demo_full_experience_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/demo_value_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/depgraph_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/entroly_wasm.d_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/entroly_wasm_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/entropy_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/epistemic_router_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/evaluate_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/evolution_logger_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/extractor_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/extractor_cogops_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/flow_orchestrator_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/fragment_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/functional_test_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/generate_demo_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/guardrails_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/health_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/hierarchical_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/knapsack_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/knapsack_sds_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/lib_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/long_term_memory_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/lsh_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/multimodal_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/nkbe_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/openclaw_benchmark_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/prefetch_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/prism_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/provenance_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/proxy_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/proxy_config_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/proxy_transform_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/query_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/query_persona_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/query_refiner_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/resonance_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/sast_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/semantic_dedup_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/server_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/skeleton_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/skill_engine_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/super_extractor_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_apa_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_auth_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_autotune_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_brutal_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_cogops_smoke_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_context_bridge_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_deep_functional_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_e2e_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_ecc_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_egtc_v2_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_functional_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_integration_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_intensive_functional_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_ios_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_mcp_protocol_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_production_e2e_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_proxy_providers_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_real_user_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_rust_cogops_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/test_wasm_e2e_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/utilization_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/value_tracker_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/vault_18a336a7.md` | support file | `support` |
| `.entroly/vault/beliefs/verification_engine_18a336a7.md` | support file | `support` |
| `.entroly/vault/evolution/registry.md` | support file | `support` |
| `.github/dependabot.yml` | support file | `support` |
| `.github/workflows/ci.yml` | support file | `support` |
| `.github/workflows/docker-publish.yml` | support file | `support` |
| `.github/workflows/entroly-publish.yml` | support file | `support` |
| `.github/workflows/publish-core-wheels.yml` | support file | `support` |
| `bench/__init__.py` | documentation, example, or benchmark asset | `support` |
| `bench/autotune.py` | documentation, example, or benchmark asset | `support` |
| `bench/cases.json` | documentation, example, or benchmark asset | `support` |
| `bench/compare.py` | documentation, example, or benchmark asset | `support` |
| `bench/evaluate.py` | documentation, example, or benchmark asset | `support` |
| `bench/results.tsv` | documentation, example, or benchmark asset | `support` |
| `bench/tuning_config.json` | documentation, example, or benchmark asset | `support` |
| `benchmarks/openclaw_benchmark.py` | documentation, example, or benchmark asset | `support` |
| `dist/entroly-0.5.5-py3-none-any.whl` | support file | `support` |
| `dist/entroly-0.5.5.tar.gz` | support file | `support` |
| `docs/assets/demo.html` | documentation, example, or benchmark asset | `support` |
| `docs/assets/demo.svg` | documentation, example, or benchmark asset | `support` |
| `docs/assets/demo_animated.svg` | documentation, example, or benchmark asset | `support` |
| `docs/assets/logo.png` | documentation, example, or benchmark asset | `support` |
| `docs/assets/openclaw_benchmark.png` | documentation, example, or benchmark asset | `support` |
| `docs/assets/pipeline.svg` | documentation, example, or benchmark asset | `support` |
| `docs/assets/value.svg` | documentation, example, or benchmark asset | `support` |
| `docs/generate_demo.py` | documentation, example, or benchmark asset | `support` |
| `examples/demo_full_experience.py` | documentation, example, or benchmark asset | `support` |
| `examples/demo_value.py` | documentation, example, or benchmark asset | `support` |
| `tuning_strategies/balanced.md` | documentation, example, or benchmark asset | `support` |
| `tuning_strategies/latency.md` | documentation, example, or benchmark asset | `support` |
| `tuning_strategies/monorepo.md` | documentation, example, or benchmark asset | `support` |
| `tuning_strategies/quality.md` | documentation, example, or benchmark asset | `support` |

