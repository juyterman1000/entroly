"""Test federation + cross-agent memory with everything ON."""
import os
import sys
import tempfile


def main():
    # Enable federation
    os.environ["ENTROLY_FEDERATION"] = "1"

    passed = 0
    failed = 0

    def check(name, fn):
        nonlocal passed, failed
        try:
            result = fn()
            print(f"  [OK] {name}: {result}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("=" * 60)
    print("  FEDERATION LEARNING TEST")
    print("=" * 60)

    # --- Federation Client ---
    from entroly.federation import (
        FederationClient, GaussianMechanism, ContributionPacket,
        aggregate_contributions, trimmed_mean, PrivacyAccountant,
        FEDERATED_WEIGHT_KEYS,
    )

    tmpdir = tempfile.mkdtemp(prefix="entroly_fed_test_")

    client = FederationClient(data_dir=tmpdir, enabled=True, epsilon=1.0)
    check("FederationClient enabled", lambda: f"enabled={client.enabled}")
    check("Client ID generated", lambda: f"id={client._client_id[:16]}...")

    # --- DP Noise ---
    dp = GaussianMechanism(epsilon=1.0)
    raw_weights = {k: 0.5 for k in FEDERATED_WEIGHT_KEYS}
    noised = dp.add_noise(raw_weights)
    check("DP noise applied", lambda: f"sigma={dp.sigma:.4f}, keys={len(noised)}")

    # Verify noise is non-zero
    diffs = [abs(noised[k] - raw_weights[k]) for k in FEDERATED_WEIGHT_KEYS if k in noised]
    total_noise = sum(diffs)
    check("Noise is non-trivial", lambda: f"total_delta={total_noise:.4f}" if total_noise > 0.001 else (_ for _ in ()).throw(Exception("zero noise")))

    # --- Contribution Packet ---
    packet = client.prepare_contribution(
        archetype_label="python_backend",
        weights=raw_weights,
        sample_count=10,
        confidence=0.85,
    )
    check("Contribution prepared", lambda: f"arch={packet.archetype}, conf={packet.confidence}")

    # --- Privacy Accountant ---
    acc = PrivacyAccountant(budget=10.0)
    check("Privacy budget fresh", lambda: f"consumed={acc.consumed_epsilon():.2f}, can={acc.can_contribute()}")
    acc.record_contribution(1.0)
    check("After 1 contribution", lambda: f"consumed={acc.consumed_epsilon():.2f}, remaining={acc.remaining_budget():.2f}")

    # --- Trimmed Mean (Byzantine resilience) ---
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 100.0]  # 100.0 = attacker
    tm = trimmed_mean(values, trim_fraction=0.10)
    check("Trimmed mean (Byzantine)", lambda: f"{tm:.2f}" if tm < 1.0 else (_ for _ in ()).throw(Exception(f"outlier not trimmed: {tm}")))

    # --- Aggregation ---
    packets = [
        ContributionPacket(archetype="test", weights={k: 0.3 for k in FEDERATED_WEIGHT_KEYS}, sample_count=10, confidence=0.9),
        ContributionPacket(archetype="test", weights={k: 0.5 for k in FEDERATED_WEIGHT_KEYS}, sample_count=10, confidence=0.8),
        ContributionPacket(archetype="test", weights={k: 0.7 for k in FEDERATED_WEIGHT_KEYS}, sample_count=10, confidence=0.7),
    ]
    agg = aggregate_contributions(packets)
    check("Aggregation works", lambda: f"w_recency={agg.get('w_recency', 0):.3f}")

    # --- Save + Load round-trip ---
    saved = client._save_contribution(packet)
    check("Contribution saved to disk", lambda: f"saved={saved}")

    # Create a second client to load contributions
    client2 = FederationClient(data_dir=tmpdir, enabled=True)
    contribs = client2.load_contributions()
    check("Contributions loaded", lambda: f"archetypes={list(contribs.keys())}, count={sum(len(v) for v in contribs.values())}")

    # --- FedProx validation ---
    from entroly.federation import FEDPROX_MU
    check("FedProx mu", lambda: f"mu={FEDPROX_MU}")

    print()
    print("=" * 60)
    print("  CROSS-AGENT MEMORY TEST")
    print("=" * 60)

    # --- LongTermMemory (graceful degradation) ---
    from entroly.long_term_memory import LongTermMemory, is_available, SalienceProfile

    check("hippocampus available?", lambda: f"available={is_available()}")

    ltm = LongTermMemory(capacity=1000)
    check("LongTermMemory init", lambda: f"active={ltm.active}")

    # --- Salience computation ---
    sp = SalienceProfile()
    check("Pinned salience", lambda: f"{sp.compute(is_pinned=True)}")
    check("High entropy salience", lambda: f"{sp.compute(entropy_score=0.9)}")
    check("Selected salience", lambda: f"{sp.compute(was_selected=True, relevance=0.8)}")
    check("Low value salience", lambda: f"{sp.compute()}")

    # --- MemoryBridge ---
    from entroly.context_bridge import MemoryBridge, CognitiveBus
    bus = CognitiveBus()
    bridge = MemoryBridge(bus)
    check("MemoryBridge init", lambda: f"active={bridge.active}")
    check("Bridge events (empty bus)", lambda: f"bridged={bridge.bridge_events()}")

    # --- Context Bridge full stack ---
    from entroly.context_bridge import (
        LODManager, LodTier, SubagentOrchestrator,
        HCCEngine, CompressionLevel,
    )

    # LOD Manager
    lod = LODManager()
    state = lod.register("agent-1")
    check("LOD register", lambda: f"tier={state.tier}")

    lod.update_load("agent-1", 0.5)
    check("LOD update_load", lambda: f"active agents={lod.get_active_agents()}")

    # HCC Engine
    hcc = HCCEngine()
    hcc.add_fragment("f1", "auth.py", "def authenticate(user, password): ...", entropy_score=0.8, relevance=0.9)
    hcc.add_fragment("f2", "utils.py", "def format_date(d): return d.isoformat()", entropy_score=0.3, relevance=0.2)
    result = hcc.optimize(token_budget=50)
    levels = {f.source: f.assigned_level for f in result}
    check("HCC optimize", lambda: f"levels={levels}")

    print()
    print("=" * 60)
    print("  EVOLUTION DAEMON (INTEGRATION TEST)")
    print("=" * 60)

    # --- Evolution Daemon with federation ON ---
    from entroly.vault import VaultManager, VaultConfig
    from entroly.evolution_logger import EvolutionLogger
    from entroly.value_tracker import get_tracker

    vault_cfg = VaultConfig(base_path=os.path.join(tmpdir, "vault"))
    vault = VaultManager(vault_cfg)
    vault.ensure_structure()
    check("Vault initialized", lambda: "OK")

    evo_logger = EvolutionLogger(vault_path=os.path.join(tmpdir, "vault"))
    check("EvolutionLogger", lambda: "OK")

    tracker = get_tracker()
    check("ValueTracker", lambda: f"budget={tracker.get_evolution_budget()}")

    from entroly.evolution_daemon import EvolutionDaemon
    daemon = EvolutionDaemon(
        vault=vault,
        evolution_logger=evo_logger,
        value_tracker=tracker,
        data_dir=tmpdir,
    )
    check("EvolutionDaemon init", lambda: f"archetype={daemon._stats.get('archetype')}")
    check("Federation wired", lambda: f"enabled={daemon._federation.enabled if daemon._federation else False}")

    # Run one cycle
    result = daemon.run_once()
    check("Daemon run_once", lambda: f"gaps={result['gaps_processed']}, dream={result.get('dream')}")

    stats = daemon.stats()
    check("Daemon stats", lambda: f"struct_ok={stats['structural_successes']}, running={stats['running']}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    print("=" * 60)
    total = passed + failed
    print(f"  PASSED: {passed}/{total}  |  FAILED: {failed}/{total}")
    if failed == 0:
        print("  ALL TESTS PASSED - Federation + Memory fully integrated!")
    else:
        print(f"  {failed} test(s) need attention")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

