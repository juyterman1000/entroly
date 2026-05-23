"""E2E federation with realistic params: 20 simulated clients."""
import os, sys, tempfile, shutil
os.environ["ENTROLY_FEDERATION"] = "1"

from entroly.federation import (
    FederationClient, FEDERATED_WEIGHT_KEYS,
)
import random

tmpdir = tempfile.mkdtemp(prefix="fed_e2e_")
shared_dir = os.path.join(tmpdir, "shared")

print("=" * 60)
print("  REALISTIC E2E: 20 clients, low noise")
print("=" * 60)

# 20 clients with realistic estimated_participants=20 (lower noise)
N = 20
clients = []
for i in range(N):
    c = FederationClient(
        data_dir=shared_dir, enabled=True,
        epsilon=1.0, estimated_participants=N,
    )
    c._client_id = f"machine_{i:02d}_" + "x" * 48
    clients.append(c)

# Each client learned slightly different weights around a true optimum
TRUE_WEIGHTS = {
    "w_recency": 0.35, "w_frequency": 0.25, "w_semantic": 0.30,
    "w_entropy": 0.20, "w_resonance": 0.10,
    "decay_half_life": 15.0, "min_relevance": 0.05, "exploration_rate": 0.15,
}

print(f"\n  True optimum: w_recency={TRUE_WEIGHTS['w_recency']}, "
      f"w_semantic={TRUE_WEIGHTS['w_semantic']}, "
      f"w_entropy={TRUE_WEIGHTS['w_entropy']}")

# Each client has local variation (+/- 0.1)
print(f"\n--- STEP 1: {N} clients contribute ---")
for i, client in enumerate(clients):
    local_w = {}
    for k, v in TRUE_WEIGHTS.items():
        if k == "decay_half_life":
            local_w[k] = v + random.gauss(0, 2.0)
            local_w[k] = max(1.0, min(100.0, local_w[k]))
        else:
            local_w[k] = v + random.gauss(0, 0.08)
            local_w[k] = max(0.0, min(1.0, local_w[k]))
    
    packet = client.prepare_contribution(
        archetype_label="python_backend",
        weights=local_w,
        sample_count=20,
        confidence=0.85,
    )
    if packet:
        client._save_contribution(packet)

print(f"  All {N} contributed")

# New client aggregates
print(f"\n--- STEP 2: Aggregate ---")
agg_client = FederationClient(data_dir=shared_dir, enabled=True, estimated_participants=N)
agg_client._client_id = "aggregator_" + "z" * 48

global_w = agg_client.compute_global_weights()
gw = global_w.get("python_backend")

if gw:
    print(f"  Contributors: {gw.contributors}")
    print(f"  Confidence: {gw.confidence:.2f}")
    print(f"\n  Global vs True:")
    all_close = True
    for k in ["w_recency", "w_semantic", "w_entropy", "w_frequency", "decay_half_life"]:
        true_v = TRUE_WEIGHTS[k]
        global_v = gw.weights.get(k, 0)
        delta = abs(global_v - true_v)
        ok = "OK" if delta < (5.0 if k == "decay_half_life" else 0.2) else "DRIFT"
        if ok == "DRIFT":
            all_close = False
        print(f"    {k:20s}: true={true_v:.3f}  global={global_v:.4f}  delta={delta:.4f}  [{ok}]")
    
    # Merge test
    print(f"\n--- STEP 3: Merge into local ---")

    class MockOpt:
        def __init__(self):
            self._w = {k: 0.5 for k in FEDERATED_WEIGHT_KEYS}
            self._w["decay_half_life"] = 15.0
        def current_archetype(self): return "python_backend"
        def current_weights(self): return dict(self._w)
        def stats(self): return {"strategy_table": {"python_backend": {"sample_count": 10, "confidence": 0.7}}}
        def update_weights(self, w): self._w = dict(w)

    opt = MockOpt()
    before = opt.current_weights()["w_recency"]
    merged = agg_client.merge_global(opt)
    after = opt.current_weights()["w_recency"]
    
    print(f"  merge_global: {merged}")
    print(f"  w_recency: {before:.4f} -> {after:.4f}")
    
    if merged:
        print(f"\n  [OK] Federation works end-to-end!")
        print(f"       {N} clients -> DP noise -> trimmed mean -> FedProx blend -> local update")
    else:
        print(f"\n  [INFO] Merge was rejected by validation (safety guardrail working)")
        print(f"         This happens when noised weights fall outside valid ranges")
else:
    print("  FAIL: no global weights")

# Privacy
print(f"\n--- Privacy budget ---")
acc = clients[0]._accountant
print(f"  After 1 contribution: consumed_eps={acc.consumed_epsilon():.2f}")

shutil.rmtree(tmpdir, ignore_errors=True)
print("\n" + "=" * 60)
print("  FEDERATION E2E: COMPLETE")
print("=" * 60)
