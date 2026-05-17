"""Quick smoke test: proxy imports + logprob extraction + fusion."""
import json
from entroly.proxy import (
    PromptCompilerProxy,
    _extract_logprobs_from_sse,
    _extract_text_from_sse,
)

# 1. Test logprob extraction from SSE
sse_data = json.dumps({
    "choices": [{
        "delta": {"content": "Hello"},
        "logprobs": {"content": [{"token": "Hello", "logprob": -0.3}]}
    }]
})
sse = f"data: {sse_data}\ndata: [DONE]\n".encode()

text = _extract_text_from_sse(sse)
lps, toks = _extract_logprobs_from_sse(sse)
print(f"Text: {text!r}")
print(f"Logprobs: {lps}, Tokens: {toks}")
assert text == "Hello", f"Expected 'Hello', got {text!r}"
assert len(lps) == 1 and lps[0] == -0.3, f"Expected [-0.3], got {lps}"
assert toks == ["Hello"], f"Expected ['Hello'], got {toks}"
print("  logprob extraction: PASS")

# 2. Test spectral + EPR modules
from entroly.ravs.spectral import compute_spectral_consistency
from entroly.ravs.epr import compute_epr, compute_fused_risk

spec = compute_spectral_consistency(
    "The Eiffel Tower in Paris was built in 1889.",
    "The Eiffel Tower was built in 1887 by Gustave Eiffel."
)
print(f"  spectral: score={spec.score:.3f}, entities ctx={spec.n_ctx_entities} resp={spec.n_resp_entities}")

epr = compute_epr("I think the answer is probably around 42.")
print(f"  epr heuristic: risk={epr.risk_score:.3f}, has_logprobs={epr.has_logprobs}")

epr_lp = compute_epr(
    "The tower is 324 meters tall.",
    logprobs=[-0.1, -0.3, -0.2, -2.5, -0.1, -0.1],
    token_texts=["The", " tower", " is", " 324", " meters", " tall"],
)
print(f"  epr logprobs: risk={epr_lp.risk_score:.3f}, has_logprobs={epr_lp.has_logprobs}, entity_entropy={epr_lp.entity_entropy:.3f}")

# 3. Test fusion
fused = compute_fused_risk(0.3, epr_lp, entity_gap=0.2)
print(f"  fused risk: {fused.fused_risk:.3f}")

# 4. Verify RAVS __init__ exports everything
from entroly.ravs import (
    compute_epr, compute_fused_risk, EPRSignal, FusedHallucinationSignal,
    compute_spectral_consistency, SpectralSignal,
    EpistemicCascadeEngine, compute_fisher_curvature,
)
print("  RAVS V5+V6+V7 exports: PASS")

print("\nAll smoke tests PASSED")
