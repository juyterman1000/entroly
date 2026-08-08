# Recoverable image context optimization

Image transformation is disabled by default. Enable the optional dependency and proxy flag explicitly:

```bash
pip install "entroly[images]"
ENTROLY_OPTIMIZE_IMAGES=1 entroly proxy
```

Entroly recognizes inline base64 images in OpenAI, Anthropic, and Gemini request shapes. It estimates provider image tokens, applies the existing quality gate, and transforms only when the provider-specific estimate decreases. Before the request is mutated, the exact original bytes are stored content-addressably under `~/.entroly/image-recovery/` with a digest-verifiable transform receipt.

Response headers include `X-Entroly-Image-Optimization`, `X-Entroly-Image-Receipt-Count`, and bounded receipt IDs. If decoding, Pillow, optimization, or recovery persistence fails, the original request is preserved.

Recover an exact original through the authenticated local sidecar endpoint:

```bash
curl "http://127.0.0.1:9377/retrieve-image?receipt_id=img:..." \
  -H "Accept: application/json"
```

The response contains the verified source digest and `original_base64`, never the local object path. The route inherits the same sidecar guard as `/retrieve` and returns `Cache-Control: no-store`. If `ENTROLY_SIDECAR_TOKEN` is configured, send it as `X-Entroly-Sidecar-Token`. Explicit remote-proxy mode additionally requires the proxy-wide `X-Entroly-Access-Token` capability and trusted transport.

Important provider boundary: reducing encoded image bytes does not necessarily reduce billed image tokens. OpenAI and Anthropic estimators account for provider-side resizing, so Entroly can legitimately preserve an oversized transport image when resizing would not lower the estimated token count. Current tests demonstrate a token-reducing Gemini tile case and preservation cases for the other estimators. This is not a universal image-savings claim.

Controls:

- `ENTROLY_IMAGE_MIN_QUALITY_RATIO` — minimum retained pixel-area ratio, default `0.72`.
- `ENTROLY_IMAGE_MAX_BYTES` — maximum original bytes accepted by the recovery store, default 20 MiB.
- `ENTROLY_DIR` — parent for local recovery state.
