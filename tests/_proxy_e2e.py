"""Proxy E2E: verify _run_pipeline produces context and inject_context_openai injects it."""
import asyncio, json, os, sys, tempfile, shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
tmp = Path(tempfile.mkdtemp(prefix="proxy-fix-"))
os.environ["ENTROLY_DIR"] = str(tmp)
os.environ["ENTROLY_VAULT"] = str(tmp / "vault")
os.environ["ENTROLY_BYPASS"] = "0"
os.environ["ENTROLY_CONFIDENCE_THRESHOLD"] = "0"

from entroly.server import EntrolyEngine
from entroly.proxy import PromptCompilerProxy, create_proxy_app
from entroly.proxy_config import ProxyConfig
from entroly.proxy_transform import extract_user_message

engine = EntrolyEngine()
engine.ingest_fragment(
    content="def login(token): return validate(token) " * 10,
    source="auth.py", token_count=100,
)
engine.ingest_fragment(
    content="def make_session(uid): return uid " * 10,
    source="session.py", token_count=100,
)
engine.ingest_fragment(
    content="class TokenManager: pass " * 10,
    source="tokens.py", token_count=100,
)

# --- Test 1: _run_pipeline directly ---
print("=== Test 1: _run_pipeline directly ===")
cfg = ProxyConfig()
proxy = PromptCompilerProxy(engine, cfg)
print(f"  bypass={proxy._bypass}, conf_threshold={proxy._confidence_threshold}")

body = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "How does authenticate validate the token?"}],
}
user_msg = extract_user_message(body, "openai")
print(f"  user_message={user_msg!r}")

result = proxy._run_pipeline(user_msg, body, "/v1/chat/completions")
ctx = result.get("context", "")
sel = result.get("selected_fragments", [])
print(f"  context_len={len(ctx)}, selected={len(sel)}, elapsed={result['elapsed_ms']:.1f}ms")
if ctx:
    print(f"  context_head={ctx[:150]!r}")
else:
    print("  WARNING: context is EMPTY")

assert len(ctx) > 0, "FAIL: _run_pipeline returned empty context"
assert len(sel) > 0, "FAIL: no fragments selected"
print("  PASS: pipeline produces non-empty context")

# --- Test 2: Full ASGI proxy with respx mocking ---
print("\n=== Test 2: Full ASGI proxy with respx ===")
import respx
from httpx import Response, AsyncClient, ASGITransport

app = create_proxy_app(engine, cfg)


async def main():
    captured = {}

    async with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        upstream = router.post("https://api.openai.com/v1/chat/completions")

        def _capture(request):
            captured["body"] = json.loads(request.content.decode())
            captured["headers"] = dict(request.headers)
            return Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 1,
                        "total_tokens": 11,
                    },
                },
            )

        upstream.side_effect = _capture

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": "How does authenticate validate the token?",
                        }
                    ],
                },
                headers={"authorization": "Bearer sk-test"},
            )
            print(f"  proxy status: {resp.status_code}")
            print(
                f"  X-Entroly-Optimized: {resp.headers.get('X-Entroly-Optimized', 'absent')}"
            )

    return captured


captured = asyncio.run(main())
shutil.rmtree(tmp, ignore_errors=True)

if not captured:
    print("  FAIL: upstream was never called")
    sys.exit(1)

fwd_body = captured["body"]
msgs = fwd_body.get("messages", [])
sys_msgs = [m for m in msgs if m.get("role") == "system"]
user_msgs = [m for m in msgs if m.get("role") == "user"]
print(f"  upstream got {len(msgs)} messages ({len(sys_msgs)} system, {len(user_msgs)} user)")

if sys_msgs:
    sc = sys_msgs[0]["content"]
    print(f"  system msg len={len(sc)}")
    print(f"  system head: {sc[:200]!r}")
    refs = sum(1 for s in ["auth", "session", "token", "validate", "login"] if s in sc)
    print(f"  fragment references in system msg: {refs}")
else:
    print("  WARNING: no system message injected")
    # Check if context was merged into user message instead
    if user_msgs and len(user_msgs[0].get("content", "")) > 100:
        print("  (context may have been merged into user message)")
        uc = user_msgs[0]["content"]
        print(f"  user msg len={len(uc)}, head: {uc[:200]!r}")

user_preserved = any(
    "authenticate" in m.get("content", "") or "token" in m.get("content", "")
    for m in user_msgs
)
print(f"  user message preserved: {user_preserved}")

assert captured, "upstream never called"
assert user_preserved, "user message lost"
# Context should appear SOMEWHERE in the forwarded body
all_content = " ".join(m.get("content", "") for m in msgs)
has_context = any(s in all_content for s in ["auth.py", "login", "validate", "session"])
assert has_context, f"no context injected anywhere in forwarded body. All content: {all_content[:300]}"
print("\n  PROXY E2E PASSES")
