"""
Entroly Prompt Compiler Proxy
==============================

An invisible HTTP reverse proxy that sits between the IDE and the LLM API.
Intercepts every request, optimizes the prompt using entroly's algorithms,
and forwards the enriched request to the real API.

The developer changes one setting (API base URL → localhost:9377) and every
query is automatically optimized. No MCP tools to call. No behavior change.

Architecture:
    IDE → localhost:9377 → entroly pipeline (3-6ms) → real API → stream back

All heavy computation runs in Rust (PyO3). The proxy adds <10ms latency.
Errors fall back to forwarding the original request unmodified.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .proxy_config import ProxyConfig
from .proxy_transform import (
    apply_temperature,
    apply_trajectory_convergence,
    compute_optimal_temperature,
    compute_token_budget,
    detect_provider,
    extract_model,
    extract_user_message,
    format_context_block,
    inject_context_anthropic,
    inject_context_openai,
)

logger = logging.getLogger("entroly.proxy")


class PromptCompilerProxy:
    """HTTP reverse proxy that optimizes every LLM request with entroly."""

    def __init__(self, engine: Any, config: Optional[ProxyConfig] = None):
        self.engine = engine
        self.config = config or ProxyConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._requests_total: int = 0
        self._requests_optimized: int = 0
        self._temperature_sum: float = 0.0
        self._temperature_count: int = 0
        self._last_temperature: Optional[float] = None
        self._trajectory_turn_count: int = 0

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            follow_redirects=True,
        )
        logger.info("Prompt compiler proxy ready")

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def handle_proxy(self, request: Request) -> StreamingResponse | JSONResponse:
        """Main proxy handler — intercept, optimize, forward."""
        self._requests_total += 1

        # Read request
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON — forward raw (e.g. health checks hitting wrong path)
            return await self._forward_raw(request, body_bytes)

        path = request.url.path
        headers = {k: v for k, v in request.headers.items()}
        provider = detect_provider(path, headers)

        # Run the optimization pipeline (synchronous Rust, off the event loop)
        try:
            user_message = extract_user_message(body, provider)
            if user_message:
                pipeline_result = await asyncio.to_thread(
                    self._run_pipeline, user_message, body
                )
                context_text = pipeline_result["context"]
                pipeline_ms = pipeline_result["elapsed_ms"]
                optimal_tau = pipeline_result.get("temperature")

                if context_text:
                    if provider == "openai":
                        body = inject_context_openai(body, context_text)
                    else:
                        body = inject_context_anthropic(body, context_text)

                    # EGTC v2: apply Fisher-derived optimal temperature
                    if self.config.enable_temperature_calibration and optimal_tau is not None:
                        # Apply trajectory convergence (temperature decays
                        # across conversation turns as task crystallises)
                        if self.config.enable_trajectory_convergence:
                            optimal_tau = apply_trajectory_convergence(
                                optimal_tau,
                                self._trajectory_turn_count,
                                c_min=self.config.trajectory_c_min,
                                lam=self.config.trajectory_lambda,
                            )
                        body = apply_temperature(body, optimal_tau)
                        self._temperature_sum += optimal_tau
                        self._temperature_count += 1
                        self._last_temperature = optimal_tau
                        self._trajectory_turn_count += 1

                    self._requests_optimized += 1
                    tau_str = f", τ={optimal_tau:.2f}" if optimal_tau else ""
                    logger.info(
                        f"Optimized in {pipeline_ms:.1f}ms{tau_str} "
                        f"({self._requests_optimized}/{self._requests_total} requests)"
                    )
        except Exception as e:
            # Cardinal rule: never block a request due to entroly errors
            logger.debug(f"Pipeline error (forwarding unmodified): {e}")

        # Forward to real API
        target_url = self._resolve_target(provider, path)
        forward_headers = self._build_headers(headers, provider)
        is_streaming = body.get("stream", False)

        if is_streaming:
            return await self._stream_response(target_url, forward_headers, body)
        else:
            return await self._forward_response(target_url, forward_headers, body)

    def _run_pipeline(self, user_message: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Run the synchronous optimization pipeline. Called via asyncio.to_thread.

        Returns dict with keys: context, elapsed_ms, temperature.
        """
        t0 = time.perf_counter()

        model = extract_model(body)
        token_budget = compute_token_budget(model, self.config)

        # optimize_context already does:
        #   1. Query refinement (py_analyze_query + py_refine_heuristic)
        #   2. LTM recall (cross-session memories)
        #   3. Knapsack optimization (Rust)
        #   4. SSSL filtering
        #   5. Ebbinghaus decay bookkeeping
        self.engine._turn_counter += 1
        self.engine.advance_turn()
        result = self.engine.optimize_context(token_budget, user_message)

        selected = result.get("selected_fragments", [])
        refinement = result.get("query_refinement")

        # Build refinement info for the context block
        refinement_info = None
        # Extract vagueness from query_analysis (always present) rather than
        # query_refinement (only present when vagueness >= 0.45)
        query_analysis = result.get("query_analysis", {})
        vagueness = query_analysis.get("vagueness_score", 0.0)
        if refinement:
            vagueness = max(vagueness, refinement.get("vagueness_score", 0.0))
            refinement_info = {
                "original": refinement.get("original_query", user_message),
                "refined": refinement.get("refined_query", user_message),
                "vagueness": vagueness,
            }

        # ── EGTC v2: Fisher-based Temperature Calibration ──
        optimal_tau = None
        if self.config.enable_temperature_calibration and selected:
            # Signal 1: vagueness (from query_analysis, always available)
            # Signal 2: fragment entropies (now from entropy_score key, Bug #1 fixed)
            fragment_entropies = [
                f.get("entropy_score", 0.5) for f in selected
            ]
            # Signal 3: sufficiency — knapsack fill ratio
            total_tokens_used = sum(f.get("token_count", 0) for f in selected)
            sufficiency = min(1.0, total_tokens_used / max(token_budget, 1))
            # Signal 4: task type from Rust classifier
            task_type = "Unknown"
            try:
                task_info = self.engine._rust.classify_task(user_message)
                task_type = task_info.get("task_type", "Unknown")
            except Exception:
                pass

            optimal_tau = compute_optimal_temperature(
                vagueness=vagueness,
                fragment_entropies=fragment_entropies,
                sufficiency=sufficiency,
                task_type=task_type,
                fisher_scale=self.config.fisher_scale,
            )

        # Security scan on selected fragments
        security_issues: list[str] = []
        if self.config.enable_security_scan and self.engine._guard.available:
            for frag in selected:
                content = frag.get("preview", frag.get("content", ""))
                source = frag.get("source", "")
                issues = self.engine._guard.scan(content, source)
                for issue in issues:
                    security_issues.append(f"[{source}] {issue}")

        # LTM memories (already injected by optimize_context, but we want to
        # show them in the context block for transparency)
        ltm_memories: list[dict] = []
        if self.config.enable_ltm and self.engine._ltm.active:
            ltm_memories = self.engine._ltm.recall_relevant(
                user_message, top_k=3, min_retention=0.3
            )

        # Format the context block
        context_text = format_context_block(
            selected, security_issues, ltm_memories, refinement_info
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if selected:
            total_tokens = sum(f.get("token_count", 0) for f in selected)
            tau_str = f", τ={optimal_tau:.4f}" if optimal_tau else ""
            logger.info(
                f"Pipeline: {elapsed_ms:.1f}ms, "
                f"{len(selected)} fragments, {total_tokens} tokens{tau_str}"
            )

        return {
            "context": context_text,
            "elapsed_ms": elapsed_ms,
            "temperature": optimal_tau,
        }

    async def _stream_response(
        self, url: str, headers: Dict[str, str], body: Dict[str, Any]
    ) -> StreamingResponse:
        """Forward a streaming request and proxy the SSE response."""
        async def event_generator():
            async with self._client.stream(
                "POST", url, json=body, headers=headers
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

        resp_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Entroly-Optimized": "true",
        }
        if self._last_temperature is not None:
            resp_headers["X-Entroly-Temperature"] = f"{self._last_temperature:.4f}"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=resp_headers,
        )

    async def _forward_response(
        self, url: str, headers: Dict[str, str], body: Dict[str, Any]
    ) -> JSONResponse:
        """Forward a non-streaming request."""
        response = await self._client.post(url, json=body, headers=headers)
        resp_headers: Dict[str, str] = {"X-Entroly-Optimized": "true"}
        if self._last_temperature is not None:
            resp_headers["X-Entroly-Temperature"] = f"{self._last_temperature:.4f}"
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers=resp_headers,
        )

    async def _forward_raw(
        self, request: Request, body_bytes: bytes
    ) -> JSONResponse:
        """Forward a raw (non-JSON) request."""
        return JSONResponse(
            {"error": "invalid request body"}, status_code=400
        )

    def _resolve_target(self, provider: str, path: str) -> str:
        if provider == "anthropic":
            return f"{self.config.anthropic_base_url}{path}"
        return f"{self.config.openai_base_url}{path}"

    def _build_headers(
        self, original: Dict[str, str], provider: str
    ) -> Dict[str, str]:
        """Build headers for the forwarded request. Pass through auth."""
        forward: Dict[str, str] = {"Content-Type": "application/json"}
        if "authorization" in original:
            forward["Authorization"] = original["authorization"]
        if "x-api-key" in original:
            forward["x-api-key"] = original["x-api-key"]
        if "anthropic-version" in original:
            forward["anthropic-version"] = original["anthropic-version"]
        return forward


async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "entroly-proxy"})


async def _proxy_stats(request: Request) -> JSONResponse:
    proxy = request.app.state.proxy
    stats: Dict[str, Any] = {
        "requests_total": proxy._requests_total,
        "requests_optimized": proxy._requests_optimized,
        "optimization_rate": (
            f"{proxy._requests_optimized / max(proxy._requests_total, 1):.0%}"
        ),
    }
    if proxy._temperature_count > 0:
        stats["egtc"] = {
            "enabled": proxy.config.enable_temperature_calibration,
            "avg_temperature": round(proxy._temperature_sum / proxy._temperature_count, 4),
            "last_temperature": proxy._last_temperature,
            "calibrations": proxy._temperature_count,
        }
    return JSONResponse(stats)


def create_proxy_app(
    engine: Any, config: Optional[ProxyConfig] = None
) -> Starlette:
    """Create the Starlette ASGI app for the prompt compiler proxy."""
    proxy = PromptCompilerProxy(engine, config)

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", proxy.handle_proxy, methods=["POST"]),
            Route("/v1/messages", proxy.handle_proxy, methods=["POST"]),
            Route("/health", _health),
            Route("/stats", _proxy_stats),
        ],
        on_startup=[proxy.startup],
        on_shutdown=[proxy.shutdown],
    )
    app.state.proxy = proxy
    return app
