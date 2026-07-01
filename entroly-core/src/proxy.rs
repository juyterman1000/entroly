//! Single-binary HTTP proxy for `entroly-rs` (Phase 2 — multi-provider).
//!
//! The request-body transform ([`compress_request_body`]) is **pure**
//! (`serde_json` + [`crate::compress`]) and always compiled + unit-tested. Only
//! the server ([`run`]) is gated behind the `proxy` feature, so its deps
//! (`tiny_http`, `ureq`) never enter the Python wheel.
//!
//! Providers (auto-detected from the request path):
//!   * **Anthropic** `/v1/messages` — `messages[].content` (string or text
//!     blocks) + top-level `system`.
//!   * **OpenAI** `/chat/completions` — `messages[].content` (string or text
//!     parts). (System is just a `role:"system"` message, handled by the loop.)
//!   * **Gemini** `…:generateContent` / `…:streamGenerateContent` —
//!     `contents[].parts[].text` + `systemInstruction.parts[].text`.
//!
//! All providers share ONE total-budget 0/1 knapsack over every text block.
//! Streaming is response-side: the request body always compresses; SSE/chunked
//! responses stream straight back. Fail-open everywhere; byte-exact passthrough
//! when nothing is compressed (keeps the provider prefix cache).

use crate::compress::{compress_text, est_tokens, knapsack_select, score_blocks, split_blocks};
use serde_json::Value;

/// Upstream API a request targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Gemini,
}

/// Detect the provider from the request path, or `None` (→ forward untouched).
pub fn detect_provider(url: &str) -> Option<Provider> {
    // Lowercase for matching: Gemini's `streamGenerateContent` has a capital G,
    // so a case-sensitive substring check would miss the streaming variant.
    let u = url.to_ascii_lowercase();
    if u.contains("/v1/messages") {
        Some(Provider::Anthropic)
    } else if u.contains("/chat/completions") {
        Some(Provider::OpenAI)
    } else if u.contains("generatecontent") {
        // covers both generateContent and streamGenerateContent
        Some(Provider::Gemini)
    } else {
        None
    }
}

/// A JSON path segment for locating a text field generically.
#[derive(Clone)]
enum Seg {
    Key(&'static str),
    Idx(usize),
}

fn get_str<'a>(v: &'a Value, path: &[Seg]) -> Option<&'a str> {
    let mut cur = v;
    for seg in path {
        cur = match seg {
            Seg::Key(k) => cur.get(k)?,
            Seg::Idx(i) => cur.get(i)?,
        };
    }
    cur.as_str()
}

fn set_str(v: &mut Value, path: &[Seg], s: String) {
    let mut cur = v;
    for seg in path {
        cur = match seg {
            Seg::Key(k) => match cur.get_mut(k) {
                Some(x) => x,
                None => return,
            },
            Seg::Idx(i) => match cur.get_mut(*i) {
                Some(x) => x,
                None => return,
            },
        };
    }
    *cur = Value::String(s);
}

/// Append paths to every text field inside an OpenAI/Anthropic `content` value
/// (string content, or an array of `{"type":"text","text":...}` parts).
fn collect_content(content: Option<&Value>, base: Vec<Seg>, out: &mut Vec<Vec<Seg>>) {
    match content {
        Some(Value::String(_)) => out.push(base),
        Some(Value::Array(blks)) => {
            for (j, b) in blks.iter().enumerate() {
                if b.get("type").and_then(Value::as_str) == Some("text")
                    && b.get("text").and_then(Value::as_str).is_some()
                {
                    let mut p = base.clone();
                    p.push(Seg::Idx(j));
                    p.push(Seg::Key("text"));
                    out.push(p);
                }
            }
        }
        _ => {}
    }
}

/// Enumerate JSON paths to every compressible text field for the given provider.
fn collect_slots(v: &Value, provider: Provider) -> Vec<Vec<Seg>> {
    let mut out = Vec::new();
    match provider {
        Provider::Anthropic | Provider::OpenAI => {
            if let Some(msgs) = v.get("messages").and_then(Value::as_array) {
                for (i, m) in msgs.iter().enumerate() {
                    collect_content(
                        m.get("content"),
                        vec![Seg::Key("messages"), Seg::Idx(i), Seg::Key("content")],
                        &mut out,
                    );
                }
            }
            if provider == Provider::Anthropic {
                collect_content(v.get("system"), vec![Seg::Key("system")], &mut out);
            }
        }
        Provider::Gemini => {
            if let Some(contents) = v.get("contents").and_then(Value::as_array) {
                for (i, c) in contents.iter().enumerate() {
                    if let Some(parts) = c.get("parts").and_then(Value::as_array) {
                        for (j, p) in parts.iter().enumerate() {
                            if p.get("text").and_then(Value::as_str).is_some() {
                                out.push(vec![
                                    Seg::Key("contents"),
                                    Seg::Idx(i),
                                    Seg::Key("parts"),
                                    Seg::Idx(j),
                                    Seg::Key("text"),
                                ]);
                            }
                        }
                    }
                }
            }
            if let Some(parts) = v
                .get("systemInstruction")
                .and_then(|s| s.get("parts"))
                .and_then(Value::as_array)
            {
                for (j, p) in parts.iter().enumerate() {
                    if p.get("text").and_then(Value::as_str).is_some() {
                        out.push(vec![
                            Seg::Key("systemInstruction"),
                            Seg::Key("parts"),
                            Seg::Idx(j),
                            Seg::Key("text"),
                        ]);
                    }
                }
            }
        }
    }
    out
}

/// Compress the text payloads of a request under a single **total** token
/// budget, preserving JSON structure. Returns `(rewritten_body, before, after)`
/// in total tokens. Non-JSON bodies and within-budget requests are returned
/// verbatim (fail-open / byte-exact).
pub fn compress_request_body(
    body: &str,
    total_budget: usize,
    provider: Provider,
    cache_aligned: bool,
) -> (String, usize, usize) {
    let mut v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return (body.to_string(), 0, 0),
    };

    let paths = collect_slots(&v, provider);
    // Read texts aligned to paths (collect_slots only emits paths that resolve).
    let mut slots: Vec<(Vec<Seg>, String)> = Vec::with_capacity(paths.len());
    for p in paths {
        if let Some(s) = get_str(&v, &p) {
            let s = s.to_string();
            slots.push((p, s));
        }
    }

    let before_total: usize = slots.iter().map(|(_, s)| est_tokens(s)).sum();
    if slots.is_empty() || before_total <= total_budget {
        return (body.to_string(), 0, 0);
    }

    // CACHE-ALIGNED (default for the proxy): compress each text field
    // INDEPENDENTLY — its output depends only on its own content, so unchanged
    // prefix fields (system, earlier turns) produce byte-identical bytes across
    // requests → the provider's prefix cache (Anthropic 90% / OpenAI 50%) keeps
    // hitting. Trades a little global optimality for cache stability, which
    // usually dominates on chatty/proxy workloads. Byte-exact passthrough if
    // nothing actually changed.
    if cache_aligned {
        let mut after_total = 0usize;
        let mut changed = false;
        for (path, text) in &slots {
            let c = compress_text(text, total_budget);
            after_total += est_tokens(&c);
            if c != *text {
                set_str(&mut v, path, c);
                changed = true;
            }
        }
        if !changed {
            return (body.to_string(), 0, 0);
        }
        let out = serde_json::to_string(&v).unwrap_or_else(|_| body.to_string());
        return (out, before_total, after_total);
    }

    // GLOBAL total-budget knapsack (one-shot large contexts; not prefix-stable).

    // Flatten all blocks across all slots into one global pool, then run a
    // single 0/1 knapsack over the total budget.
    let slot_blocks: Vec<Vec<String>> = slots
        .iter()
        .map(|(_, s)| split_blocks(s).iter().map(|b| b.to_string()).collect())
        .collect();
    let mut pool_refs: Vec<(usize, usize)> = Vec::new();
    let mut pool_text: Vec<&str> = Vec::new();
    for (si, blocks) in slot_blocks.iter().enumerate() {
        for (bi, b) in blocks.iter().enumerate() {
            pool_refs.push((si, bi));
            pool_text.push(b.as_str());
        }
    }
    let values = score_blocks(&pool_text);
    let weights: Vec<usize> = pool_text.iter().map(|b| est_tokens(b)).collect();
    let keep = knapsack_select(&values, &weights, total_budget);

    let mut keep_per_slot: Vec<Vec<bool>> =
        slot_blocks.iter().map(|b| vec![false; b.len()]).collect();
    for (p, &(si, bi)) in pool_refs.iter().enumerate() {
        keep_per_slot[si][bi] = keep[p];
    }
    if !keep.iter().any(|&k| k) && !pool_refs.is_empty() {
        let best = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let (si, bi) = pool_refs[best];
        keep_per_slot[si][bi] = true;
    }

    let new_texts: Vec<String> = slot_blocks
        .iter()
        .enumerate()
        .map(|(si, blocks)| {
            blocks
                .iter()
                .enumerate()
                .filter(|(bi, _)| keep_per_slot[si][*bi])
                .map(|(_, b)| b.clone())
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .collect();
    let after_total: usize = new_texts.iter().map(|s| est_tokens(s)).sum();

    for ((path, _), new_text) in slots.iter().zip(new_texts) {
        set_str(&mut v, path, new_text);
    }

    let out = serde_json::to_string(&v).unwrap_or_else(|_| body.to_string());
    (out, before_total, after_total)
}

// ── HTTP server (feature-gated; deps never enter the Python wheel) ──────────

/// Max request body we will buffer (bounds memory against abusive clients).
#[cfg(feature = "proxy")]
const MAX_BODY_BYTES: u64 = 64 * 1024 * 1024;

/// Run the proxy: listen on `127.0.0.1:port`, compress provider message context
/// under `total_budget`, and forward to `upstream`. Concurrent worker pool.
#[cfg(feature = "proxy")]
pub fn run(
    port: u16,
    upstream: &str,
    total_budget: usize,
    cache_aligned: bool,
) -> std::io::Result<()> {
    use std::sync::Arc;
    use tiny_http::Server;

    let addr = format!("127.0.0.1:{port}");
    let server = Arc::new(
        Server::http(&addr).map_err(|e| std::io::Error::other(format!("bind {addr}: {e}")))?,
    );
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 16);
    eprintln!("entroly-rs proxy on http://{addr}  ->  {upstream}  ({workers} workers)");
    eprintln!(
        "  Anthropic/OpenAI/Gemini auto-detected by path; point your client's base URL here."
    );

    let upstream: Arc<str> = Arc::from(upstream);
    // Shared agent (connection pooling) with a connect timeout so an
    // unreachable upstream can never hang a worker forever. No read timeout —
    // streaming (SSE) responses are long-lived by design.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(10))
        .build();
    let mut handles = Vec::new();
    for _ in 0..workers {
        let server = Arc::clone(&server);
        let upstream = Arc::clone(&upstream);
        let agent = agent.clone();
        handles.push(std::thread::spawn(move || {
            for req in server.incoming_requests() {
                // Per-request panic isolation: a malformed request can never
                // kill the worker (which would permanently shrink the pool).
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    handle(&agent, req, &upstream, total_budget, cache_aligned)
                }))
                .is_err()
                {
                    eprintln!("entroly-rs: recovered from a request panic");
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

#[cfg(feature = "proxy")]
fn handle(
    agent: &ureq::Agent,
    mut req: tiny_http::Request,
    upstream: &str,
    total_budget: usize,
    cache_aligned: bool,
) {
    use std::io::Read;
    use tiny_http::{Header, Response, StatusCode};

    let method = req.method().as_str().to_uppercase();
    let is_get = method == "GET";
    let url = req.url().to_string();
    let fwd_headers: Vec<(String, String)> = req
        .headers()
        .iter()
        .filter(|h| {
            matches!(
                h.field.as_str().as_str().to_ascii_lowercase().as_str(),
                "x-api-key"
                    | "authorization"
                    | "anthropic-version"
                    | "anthropic-beta"
                    | "openai-organization"
                    | "openai-beta"
                    | "x-goog-api-key"
                    | "content-type"
                    | "accept"
            )
        })
        .map(|h| {
            (
                h.field.as_str().as_str().to_string(),
                h.value.as_str().to_string(),
            )
        })
        .collect();

    if is_get && (url == "/" || url == "/health") {
        let _ = req.respond(Response::from_string("entroly-rs proxy ok"));
        return;
    }

    let mut body = String::new();
    let _ = req
        .as_reader()
        .take(MAX_BODY_BYTES)
        .read_to_string(&mut body);

    let (new_body, before, after) = match detect_provider(&url) {
        Some(p) => compress_request_body(&body, total_budget, p, cache_aligned),
        None => (body, 0, 0),
    };
    if before > 0 {
        let pct = before.saturating_sub(after) as f64 / before as f64 * 100.0;
        eprintln!("entroly-rs: request ~{before} -> ~{after} tokens ({pct:.1}% saved)");
    }

    let target = format!("{}{}", upstream.trim_end_matches('/'), url);
    // Preserve the client's HTTP method (GET /v1/models must not become a POST).
    let mut up = agent.request(&method, &target);
    for (k, val) in &fwd_headers {
        up = up.set(k, val);
    }

    // Bodyless methods (GET/HEAD/DELETE) use .call(); methods with a body send it.
    let sent = if new_body.is_empty() {
        up.call()
    } else {
        up.send_string(&new_body)
    };
    let resp = match sent {
        Ok(r) | Err(ureq::Error::Status(_, r)) => r,
        Err(e) => {
            let _ = req.respond(
                Response::from_string(format!("entroly-rs upstream error: {e}"))
                    .with_status_code(502),
            );
            return;
        }
    };

    let status = resp.status();
    let mut out_headers: Vec<Header> = Vec::new();
    for name in resp.headers_names() {
        // ureq already decoded content-encoding; never re-forward length/encoding
        // headers (would mis-frame or double-decode). tiny_http frames the
        // decoded stream itself (chunked).
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "content-length" | "transfer-encoding" | "connection" | "content-encoding"
        ) {
            continue;
        }
        if let Some(val) = resp.header(&name) {
            if let Ok(h) = Header::from_bytes(name.as_bytes(), val.as_bytes()) {
                out_headers.push(h);
            }
        }
    }
    let reader = resp.into_reader();
    let response = Response::new(StatusCode(status), out_headers, reader, None, None);
    let _ = req.respond(response);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(n: usize) -> String {
        (0..n)
            .map(|i| format!("Line {i}: genuine content about module {i} behavior and edge cases."))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_detect_provider() {
        assert_eq!(detect_provider("/v1/messages"), Some(Provider::Anthropic));
        assert_eq!(
            detect_provider("/v1/chat/completions"),
            Some(Provider::OpenAI)
        );
        assert_eq!(
            detect_provider("/v1beta/models/gemini-2.5-pro:generateContent"),
            Some(Provider::Gemini)
        );
        assert_eq!(
            detect_provider("/v1beta/models/gemini-2.5-flash:streamGenerateContent"),
            Some(Provider::Gemini)
        );
        assert_eq!(detect_provider("/v1/embeddings"), None);
    }

    #[test]
    fn test_anthropic_string_and_system() {
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":{}}}],"system":{}}}"#,
            serde_json::to_string(&big(200)).unwrap(),
            serde_json::to_string(&big(200)).unwrap()
        );
        let (out, before, after) = compress_request_body(&body, 60, Provider::Anthropic, true);
        assert!(before > 0 && after < before);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["messages"][0]["content"].is_string());
        assert!(v["system"].is_string());
    }

    #[test]
    fn test_openai_chat_completions() {
        let body = format!(
            r#"{{"model":"gpt-4o","messages":[{{"role":"system","content":{}}},{{"role":"user","content":[{{"type":"text","text":{}}}]}}]}}"#,
            serde_json::to_string(&big(200)).unwrap(),
            serde_json::to_string(&big(200)).unwrap()
        );
        let (out, before, after) = compress_request_body(&body, 60, Provider::OpenAI, true);
        assert!(
            before > 0 && after < before,
            "openai should compress: {before}->{after}"
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["messages"][0]["role"], "system");
        assert_eq!(v["messages"][1]["content"][0]["type"], "text");
    }

    #[test]
    fn test_gemini_contents_and_system_instruction() {
        let body = format!(
            r#"{{"contents":[{{"role":"user","parts":[{{"text":{}}}]}}],"systemInstruction":{{"parts":[{{"text":{}}}]}}}}"#,
            serde_json::to_string(&big(200)).unwrap(),
            serde_json::to_string(&big(200)).unwrap()
        );
        let (out, before, after) = compress_request_body(&body, 60, Provider::Gemini, true);
        assert!(
            before > 0 && after < before,
            "gemini should compress: {before}->{after}"
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["contents"][0]["parts"][0]["text"].is_string());
        assert!(v["systemInstruction"]["parts"][0]["text"].is_string());
    }

    #[test]
    fn test_total_budget_shared_across_providers_fields() {
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":{}}},{{"role":"user","content":{}}}]}}"#,
            serde_json::to_string(&big(300)).unwrap(),
            serde_json::to_string(&big(300)).unwrap()
        );
        // Global mode (cache_aligned=false): one budget shared across fields.
        let (_, before, after) = compress_request_body(&body, 80, Provider::OpenAI, false);
        assert!(
            before > 0 && after <= 130,
            "shared total budget ~80: {after}"
        );
    }

    #[test]
    fn test_cache_aligned_prefix_is_stable_across_turns() {
        // The whole point of cache alignment: appending a new turn must NOT
        // change the compressed bytes of the prefix (system + earlier turn),
        // so the provider's prefix cache keeps hitting.
        let sys = big(300);
        let m1 = big(300);
        let m2 = big(60);
        let req1 = format!(
            r#"{{"system":{},"messages":[{{"role":"user","content":{}}}]}}"#,
            serde_json::to_string(&sys).unwrap(),
            serde_json::to_string(&m1).unwrap()
        );
        let req2 = format!(
            r#"{{"system":{},"messages":[{{"role":"user","content":{}}},{{"role":"assistant","content":{}}}]}}"#,
            serde_json::to_string(&sys).unwrap(),
            serde_json::to_string(&m1).unwrap(),
            serde_json::to_string(&m2).unwrap()
        );
        let (o1, _, _) = compress_request_body(&req1, 60, Provider::Anthropic, true);
        let (o2, _, _) = compress_request_body(&req2, 60, Provider::Anthropic, true);
        let v1: Value = serde_json::from_str(&o1).unwrap();
        let v2: Value = serde_json::from_str(&o2).unwrap();
        assert_eq!(
            v1["system"], v2["system"],
            "system prefix must stay byte-stable"
        );
        assert_eq!(
            v1["messages"][0]["content"], v2["messages"][0]["content"],
            "earlier turn must stay byte-stable"
        );
    }

    #[test]
    fn test_small_content_byte_exact() {
        let body = r#"{"messages":[{"role":"user","content":"hi there"}]}"#;
        let (out, before, _) = compress_request_body(body, 2000, Provider::OpenAI, true);
        assert_eq!(before, 0);
        assert_eq!(out, body);
    }

    #[test]
    fn test_non_json_failopen() {
        let (out, before, _) = compress_request_body("not json", 50, Provider::Anthropic, true);
        assert_eq!(before, 0);
        assert_eq!(out, "not json");
    }
}
