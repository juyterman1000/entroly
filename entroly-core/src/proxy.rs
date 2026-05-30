//! Single-binary HTTP proxy for `entroly-rs` (Phase 2 — Anthropic).
//!
//! Architecture: the request-body transform ([`compress_request_body`]) is
//! **pure** (`serde_json` + [`crate::compress`]) and always compiled + tested.
//! Only the server ([`run`]) is gated behind the `proxy` feature, so its deps
//! (`tiny_http`, `ureq`) never enter the Python wheel.
//!
//! Behaviour:
//!   * **Total token budget** across the whole request — all text blocks in
//!     `messages[].content` and `system` are pooled and the optimal subset is
//!     kept via a 0/1 knapsack (not a per-message budget).
//!   * **Streaming-aware**: the request body is compressed regardless of
//!     `stream` (streaming is response-side); SSE responses are streamed back
//!     without buffering.
//!   * **Concurrent**: a worker pool handles requests in parallel.
//!   * **Fail-open**: parse/transport errors forward the original bytes.
//!   * Byte-exact passthrough when nothing is compressed (keeps prefix cache).

use crate::compress::{est_tokens, knapsack_select, score_blocks, split_blocks};
use serde_json::Value;

/// Locates a text slot inside an Anthropic request body for write-back.
enum Loc {
    MsgStr(usize),
    MsgBlk(usize, usize),
    SysStr,
    SysBlk(usize),
}

/// Compress the text payloads of an Anthropic `/v1/messages` request under a
/// single **total** token budget, preserving JSON structure. Returns
/// `(rewritten_body, total_tokens_before, total_tokens_after)`. Non-JSON bodies
/// and already-within-budget requests are returned verbatim (fail-open).
pub fn compress_request_body(body: &str, total_budget: usize) -> (String, usize, usize) {
    let mut v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return (body.to_string(), 0, 0),
    };

    // Pass 1 — gather every text slot (owned, with a write-back locator).
    let mut slots: Vec<(Loc, String)> = Vec::new();
    if let Some(msgs) = v.get("messages").and_then(Value::as_array) {
        for (i, m) in msgs.iter().enumerate() {
            match m.get("content") {
                Some(Value::String(s)) => slots.push((Loc::MsgStr(i), s.clone())),
                Some(Value::Array(blks)) => {
                    for (j, b) in blks.iter().enumerate() {
                        if b.get("type").and_then(Value::as_str) == Some("text") {
                            if let Some(t) = b.get("text").and_then(Value::as_str) {
                                slots.push((Loc::MsgBlk(i, j), t.to_string()));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    match v.get("system") {
        Some(Value::String(s)) => slots.push((Loc::SysStr, s.clone())),
        Some(Value::Array(blks)) => {
            for (j, b) in blks.iter().enumerate() {
                if b.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(t) = b.get("text").and_then(Value::as_str) {
                        slots.push((Loc::SysBlk(j), t.to_string()));
                    }
                }
            }
        }
        _ => {}
    }

    let before_total: usize = slots.iter().map(|(_, s)| est_tokens(s)).sum();
    if slots.is_empty() || before_total <= total_budget {
        return (body.to_string(), 0, 0); // byte-exact passthrough
    }

    // Pass 2 — flatten all blocks across slots into one global pool, then run a
    // single knapsack over the total budget.
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

    // Map keep decisions back per slot.
    let mut keep_per_slot: Vec<Vec<bool>> =
        slot_blocks.iter().map(|b| vec![false; b.len()]).collect();
    for (p, &(si, bi)) in pool_refs.iter().enumerate() {
        keep_per_slot[si][bi] = keep[p];
    }
    // Non-annihilation: if the budget admits nothing, keep the densest block.
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

    // Pass 3 — rebuild each slot's text and write it back.
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

    for ((loc, _), new_text) in slots.iter().zip(new_texts.into_iter()) {
        write_slot(&mut v, loc, new_text);
    }

    let out = serde_json::to_string(&v).unwrap_or_else(|_| body.to_string());
    (out, before_total, after_total)
}

fn write_slot(v: &mut Value, loc: &Loc, text: String) {
    let nv = Value::String(text);
    match loc {
        Loc::MsgStr(i) => {
            if let Some(c) = v
                .get_mut("messages")
                .and_then(|m| m.get_mut(*i))
                .and_then(|m| m.get_mut("content"))
            {
                *c = nv;
            }
        }
        Loc::MsgBlk(i, j) => {
            if let Some(t) = v
                .get_mut("messages")
                .and_then(|m| m.get_mut(*i))
                .and_then(|m| m.get_mut("content"))
                .and_then(|c| c.get_mut(*j))
                .and_then(|b| b.get_mut("text"))
            {
                *t = nv;
            }
        }
        Loc::SysStr => {
            if let Some(s) = v.get_mut("system") {
                *s = nv;
            }
        }
        Loc::SysBlk(j) => {
            if let Some(t) = v
                .get_mut("system")
                .and_then(|s| s.get_mut(*j))
                .and_then(|b| b.get_mut("text"))
            {
                *t = nv;
            }
        }
    }
}

// ── HTTP server (feature-gated; deps never enter the Python wheel) ──────────

/// Max request body we will buffer (bounds memory against abusive clients).
#[cfg(feature = "proxy")]
const MAX_BODY_BYTES: u64 = 64 * 1024 * 1024;

/// Run the proxy: listen on `127.0.0.1:port`, compress Anthropic message
/// context under `total_budget`, and forward to `upstream`. Concurrent worker
/// pool; blocks forever.
#[cfg(feature = "proxy")]
pub fn run(port: u16, upstream: &str, total_budget: usize) -> std::io::Result<()> {
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
    eprintln!("  point your client:  ANTHROPIC_BASE_URL=http://{addr}");

    let upstream: Arc<str> = Arc::from(upstream);
    let mut handles = Vec::new();
    for _ in 0..workers {
        let server = Arc::clone(&server);
        let upstream = Arc::clone(&upstream);
        handles.push(std::thread::spawn(move || {
            for req in server.incoming_requests() {
                handle(req, &upstream, total_budget);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

#[cfg(feature = "proxy")]
fn handle(mut req: tiny_http::Request, upstream: &str, total_budget: usize) {
    use std::io::Read;
    use tiny_http::{Header, Method, Response, StatusCode};

    let is_get = *req.method() == Method::Get;
    let url = req.url().to_string();
    let fwd_headers: Vec<(String, String)> = req
        .headers()
        .iter()
        .filter(|h| {
            matches!(
                h.field.as_str().as_str().to_ascii_lowercase().as_str(),
                "x-api-key" | "authorization" | "anthropic-version" | "anthropic-beta"
                    | "content-type" | "accept"
            )
        })
        .map(|h| (h.field.as_str().as_str().to_string(), h.value.as_str().to_string()))
        .collect();

    if is_get && (url == "/" || url == "/health") {
        let _ = req.respond(Response::from_string("entroly-rs proxy ok"));
        return;
    }

    // Bounded body read (avoids OOM on abusive clients).
    let mut body = String::new();
    let _ = req.as_reader().take(MAX_BODY_BYTES).read_to_string(&mut body);

    let (new_body, before, after) = if url.contains("/v1/messages") {
        compress_request_body(&body, total_budget)
    } else {
        (body, 0, 0)
    };
    if before > 0 {
        let pct = before.saturating_sub(after) as f64 / before as f64 * 100.0;
        eprintln!("entroly-rs: request ~{before} -> ~{after} tokens ({pct:.1}% saved)");
    }

    let target = format!("{}{}", upstream.trim_end_matches('/'), url);
    let mut up = ureq::post(&target);
    for (k, val) in &fwd_headers {
        up = up.set(k, val);
    }

    let resp = match up.send_string(&new_body) {
        Ok(r) | Err(ureq::Error::Status(_, r)) => r,
        Err(e) => {
            let _ = req.respond(
                Response::from_string(format!("entroly-rs upstream error: {e}"))
                    .with_status_code(502),
            );
            return;
        }
    };

    // Forward status + response headers, and STREAM the body (no buffering) so
    // SSE / large responses pass straight through.
    let status = resp.status();
    let mut out_headers: Vec<Header> = Vec::new();
    for name in resp.headers_names() {
        let lname = name.to_ascii_lowercase();
        // Skip hop-by-hop, length, and encoding headers: ureq has ALREADY
        // decoded any content-encoding (gzip), so the reader yields decoded
        // bytes. Forwarding the upstream content-length (compressed size) or
        // content-encoding would frame/decode the body wrong and corrupt it.
        // tiny_http frames the decoded stream itself (chunked).
        if matches!(
            lname.as_str(),
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
    // data_length = None: stream the (decoded) body to EOF as chunked. Never
    // trust the upstream content-length here — it describes the compressed body.
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
    fn test_compresses_large_string_content_total_budget() {
        let body = format!(
            r#"{{"model":"claude","messages":[{{"role":"user","content":{}}}]}}"#,
            serde_json::to_string(&big(300)).unwrap()
        );
        let (out, before, after) = compress_request_body(&body, 60);
        assert!(before > 0 && after < before, "should compress: {before}->{after}");
        assert!(after <= 90, "total budget ~60 (+rounding): {after}");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["messages"][0]["role"], "user");
    }

    #[test]
    fn test_total_budget_across_two_messages() {
        // Two big messages share ONE total budget (not per-message).
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":{}}},{{"role":"user","content":{}}}]}}"#,
            serde_json::to_string(&big(300)).unwrap(),
            serde_json::to_string(&big(300)).unwrap()
        );
        let (_, before, after) = compress_request_body(&body, 80);
        assert!(before > 0 && after <= 130, "shared total budget ~80: {after}");
    }

    #[test]
    fn test_compresses_text_blocks_array() {
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":{}}}]}}]}}"#,
            serde_json::to_string(&big(300)).unwrap()
        );
        let (out, before, after) = compress_request_body(&body, 60);
        assert!(before > 0 && after < before);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["messages"][0]["content"][0]["type"], "text");
    }

    #[test]
    fn test_streaming_request_is_compressed() {
        // The fix: streaming is response-side; the request body still compresses.
        let body = format!(
            r#"{{"stream":true,"messages":[{{"role":"user","content":{}}}]}}"#,
            serde_json::to_string(&big(300)).unwrap()
        );
        let (out, before, after) = compress_request_body(&body, 60);
        assert!(before > 0 && after < before, "streaming request body must compress");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["stream"], true, "stream flag preserved");
    }

    #[test]
    fn test_small_content_untouched_byte_exact() {
        let body = r#"{"messages":[{"role":"user","content":"hello there"}]}"#;
        let (out, before, _) = compress_request_body(body, 2000);
        assert_eq!(before, 0);
        assert_eq!(out, body, "byte-exact passthrough preserves prefix cache");
    }

    #[test]
    fn test_non_json_failopen() {
        let (out, before, _) = compress_request_body("not json at all", 50);
        assert_eq!(before, 0);
        assert_eq!(out, "not json at all");
    }
}
