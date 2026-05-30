//! Single-binary HTTP proxy for `entroly-rs` (Phase 2 — Anthropic vertical slice).
//!
//! Architecture: the request-body compression transform
//! ([`compress_request_body`]) is **pure** (`serde_json` + [`crate::compress`])
//! and is always compiled and unit-tested. Only the HTTP server + upstream
//! client ([`run`]) is gated behind the `proxy` Cargo feature, so its deps
//! (`tiny_http`, `ureq`) never enter the Python wheel.
//!
//! Behaviour (Anthropic `/v1/messages`):
//!   * Large text inside `messages[].content` and `system` is compressed under
//!     a per-block token budget (small text passes through untouched).
//!   * `stream: true` requests are forwarded unchanged (we never buffer a
//!     stream to rewrite it).
//!   * Fail-open everywhere: any parse/transport error forwards the original
//!     bytes rather than blocking the request — the same "never block" contract
//!     the Python proxy and RTK hold.

use crate::compress::compress_text;
use serde_json::Value;

#[inline]
fn est_tokens(s: &str) -> usize {
    (s.chars().count() / 4).max(1)
}

/// Compress the large text payloads inside an Anthropic `/v1/messages` request
/// body, preserving JSON structure.
///
/// Returns `(rewritten_body, tokens_before, tokens_after)` measured over only
/// the text that was actually compressed. Streaming requests and non-JSON
/// bodies are returned verbatim (fail-open).
pub fn compress_request_body(body: &str, budget_per_block: usize) -> (String, usize, usize) {
    let mut v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return (body.to_string(), 0, 0),
    };
    if v.get("stream").and_then(Value::as_bool) == Some(true) {
        return (body.to_string(), 0, 0);
    }

    let mut before = 0usize;
    let mut after = 0usize;

    if let Some(msgs) = v.get_mut("messages").and_then(Value::as_array_mut) {
        for msg in msgs.iter_mut() {
            if let Some(content) = msg.get_mut("content") {
                compress_content(content, budget_per_block, &mut before, &mut after);
            }
        }
    }
    if let Some(sys) = v.get_mut("system") {
        compress_content(sys, budget_per_block, &mut before, &mut after);
    }

    // Nothing was compressed → return the body byte-for-byte. This avoids a
    // spurious re-serialization (which would reorder keys / change whitespace
    // and bust the provider's prefix cache for no benefit).
    if before == 0 {
        return (body.to_string(), 0, 0);
    }

    let out = serde_json::to_string(&v).unwrap_or_else(|_| body.to_string());
    (out, before, after)
}

/// Compress a content field that is either a plain `String` or an array of
/// Anthropic content blocks (`{"type":"text","text": "..."}`). Only text whose
/// estimate exceeds `budget` is compressed; smaller text is left untouched.
fn compress_content(content: &mut Value, budget: usize, before: &mut usize, after: &mut usize) {
    match content {
        Value::String(s) => {
            if est_tokens(s) > budget {
                *before += est_tokens(s);
                let c = compress_text(s, budget);
                *after += est_tokens(&c);
                *s = c;
            }
        }
        Value::Array(blocks) => {
            for b in blocks.iter_mut() {
                if b.get("type").and_then(Value::as_str) != Some("text") {
                    continue;
                }
                // Read to an owned String first to avoid an aliasing borrow.
                let cur = b.get("text").and_then(Value::as_str).map(str::to_string);
                if let Some(s) = cur {
                    if est_tokens(&s) > budget {
                        *before += est_tokens(&s);
                        let c = compress_text(&s, budget);
                        *after += est_tokens(&c);
                        if let Some(slot) = b.get_mut("text") {
                            *slot = Value::String(c);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Run the proxy: listen on `127.0.0.1:port`, compress Anthropic message
/// context, and forward to `upstream`. Blocks forever. Fail-open per request.
#[cfg(feature = "proxy")]
pub fn run(port: u16, upstream: &str, budget_per_block: usize) -> std::io::Result<()> {
    use tiny_http::{Header, Method, Response, Server};

    let addr = format!("127.0.0.1:{port}");
    let server = Server::http(&addr)
        .map_err(|e| std::io::Error::other(format!("bind {addr}: {e}")))?;
    eprintln!("entroly-rs proxy on http://{addr}  ->  {upstream}");
    eprintln!("  point your client:  ANTHROPIC_BASE_URL=http://{addr}");

    for mut req in server.incoming_requests() {
        // Snapshot request metadata before consuming the body reader.
        let is_get = *req.method() == Method::Get;
        let url = req.url().to_string();
        let fwd_headers: Vec<(String, String)> = req
            .headers()
            .iter()
            .filter_map(|h| {
                let name = h.field.as_str().as_str().to_string();
                match name.to_ascii_lowercase().as_str() {
                    "x-api-key" | "authorization" | "anthropic-version" | "anthropic-beta"
                    | "content-type" => Some((name, h.value.as_str().to_string())),
                    _ => None,
                }
            })
            .collect();

        if is_get && (url == "/" || url == "/health") {
            let _ = req.respond(Response::from_string("entroly-rs proxy ok"));
            continue;
        }

        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);

        let (new_body, before, after) = if url.contains("/v1/messages") {
            compress_request_body(&body, budget_per_block)
        } else {
            (body, 0, 0)
        };
        if before > 0 {
            let pct = (before.saturating_sub(after)) as f64 / before as f64 * 100.0;
            eprintln!("entroly-rs: request ~{before} -> ~{after} tokens ({pct:.1}% saved)");
        }

        let target = format!("{}{}", upstream.trim_end_matches('/'), url);
        let mut up = ureq::post(&target);
        for (k, val) in &fwd_headers {
            up = up.set(k, val);
        }

        // Resolve the upstream response into plain data first, so `req` is
        // consumed exactly once (the response or the error path, never both).
        let resolved: Result<(u16, String, String), String> = match up.send_string(&new_body) {
            Ok(resp) | Err(ureq::Error::Status(_, resp)) => {
                let status = resp.status();
                let ctype = resp
                    .header("content-type")
                    .unwrap_or("application/json")
                    .to_string();
                Ok((status, ctype, resp.into_string().unwrap_or_default()))
            }
            Err(e) => Err(format!("entroly-rs upstream error: {e}")),
        };

        match resolved {
            Ok((status, ctype, text)) => {
                let hdr = Header::from_bytes(&b"Content-Type"[..], ctype.as_bytes())
                    .unwrap_or_else(|_| {
                        Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
                    });
                let _ = req.respond(
                    Response::from_string(text)
                        .with_status_code(status)
                        .with_header(hdr),
                );
            }
            Err(msg) => {
                let _ = req.respond(Response::from_string(msg).with_status_code(502));
            }
        }
    }
    Ok(())
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
    fn test_compresses_large_string_content() {
        let body = format!(
            r#"{{"model":"claude","messages":[{{"role":"user","content":{}}}]}}"#,
            serde_json::to_string(&big(200)).unwrap()
        );
        let (out, before, after) = compress_request_body(&body, 50);
        assert!(before > 0 && after < before, "should compress: {before}->{after}");
        // Output must still be valid JSON with the structure intact.
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["messages"][0]["role"], "user");
        assert!(v["messages"][0]["content"].is_string());
    }

    #[test]
    fn test_compresses_text_blocks_array() {
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":{}}}]}}]}}"#,
            serde_json::to_string(&big(200)).unwrap()
        );
        let (out, before, after) = compress_request_body(&body, 50);
        assert!(before > 0 && after < before);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["messages"][0]["content"][0]["type"], "text");
    }

    #[test]
    fn test_small_content_untouched() {
        let body = r#"{"messages":[{"role":"user","content":"hello there"}]}"#;
        let (out, before, _) = compress_request_body(body, 2000);
        assert_eq!(before, 0, "small content must not be compressed");
        assert_eq!(out, body);
    }

    #[test]
    fn test_streaming_passthrough() {
        let body = format!(
            r#"{{"stream":true,"messages":[{{"role":"user","content":{}}}]}}"#,
            serde_json::to_string(&big(200)).unwrap()
        );
        let (out, before, _) = compress_request_body(&body, 50);
        assert_eq!(before, 0, "streaming requests must pass through unchanged");
        assert_eq!(out, body);
    }

    #[test]
    fn test_non_json_failopen() {
        let (out, before, _) = compress_request_body("not json at all", 50);
        assert_eq!(before, 0);
        assert_eq!(out, "not json at all");
    }
}
