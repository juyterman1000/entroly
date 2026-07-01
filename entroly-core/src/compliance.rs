//! Kernel Compliance Gate — middleware wrapper over ebbiforge ComplianceEngine patterns
//!
//! Provides:
//!   1. PII detection in IPC messages before delivery (mirrors ComplianceEngine.check_action)
//!   2. Rate limiting per sender (mirrors RateLimiter: token bucket)
//!   3. Prompt injection detection (mirrors InputSanitizer)
//!   4. Audit log of all kernel-level decisions
//!
//! Design principle: compliance checks are INLINED into the kernel hot path,
//! not an external call — same as ebbiforge's Arc<ComplianceEngine> approach.
//! A PII regex hit → message blocked + audit entry + optional escalation flag.
//!
//! This module deliberately has NO ebbiforge_core import — the kernel is
//! standalone. We replicate the core regex logic from compliance/pii.rs
//! to stay dependency-free.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// ── Minimal PII patterns (mirrors compliance/pii.rs regex set) ────────────────

/// Detect if text contains obvious PII. Returns list of PII type labels found.
/// Simplified version for kernel hot path (no regex overhead — byte scanning).
pub fn detect_pii_fast(text: &str) -> Vec<&'static str> {
    let mut found = Vec::new();

    // Email: contains @
    if text.contains('@') && text.contains('.') {
        found.push("Email");
    }
    // API key pattern: "sk-" prefix with long alphanumeric string
    if text.contains("sk-") {
        let after = &text[text.find("sk-").unwrap()..];
        let key_len = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '-')
            .count();
        if key_len > 20 {
            found.push("APIKey");
        }
    }
    // SSN pattern: NNN-NN-NNNN
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(11) {
        if bytes[i].is_ascii_digit()
            && bytes[i + 3] == b'-'
            && bytes[i + 4].is_ascii_digit()
            && bytes[i + 6] == b'-'
            && bytes[i + 7].is_ascii_digit()
        {
            found.push("SSN");
            break;
        }
    }

    found
}

/// Detect prompt injection attempts (mirrors InputSanitizer).
pub fn detect_injection(text: &str) -> bool {
    let lower = text.to_lowercase();
    // Common injection patterns
    let patterns = [
        "ignore previous instructions",
        "ignore all previous",
        "system prompt:",
        "you are now",
        "jailbreak",
        "disregard the above",
        "forget your instructions",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

/// A single audit entry for kernel compliance decisions.
#[derive(Clone, Debug)]
pub struct AuditEntry {
    pub timestamp_ns: u64,
    pub sender_id: u64,
    pub receiver_id: u64,
    pub decision: &'static str, // "allowed", "pii_blocked", "injection_blocked", "rate_limited"
    pub pii_types: Vec<&'static str>,
}

/// Token bucket rate limiter (mirrors compliance/ratelimit.rs).
struct TokenBucket {
    tokens: f32,
    capacity: f32,
    refill_rate: f32, // tokens per ns
    last_refill_ns: u64,
}

impl TokenBucket {
    fn new(capacity: f32, refill_per_second: f32) -> Self {
        TokenBucket {
            tokens: capacity,
            capacity,
            refill_rate: refill_per_second / 1_000_000_000.0,
            last_refill_ns: now_ns(),
        }
    }

    fn try_consume(&mut self, cost: f32) -> bool {
        let now = now_ns();
        let elapsed = (now - self.last_refill_ns) as f32;
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill_ns = now;

        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Kernel Compliance Gate — wraps all inter-agent message decisions.
///
/// One ComplianceGate per IpcBus instance.
#[pyclass]
pub struct ComplianceGate {
    /// Enable PII blocking
    pii_blocking: bool,
    /// Enable injection detection
    injection_blocking: bool,
    /// Per-sender token bucket (messages/second)
    rate_buckets: HashMap<u64, TokenBucket>,
    /// Messages per second allowed per sender
    rate_limit_mps: f32,
    /// Audit log (last N entries)
    audit_log: Vec<AuditEntry>,
    audit_capacity: usize,
    // Stats
    total_blocked_pii: u64,
    total_blocked_injection: u64,
    total_blocked_rate: u64,
    total_allowed: u64,
}

#[pymethods]
impl ComplianceGate {
    #[new]
    #[pyo3(signature = (pii_blocking = true, injection_blocking = true, rate_limit_mps = 100.0, audit_capacity = 10000))]
    pub fn new(
        pii_blocking: bool,
        injection_blocking: bool,
        rate_limit_mps: f32,
        audit_capacity: usize,
    ) -> Self {
        ComplianceGate {
            pii_blocking,
            injection_blocking,
            rate_buckets: HashMap::new(),
            rate_limit_mps: rate_limit_mps.max(1.0),
            audit_log: Vec::new(),
            audit_capacity,
            total_blocked_pii: 0,
            total_blocked_injection: 0,
            total_blocked_rate: 0,
            total_allowed: 0,
        }
    }

    /// Check whether a message is allowed to pass through.
    ///
    /// Returns a dict:
    ///   {"allowed": bool, "reason": str, "pii_types": list}
    pub fn check_message(&mut self, sender_id: u64, receiver_id: u64, content: &str) -> PyObject {
        Python::with_gil(|py| {
            let result = PyDict::new(py);

            // 1. Prompt injection check
            if self.injection_blocking && detect_injection(content) {
                self.total_blocked_injection += 1;
                self.append_audit(AuditEntry {
                    timestamp_ns: now_ns(),
                    sender_id,
                    receiver_id,
                    decision: "injection_blocked",
                    pii_types: vec![],
                });
                result.set_item("allowed", false).unwrap();
                result.set_item("reason", "injection_detected").unwrap();
                result.set_item("pii_types", Vec::<String>::new()).unwrap();
                return result.into();
            }

            // 2. PII check
            if self.pii_blocking {
                let pii = detect_pii_fast(content);
                if !pii.is_empty() {
                    self.total_blocked_pii += 1;
                    let pii_strs: Vec<String> = pii.iter().map(|s| s.to_string()).collect();
                    self.append_audit(AuditEntry {
                        timestamp_ns: now_ns(),
                        sender_id,
                        receiver_id,
                        decision: "pii_blocked",
                        pii_types: pii.clone(),
                    });
                    result.set_item("allowed", false).unwrap();
                    result.set_item("reason", "pii_detected").unwrap();
                    result.set_item("pii_types", pii_strs).unwrap();
                    return result.into();
                }
            }

            // 3. Rate limit check (token bucket)
            let bucket = self
                .rate_buckets
                .entry(sender_id)
                .or_insert_with(|| TokenBucket::new(10.0, self.rate_limit_mps));
            if !bucket.try_consume(1.0) {
                self.total_blocked_rate += 1;
                self.append_audit(AuditEntry {
                    timestamp_ns: now_ns(),
                    sender_id,
                    receiver_id,
                    decision: "rate_limited",
                    pii_types: vec![],
                });
                result.set_item("allowed", false).unwrap();
                result.set_item("reason", "rate_limited").unwrap();
                result.set_item("pii_types", Vec::<String>::new()).unwrap();
                return result.into();
            }

            // 4. Allow
            self.total_allowed += 1;
            self.append_audit(AuditEntry {
                timestamp_ns: now_ns(),
                sender_id,
                receiver_id,
                decision: "allowed",
                pii_types: vec![],
            });
            result.set_item("allowed", true).unwrap();
            result.set_item("reason", "ok").unwrap();
            result.set_item("pii_types", Vec::<String>::new()).unwrap();
            result.into()
        })
    }

    /// Compliance statistics.
    pub fn stats(&self) -> PyObject {
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("total_allowed", self.total_allowed).unwrap();
            d.set_item("blocked_pii", self.total_blocked_pii).unwrap();
            d.set_item("blocked_injection", self.total_blocked_injection)
                .unwrap();
            d.set_item("blocked_rate", self.total_blocked_rate).unwrap();
            let total = self.total_allowed
                + self.total_blocked_pii
                + self.total_blocked_injection
                + self.total_blocked_rate;
            let denial_rate = if total > 0 {
                (total - self.total_allowed) as f32 / total as f32
            } else {
                0.0
            };
            d.set_item("denial_rate", denial_rate).unwrap();
            d.set_item("audit_entries", self.audit_log.len()).unwrap();
            d.into()
        })
    }

    /// Export audit log as JSON string.
    pub fn export_audit(&self) -> String {
        let entries: Vec<serde_json::Value> = self
            .audit_log
            .iter()
            .map(|e| {
                serde_json::json!({
                    "ts": e.timestamp_ns,
                    "sender": e.sender_id,
                    "receiver": e.receiver_id,
                    "decision": e.decision,
                    "pii": e.pii_types,
                })
            })
            .collect();
        serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string())
    }

    /// GDPR: delete all audit entries for a sender.
    pub fn delete_sender_audit(&mut self, sender_id: u64) {
        self.audit_log.retain(|e| e.sender_id != sender_id);
        self.rate_buckets.remove(&sender_id);
    }
}

impl ComplianceGate {
    fn append_audit(&mut self, entry: AuditEntry) {
        if self.audit_log.len() >= self.audit_capacity {
            self.audit_log.remove(0); // ring-style eviction (oldest first)
        }
        self.audit_log.push(entry);
    }
}
