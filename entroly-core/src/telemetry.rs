//! JSONL retention pruning — keep rows whose timestamp >= cutoff.
//!
//! Used by the privacy/telemetry retention enforcement layer to age out
//! old activity rows from append-only JSONL files (`activity.jsonl`,
//! audit logs, etc.).
//!
//! Design discipline:
//!   - `prune_jsonl_lines` is pure — takes lines in, returns kept lines
//!     + stats. No IO. Trivially unit-testable.
//!   - `prune_jsonl_by_ts` is a thin IO wrapper: read → call pure fn →
//!     write back via temp + rename for atomicity.
//!
//! Robustness invariants:
//!   - Fail-open: lines that don't parse as JSON are kept (no data loss
//!     from a one-line schema bug).
//!   - Atomic replace: writes go to `<path>.tmp` and rename onto
//!     `<path>` so a crash mid-write doesn't corrupt the source.
//!   - Skip-when-clean: if nothing was removed, the source file is not
//!     touched at all (preserves mtime).
//!
//! Module name kept as `telemetry` for cross-repo import-path
//! compatibility, even though the function is dataset-agnostic.
use std::fs;
use std::io::{self, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    pub kept: usize,
    pub removed: usize,
    pub changed: bool,
}

/// Pure prune logic. Takes an iterator of lines (any string source);
/// returns the kept lines + stats. No IO, no panics, no side effects.
///
/// This is the part you unit-test exhaustively. The IO wrapper below is
/// just plumbing.
pub fn prune_jsonl_lines<'a, I>(lines: I, ts_key: &str, cutoff_ts: f64) -> (Vec<String>, PruneStats)
where
    I: IntoIterator<Item = &'a str>,
{
    let mut kept_lines: Vec<String> = Vec::new();
    let mut kept: usize = 0;
    let mut removed: usize = 0;

    for raw in lines {
        let s = raw.trim();
        if s.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(s) {
            Ok(v) => {
                let ts_ok = v.get(ts_key).and_then(|x| x.as_f64()).unwrap_or(0.0) >= cutoff_ts;
                if ts_ok {
                    kept_lines.push(serde_json::to_string(&v).unwrap_or_else(|_| s.to_string()));
                    kept += 1;
                } else {
                    removed += 1;
                }
            }
            Err(_) => {
                // Fail-open: keep non-JSON lines verbatim.
                kept_lines.push(s.to_string());
                kept += 1;
            }
        }
    }

    let stats = PruneStats {
        kept,
        removed,
        changed: removed > 0,
    };
    (kept_lines, stats)
}

/// Prune a JSONL file in-place, keeping rows where `row[ts_key] >= cutoff_ts`.
///
/// Atomic: writes to a sibling `.tmp` file then renames onto the
/// original. A crash mid-write leaves the original intact (the `.tmp`
/// is orphaned, easy to spot).
pub fn prune_jsonl_by_ts<P: AsRef<Path>>(
    path: P,
    ts_key: &str,
    cutoff_ts: f64,
) -> io::Result<PruneStats> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(PruneStats {
            kept: 0,
            removed: 0,
            changed: false,
        });
    }

    let raw = fs::read_to_string(path)?;
    let (kept_lines, stats) = prune_jsonl_lines(raw.lines(), ts_key, cutoff_ts);

    if !stats.changed {
        return Ok(stats);
    }

    let tmp = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| format!("{e}."))
            .unwrap_or_default()
    ));
    {
        let mut f = fs::File::create(&tmp)?;
        if !kept_lines.is_empty() {
            f.write_all(kept_lines.join("\n").as_bytes())?;
            f.write_all(b"\n")?;
        }
        f.flush()?;
    }
    fs::rename(&tmp, path)?;

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    // ─── Pure-function tests (no IO, no tempdir) ───────────────────

    fn now_ts() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    #[test]
    fn pure_removes_only_old() {
        let now = now_ts();
        let old_line = format!(r#"{{"ts":{},"v":"old"}}"#, now - 100.0);
        let new_line = format!(r#"{{"ts":{},"v":"new"}}"#, now + 100.0);
        let lines = [old_line.as_str(), new_line.as_str()];

        let (kept, stats) = prune_jsonl_lines(lines, "ts", now);
        assert_eq!(stats.removed, 1);
        assert_eq!(stats.kept, 1);
        assert!(stats.changed);
        assert_eq!(kept.len(), 1);
        assert!(kept[0].contains("\"new\""));
    }

    #[test]
    fn pure_keeps_invalid_json_lines() {
        // Fail-open contract: a non-JSON line must NOT be dropped.
        let now = now_ts();
        let lines = ["not-json", &format!(r#"{{"ts":{},"v":"keep"}}"#, now + 1.0)];
        let (kept, stats) = prune_jsonl_lines(lines, "ts", now);
        assert_eq!(stats.removed, 0);
        assert_eq!(stats.kept, 2);
        assert!(kept.iter().any(|l| l == "not-json"));
    }

    #[test]
    fn pure_skips_blank_lines() {
        let (kept, stats) = prune_jsonl_lines(["", "   ", ""], "ts", 0.0);
        assert_eq!(stats.kept, 0);
        assert_eq!(stats.removed, 0);
        assert!(kept.is_empty());
    }

    #[test]
    fn pure_missing_ts_treated_as_zero() {
        // Row without the ts_key gets ts=0, so it's removed when cutoff > 0.
        let lines = [r#"{"other":"x"}"#];
        let (_, stats) = prune_jsonl_lines(lines, "ts", 1.0);
        assert_eq!(stats.removed, 1);
        assert_eq!(stats.kept, 0);
    }

    #[test]
    fn pure_stats_changed_only_when_remove() {
        let lines = [r#"{"ts":100.0}"#];
        let (_, stats) = prune_jsonl_lines(lines, "ts", 50.0);
        assert!(!stats.changed); // nothing removed, so file would NOT be touched
    }

    // ─── IO wrapper test (one happy-path integration) ──────────────

    #[test]
    fn io_prunes_old_rows_keeps_new_and_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("activity.jsonl");
        let now = now_ts();
        let old = serde_json::json!({"ts": now - 10.0, "summary": "old"});
        let new = serde_json::json!({"ts": now + 10.0, "summary": "new"});

        // Three separate writes, no embedded test-data inside format strings.
        let mut content = String::new();
        content.push_str(&old.to_string());
        content.push('\n');
        content.push_str(&new.to_string());
        content.push('\n');
        content.push_str("not-json\n");
        fs::write(&path, content).unwrap();

        let out = prune_jsonl_by_ts(&path, "ts", now).unwrap();
        assert_eq!(out.removed, 1);
        assert!(out.changed);

        let after = fs::read_to_string(&path).unwrap();
        assert!(after.contains("new"));
        assert!(after.contains("not-json"));
        assert!(!after.contains("old"));
    }

    #[test]
    fn io_missing_file_returns_empty_stats() {
        let stats = prune_jsonl_by_ts("/tmp/does-not-exist.jsonl", "ts", 0.0).unwrap();
        assert_eq!(
            stats,
            PruneStats {
                kept: 0,
                removed: 0,
                changed: false
            }
        );
    }
}
