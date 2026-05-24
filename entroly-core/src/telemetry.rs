use std::fs;
use std::io::{self, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct PruneStats {
    pub kept: usize,
    pub removed: usize,
    pub changed: bool,
}

/// Prune a JSONL file in-place, keeping rows where `row[ts_key] >= cutoff_ts`.
///
/// Safety/robustness:
/// - Fail-open: unparseable JSON lines are kept (no data loss).
/// - Only rewrites when at least one row is removed.
pub fn prune_jsonl_by_ts<P: AsRef<Path>>(
    path: P,
    ts_key: &str,
    cutoff_ts: f64,
) -> io::Result<PruneStats> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(PruneStats { kept: 0, removed: 0, changed: false });
    }

    let raw = fs::read_to_string(path)?;
    let mut kept_lines: Vec<String> = Vec::new();
    let mut kept: usize = 0;
    let mut removed: usize = 0;

    for line in raw.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(s) {
            Ok(v) => {
                let ts_ok = v
                    .get(ts_key)
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0)
                    >= cutoff_ts;
                if ts_ok {
                    kept_lines.push(serde_json::to_string(&v).unwrap_or_else(|_| s.to_string()));
                    kept += 1;
                } else {
                    removed += 1;
                }
            }
            Err(_) => {
                kept_lines.push(s.to_string());
                kept += 1;
            }
        }
    }

    if removed == 0 {
        return Ok(PruneStats { kept, removed: 0, changed: false });
    }

    let tmp = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| format!("{}.", e))
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

    Ok(PruneStats { kept, removed, changed: true })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn prunes_old_rows_keeps_new_and_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("activity.jsonl");
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        let old = serde_json::json!({"ts": now - 10.0, "summary": "old"});
        let new = serde_json::json!({"ts": now + 10.0, "summary": "new"});
        fs::write(
            &path,
            format!("{}\n{}\nnot-json\n", old.to_string(), new.to_string()),
        )
        .unwrap();

        let out = prune_jsonl_by_ts(&path, "ts", now).unwrap();
        assert_eq!(out.removed, 1);
        let after = fs::read_to_string(&path).unwrap();
        assert!(after.contains("new"));
        assert!(after.contains("not-json"));
        assert!(!after.contains("old"));
    }
}
