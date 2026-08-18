from __future__ import annotations

from pathlib import Path


path = Path("entroly-engine/src/work_graph.rs")
text = path.read_text(encoding="utf-8")
if "fn passive_repository_snapshot_fingerprint(" in text:
    raise SystemExit("passive snapshot dedupe already present")

old_observe = '''    pub fn observe_repository(
        &mut self,
        observation: RepositoryObservation,
    ) -> Result<String, WorkGraphError> {
        if observation.repo_id != self.repo_id {
            return Err(WorkGraphError::RepoMismatch {
                expected: self.repo_id.clone(),
                actual: observation.repo_id,
            });
        }
        let event = observation_to_event(&self.repo_id, observation)?;
        self.apply_event(event)
    }
'''
new_observe = '''    pub fn observe_repository(
        &mut self,
        observation: RepositoryObservation,
    ) -> Result<String, WorkGraphError> {
        if observation.repo_id != self.repo_id {
            return Err(WorkGraphError::RepoMismatch {
                expected: self.repo_id.clone(),
                actual: observation.repo_id.clone(),
            });
        }
        let passive_fingerprint = passive_repository_snapshot_fingerprint(&observation)?;
        let mut event = observation_to_event(&self.repo_id, observation)?;
        if let Some(fingerprint) = passive_fingerprint {
            let source_ref = format!("repo-snapshot:{fingerprint}");
            if let Some(last) = self.events.last() {
                if last.source_kind == EvidenceKind::RepositoryFact && last.source_ref == source_ref {
                    return Ok(last.event_id.clone());
                }
            }
            event.source_ref = source_ref;
        }
        self.apply_event(event)
    }
'''
if text.count(old_observe) != 1:
    raise SystemExit(f"observe_repository anchor changed: {text.count(old_observe)} matches")
text = text.replace(old_observe, new_observe, 1)

start_marker = '''    // Adapters discover filesystem/Git/provider facts in different orders.
    // Canonicalize semantically unordered observations so identical work state
    // produces identical events and graph commitments across Python/npm/native.
'''
start = text.index(start_marker, text.index("fn observation_to_event("))
block_start = start + len(start_marker)
end_marker = '''    if let Some(task) = obs.task_hint.as_mut() {
        task.remaining_work.sort();
        task.remaining_work.dedup();
    }
'''
block_end = text.index(end_marker, block_start) + len(end_marker)
canonical_body = text[block_start:block_end]
# Make ordering deterministic even for malformed/duplicate same-path observations
# whose only difference is content identity.
old_change_tail = '''            .then_with(|| a.staged.cmp(&b.staged))
            .then_with(|| a.conflicted.cmp(&b.conflicted))
'''
new_change_tail = '''            .then_with(|| a.staged.cmp(&b.staged))
            .then_with(|| a.conflicted.cmp(&b.conflicted))
            .then_with(|| a.content_digest.cmp(&b.content_digest))
'''
if canonical_body.count(old_change_tail) != 1:
    raise SystemExit("change canonicalization anchor changed")
canonical_body = canonical_body.replace(old_change_tail, new_change_tail, 1)

text = text[:start] + '''    // Keep one canonicalization rule for event construction and passive
    // semantic fingerprints. This is the parity boundary shared by every
    // adapter that submits RepositoryObservation.
    canonicalize_repository_observation(&mut obs);
''' + text[block_end:]

helper = '''fn canonicalize_repository_observation(obs: &mut RepositoryObservation) {
''' + canonical_body + '''}

fn valid_passive_content_digest(change: &FileChangeObservation) -> bool {
    if change.staged || change.conflicted {
        return false;
    }
    if change.kind == FileChangeKind::Deleted {
        return change.content_digest == "worktree:deleted";
    }
    let Some(hex) = change.content_digest.strip_prefix("git-blob:") else {
        return false;
    };
    matches!(hex.len(), 40 | 64) && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn passive_repository_snapshot_fingerprint(
    observation: &RepositoryObservation,
) -> Result<Option<String>, WorkGraphError> {
    // Only passive repository/checkpoint observations may collapse. Active
    // operations are audit history and must remain distinct even when their
    // payload happens to repeat.
    if !observation.agent_id.is_empty()
        || !observation.session_id.is_empty()
        || !observation.verifications.is_empty()
        || !observation.claims.is_empty()
        || !observation.leases.is_empty()
        || !observation.model_executions.is_empty()
    {
        return Ok(None);
    }
    if observation
        .task_hint
        .as_ref()
        .is_some_and(|task| task.source_kind != EvidenceKind::Checkpoint)
    {
        return Ok(None);
    }
    if observation
        .decisions
        .iter()
        .any(|decision| decision.source_kind != EvidenceKind::Checkpoint)
    {
        return Ok(None);
    }
    // A timestamp-only equality decision is safe only when every worktree
    // change has exact content identity. Staged/conflicted/special/oversized
    // paths deliberately fail closed in the adapters and therefore remain
    // separate audit events.
    if observation
        .changes
        .iter()
        .any(|change| !valid_passive_content_digest(change))
    {
        return Ok(None);
    }

    let mut semantic = observation.clone();
    semantic.observed_at_ms = 0;
    canonicalize_repository_observation(&mut semantic);
    Ok(Some(sha256_json(&semantic)?))
}

'''
insert_at = text.index("fn observation_to_event(")
text = text[:insert_at] + helper + text[insert_at:]

anchor = '''    #[test]
    fn clean_repo_is_null_control() {
'''
if text.count(anchor) != 1:
    raise SystemExit("Rust test insertion anchor changed")
tests = '''    fn passive_dirty_observation(digest: &str, observed_at_ms: i64) -> RepositoryObservation {
        let mut obs = clean_observation();
        obs.observed_at_ms = observed_at_ms;
        obs.branch.name = "feature/passive".to_string();
        obs.changes.push(FileChangeObservation {
            path: "src/auth.rs".to_string(),
            kind: FileChangeKind::Modified,
            staged: false,
            conflicted: false,
            old_path: String::new(),
            content_digest: digest.to_string(),
        });
        obs
    }

    #[test]
    fn identical_content_complete_passive_snapshots_do_not_grow_history() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let first = graph
            .observe_repository(passive_dirty_observation(
                "git-blob:1111111111111111111111111111111111111111",
                1_000,
            ))
            .unwrap();
        let commitment = graph.graph_commitment().to_string();
        let second = graph
            .observe_repository(passive_dirty_observation(
                "git-blob:1111111111111111111111111111111111111111",
                9_000,
            ))
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(graph.event_count(), 1);
        assert_eq!(graph.graph_commitment(), commitment);
    }

    #[test]
    fn passive_snapshot_byte_change_appends_new_event() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        graph
            .observe_repository(passive_dirty_observation(
                "git-blob:1111111111111111111111111111111111111111",
                1_000,
            ))
            .unwrap();
        graph
            .observe_repository(passive_dirty_observation(
                "git-blob:2222222222222222222222222222222222222222",
                2_000,
            ))
            .unwrap();
        assert_eq!(graph.event_count(), 2);
    }

    #[test]
    fn passive_change_away_and_back_remains_auditable() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        for (time, digest) in [
            (1_000, "git-blob:1111111111111111111111111111111111111111"),
            (2_000, "git-blob:2222222222222222222222222222222222222222"),
            (3_000, "git-blob:1111111111111111111111111111111111111111"),
        ] {
            graph
                .observe_repository(passive_dirty_observation(digest, time))
                .unwrap();
        }
        assert_eq!(graph.event_count(), 3);
    }

    #[test]
    fn passive_snapshot_without_complete_digest_never_dedupes() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        graph
            .observe_repository(passive_dirty_observation("", 1_000))
            .unwrap();
        graph
            .observe_repository(passive_dirty_observation("", 2_000))
            .unwrap();
        assert_eq!(graph.event_count(), 2);
    }

    #[test]
    fn repeated_active_verification_is_never_collapsed() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut first = passive_dirty_observation(
            "git-blob:1111111111111111111111111111111111111111",
            1_000,
        );
        first.verifications.push(VerificationObservation {
            verification_id: "test:repeat".to_string(),
            name: "focused test".to_string(),
            state: VerificationState::Passed,
            evidence_kind: EvidenceKind::TestResult,
            source_ref: "pytest:test_repeat".to_string(),
            digest: "pass".to_string(),
            observed_at_ms: 1_000,
        });
        let mut second = first.clone();
        second.observed_at_ms = 2_000;
        second.verifications[0].observed_at_ms = 2_000;
        graph.observe_repository(first).unwrap();
        graph.observe_repository(second).unwrap();
        assert_eq!(graph.event_count(), 2);
    }

    #[test]
    fn repeated_model_execution_is_never_collapsed() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut first = passive_dirty_observation(
            "git-blob:1111111111111111111111111111111111111111",
            1_000,
        );
        first.model_executions.push(ModelExecutionObservation {
            execution_id: "exec:repeat".to_string(),
            provider: "provider".to_string(),
            model: "model".to_string(),
            success: Some(true),
            latency_ms: 10,
            cost_micro_usd: 1,
            source_ref: "runtime:repeat".to_string(),
        });
        let mut second = first.clone();
        second.observed_at_ms = 2_000;
        graph.observe_repository(first).unwrap();
        graph.observe_repository(second).unwrap();
        assert_eq!(graph.event_count(), 2);
    }

'''
text = text.replace(anchor, tests + anchor, 1)
path.write_text(text, encoding="utf-8")
print("passive snapshot dedupe v2 patch applied")
