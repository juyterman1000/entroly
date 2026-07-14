"""Deterministic, replayable retrospective context-quality checks.

A Context Check measures whether files changed by an agent or pull request were
represented in the verified Context Commit that preceded the change.  The
result is evidence about context coverage, not proof of task correctness or
model behaviour.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePath
from typing import Any

from .context_commit import create_context_commit, verify_context_commit
from .context_receipts.ingest import read_documents_from_path
from .context_receipts.models import stable_hash

CONTEXT_CHECK_SCHEMA = "entroly.context-check.v1"
_RISK_RANK = {"low": 0, "medium": 1, "high": 2, "unknown": 2}
_SENSITIVE_MARKERS = (
    "/auth", "auth.", "security", "permission", "secret", "crypto",
    ".github/workflows/", "migration", "dockerfile", "pyproject.toml",
    "package.json", "cargo.toml",
)


def _path_text(value: object) -> str:
    return PurePath(str(value).replace("\\", "/")).as_posix().removeprefix("./")


def _relative_path(value: object, root: Path) -> str:
    candidate = Path(str(value))
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"path is outside the checked root: {value}") from exc
    normalized = _path_text(value)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ValueError(f"path is outside the checked root: {value}")
    return normalized


def _normalize_paths(values: Iterable[str | Path], root: Path | None = None) -> list[str]:
    result: set[str] = set()
    for value in values:
        normalized = _relative_path(value, root.resolve()) if root else _path_text(value)
        if normalized:
            result.add(normalized)
    return sorted(result)


def _is_sensitive(path: str) -> bool:
    lowered = "/" + path.casefold().lstrip("/")
    return any(marker in lowered for marker in _SENSITIVE_MARKERS)


def _max_risk(receipt_level: object, flags: list[dict[str, Any]]) -> str:
    level = str(receipt_level).casefold()
    if level not in {"low", "medium", "high"}:
        level = "low"
    for flag in flags:
        candidate = str(flag.get("severity", "low"))
        if _RISK_RANK.get(candidate, 0) > _RISK_RANK[level]:
            level = candidate
    return level


def _commit_summary(commit: Mapping[str, Any]) -> dict[str, Any]:
    receipt = commit.get("receipt", {})
    receipt = receipt if isinstance(receipt, Mapping) else {}
    engine = commit.get("engine", {})
    return {
        "commit_id": str(commit.get("commit_id", "")),
        "schema_version": str(commit.get("schema_version", "")),
        "receipt_id": str(receipt.get("receipt_id", "")),
        "engine": dict(engine) if isinstance(engine, Mapping) else {},
        "selected_context_digest": str(commit.get("selected_context_digest", "")),
        "recovery_bundle_digest": str(commit.get("recovery_bundle_digest", "")),
    }


def assess_context_commit(
    commit: Mapping[str, Any],
    *,
    changed_files: Iterable[str | Path] | None = None,
    corpus_total_files: int | None = None,
    corpus_included_files: int | None = None,
) -> dict[str, Any]:
    """Create a stable Context Check from a verified Context Commit."""
    verification = verify_context_commit(commit)
    if not verification.valid:
        raise ValueError("Context Commit verification failed: " + ", ".join(verification.errors))

    receipt_raw = commit.get("receipt", {})
    receipt = receipt_raw if isinstance(receipt_raw, Mapping) else {}
    fingerprints = receipt.get("source_fingerprints", {})
    indexed = sorted(str(path) for path in fingerprints) if isinstance(fingerprints, Mapping) else []
    selected_context = receipt.get("selected_context", [])
    selected = sorted({
        str(item.get("source_path"))
        for item in selected_context
        if isinstance(item, Mapping) and item.get("source_path")
    }) if isinstance(selected_context, list) else []

    measured = changed_files is not None
    changed = _normalize_paths(changed_files or ())
    selected_set, indexed_set = set(selected), set(indexed)
    covered = [path for path in changed if path in selected_set]
    missing = [path for path in changed if path not in selected_set]
    unsupported = [path for path in changed if path not in indexed_set]
    indexed_changed = [path for path in changed if path in indexed_set]

    recall = round(len(covered) / len(changed), 4) if measured and changed else (1.0 if measured else None)
    indexed_recall = round(len(covered) / len(indexed_changed), 4) if measured and indexed_changed else None
    precision = round(len(covered) / len(selected), 4) if measured and selected else None

    flags: list[dict[str, Any]] = []
    if not measured:
        flags.append({"code": "comparison_evidence_missing", "severity": "unknown", "message": "No changed-file evidence was supplied; coverage was not measured.", "files": []})
    elif missing:
        sensitive = [path for path in missing if _is_sensitive(path)]
        flags.append({"code": "changed_files_not_selected", "severity": "high" if sensitive else "medium", "message": f"{len(missing)} changed file(s) were absent from selected context.", "files": missing})
        if sensitive:
            flags.append({"code": "sensitive_changed_files_not_selected", "severity": "high", "message": "Sensitive, dependency, workflow, or configuration changes were absent from selected context.", "files": sensitive})
    if unsupported:
        flags.append({"code": "changed_files_not_indexed", "severity": "high", "message": f"{len(unsupported)} changed file(s) were outside the indexed corpus.", "files": unsupported})

    total = len(indexed) if corpus_total_files is None else corpus_total_files
    included = len(indexed) if corpus_included_files is None else corpus_included_files
    if total > included:
        flags.append({"code": "corpus_truncated", "severity": "medium", "message": f"The bounded scan indexed {included} of {total} supported files.", "files": []})

    receipt_risk_raw = receipt.get("risk_summary", {})
    receipt_risk = receipt_risk_raw if isinstance(receipt_risk_raw, Mapping) else {}
    level = "unknown" if not measured else _max_risk(receipt_risk.get("review_level", "low"), flags)
    warnings = [str(item) for item in receipt.get("warnings", [])] if isinstance(receipt.get("warnings"), list) else []
    warnings.extend(str(flag["message"]) for flag in flags)

    payload: dict[str, Any] = {
        "schema_version": CONTEXT_CHECK_SCHEMA,
        "task": str(receipt.get("query", "")),
        "context_commit": _commit_summary(commit),
        "comparison": {
            "status": "measured" if measured else "not_measured",
            "changed_files": changed,
            "changed_files_selected": covered,
            "changed_files_missing": missing,
            "changed_files_not_indexed": unsupported,
        },
        "corpus": {"supported_files_found": total, "files_indexed": included, "indexed_files": indexed, "selected_files": selected},
        "metrics": {
            "changed_file_recall": recall,
            "indexed_changed_file_recall": indexed_recall,
            "changed_file_precision": precision,
            "selected_chunks": verification.selected_chunks,
            "omitted_chunks": verification.omitted_chunks,
            "token_budget": int(receipt.get("token_budget", 0) or 0),
        },
        "risk": {"level": level, "receipt_review_level": str(receipt_risk.get("review_level", "unknown")), "flags": flags},
        "warnings": list(dict.fromkeys(warnings)),
        "interpretation": "Changed-file recall is retrospective coverage evidence. It does not prove task correctness, complete task knowledge, or that a model used the selected context.",
    }
    payload["check_id"] = "cck_" + stable_hash(payload)[:24]
    return payload


def _documents(path: str | Path, max_files: int) -> tuple[list[tuple[str, str]], int, Path]:
    source = Path(path).resolve()
    root = source.parent if source.is_file() else source
    documents = [(_relative_path(name, root), text) for name, text in read_documents_from_path(source)]
    documents.sort(key=lambda item: item[0])
    total = len(documents)
    return (documents[:max_files] if max_files > 0 else documents), total, root


def create_context_check_from_path(
    path: str | Path,
    *,
    task: str,
    token_budget: int = 8000,
    changed_files: Iterable[str | Path] | None = None,
    max_files: int = 0,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    prefer_rust: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not task.strip():
        raise ValueError("task must not be empty")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_files < 0:
        raise ValueError("max_files must be zero or positive")
    documents, total, root = _documents(path, max_files)
    if not documents:
        raise ValueError("no supported documents found")
    normalized = None if changed_files is None else _normalize_paths(changed_files, root)
    commit = create_context_commit(documents, query=task.strip(), token_budget=token_budget, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens, prefer_rust=prefer_rust)
    return assess_context_commit(commit, changed_files=normalized, corpus_total_files=total, corpus_included_files=len(documents)), commit


def git_changed_files(path: str | Path, *, base: str, head: str = "HEAD") -> list[str]:
    if not base.strip() or base.startswith("-") or not head.strip() or head.startswith("-"):
        raise ValueError("git refs must be non-empty and must not start with '-'")
    source = Path(path).resolve()
    root = source.parent if source.is_file() else source
    try:
        result = subprocess.run(["git", "-C", str(root), "diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base}...{head}", "--"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"unable to compare git refs: {exc}") from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git diff failed")
    return _normalize_paths(result.stdout.splitlines())


def context_check_markdown(check: Mapping[str, Any]) -> str:
    comparison = check.get("comparison", {}) if isinstance(check.get("comparison"), Mapping) else {}
    metrics = check.get("metrics", {}) if isinstance(check.get("metrics"), Mapping) else {}
    risk = check.get("risk", {}) if isinstance(check.get("risk"), Mapping) else {}
    commit = check.get("context_commit", {}) if isinstance(check.get("context_commit"), Mapping) else {}
    fmt = lambda name: "not measured" if metrics.get(name) is None else f"{float(metrics[name]):.1%}"
    lines = [
        "# Entroly Context Check", "",
        f"- Check: `{check.get('check_id', '')}`", f"- Task: {check.get('task', '')}",
        f"- Context Commit: `{commit.get('commit_id', '')}`", f"- Risk: **{risk.get('level', 'unknown')}**",
        f"- Comparison: {comparison.get('status', 'not_measured')}",
        f"- Changed-file recall: {fmt('changed_file_recall')}",
        f"- Indexed changed-file recall: {fmt('indexed_changed_file_recall')}",
        f"- Changed-file precision: {fmt('changed_file_precision')}", "", "## Changed-file evidence", "",
    ]
    for label, key in (("Selected", "changed_files_selected"), ("Missing", "changed_files_missing"), ("Not indexed", "changed_files_not_indexed")):
        values = comparison.get(key, [])
        rendered = ", ".join(f"`{value}`" for value in values) if values else "none"
        lines.append(f"- {label}: {rendered}")
    lines.extend(["", "## Risk flags", ""])
    flags = risk.get("flags", [])
    if isinstance(flags, list) and flags:
        lines.extend(f"- **{flag.get('severity', 'unknown')}** `{flag.get('code', '')}`: {flag.get('message', '')}" for flag in flags if isinstance(flag, Mapping))
    else:
        lines.append("- No additional changed-file risk flags.")
    lines.extend(["", "## Interpretation", "", str(check.get("interpretation", "")), ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit selected context against changed files")
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--task", required=True)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--git-base")
    parser.add_argument("--git-head", default="HEAD")
    parser.add_argument("--out", default="entroly-context-check.json")
    parser.add_argument("--report", default="entroly-context-check.md")
    parser.add_argument("--commit-out", default="entroly-context-commit.json")
    parser.add_argument("--fail-on-risk", choices=("none", "medium", "high"), default="none")
    args = parser.parse_args(argv)
    changed = git_changed_files(args.path, base=args.git_base, head=args.git_head) if args.git_base else None
    check, commit = create_context_check_from_path(args.path, task=args.task, token_budget=args.budget, changed_files=changed, max_files=args.max_files)
    Path(args.out).write_text(json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.report).write_text(context_check_markdown(check), encoding="utf-8")
    Path(args.commit_out).write_text(json.dumps(commit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"check_id": check["check_id"], "context_commit_id": check["context_commit"]["commit_id"], "risk": check["risk"]["level"]}, sort_keys=True))
    rank = _RISK_RANK.get(str(check["risk"]["level"]), 2)
    threshold = {"none": 99, "medium": 1, "high": 2}[args.fail_on_risk]
    return 1 if rank >= threshold else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONTEXT_CHECK_SCHEMA", "assess_context_commit", "context_check_markdown", "create_context_check_from_path", "git_changed_files", "main"]
