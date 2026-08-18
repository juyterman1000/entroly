from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


py_path = Path("entroly/work_graph_repo.py")
py = py_path.read_text(encoding="utf-8")
py = replace_once(
    py,
    "_MAX_COMMITS = 20\n",
    "_MAX_COMMITS = 20\n_MAX_COMMIT_PATHS = 512\n",
    "python commit path bound",
)
helper_anchor = "\ndef _branch_commits(\n"
helper = '''\ndef _commit_changed_paths(root: Path, sha: str, parent_shas: list[str]) -> list[str]:
    """Return a complete bounded first-parent path delta for one local commit."""

    if parent_shas:
        raw = _run_git(
            root,
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--name-only",
            "-z",
            parent_shas[0],
            sha,
        )
    else:
        raw = _run_git(
            root,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--no-renames",
            "--name-only",
            "-r",
            "-z",
            sha,
        )
    paths = sorted({item.replace("\\\\", "/") for item in raw.split("\\0") if item})
    return paths

'''
if helper_anchor not in py:
    raise SystemExit("python commit helper anchor changed")
py = py.replace(helper_anchor, helper + helper_anchor, 1)
old = '''    commits: list[dict[str, Any]] = []
    for offset in range(0, len(fields) - 3, 4):
        sha, timestamp, parents, subject = fields[offset : offset + 4]
        sha = sha.strip()
        if not sha:
            continue
        try:
            timestamp_ms = int(timestamp.strip()) * 1000
        except ValueError:
            timestamp_ms = 0
        commits.append(
            {
                "sha": sha,
                "subject": subject.strip(),
                "timestamp_ms": timestamp_ms,
                "parent_shas": [item for item in parents.split() if item],
                # Deliberately omitted in v1: expanding every historical commit
                # into files can exceed the bounded Rust event size. Current
                # worktree paths are observed exactly above; commit/file impact
                # can be supplied later by repository intelligence as a separate
                # evidence event.
                "changed_paths": [],
            }
        )
'''
new = '''    commits: list[dict[str, Any]] = []
    total_changed_paths = 0
    for offset in range(0, len(fields) - 3, 4):
        sha, timestamp, parents, subject = fields[offset : offset + 4]
        sha = sha.strip()
        if not sha:
            continue
        try:
            timestamp_ms = int(timestamp.strip()) * 1000
        except ValueError:
            timestamp_ms = 0
        parent_shas = [item for item in parents.split() if item]
        changed_paths = _commit_changed_paths(root, sha, parent_shas)
        total_changed_paths += len(changed_paths)
        if total_changed_paths > _MAX_COMMIT_PATHS:
            raise RepositoryDiscoveryError(
                f"observed branch commits touch more than {_MAX_COMMIT_PATHS} path occurrences; "
                "refusing a partial Work Graph commit-path observation"
            )
        commits.append(
            {
                "sha": sha,
                "subject": subject.strip(),
                "timestamp_ms": timestamp_ms,
                "parent_shas": parent_shas,
                "changed_paths": changed_paths,
            }
        )
'''
py = replace_once(py, old, new, "python branch commits")
py_path.write_text(py, encoding="utf-8")

js_path = Path("entroly-wasm/js/work_graph_repo.js")
js = js_path.read_text(encoding="utf-8")
js = replace_once(
    js,
    "const MAX_COMMITS = 20;\n",
    "const MAX_COMMITS = 20;\nconst MAX_COMMIT_PATHS = 512;\n",
    "node commit path bound",
)
helper_anchor = "\nfunction branchCommits(root, base, ahead, maxCommits) {"
helper = '''\nfunction commitChangedPaths(root, sha, parentShas) {
  const args = parentShas.length
    ? ['diff', '--no-ext-diff', '--no-renames', '--name-only', '-z', parentShas[0], sha]
    : ['diff-tree', '--root', '--no-commit-id', '--no-renames', '--name-only', '-r', '-z', sha];
  const raw = runGit(root, args);
  return [...new Set(raw.split('\\0').filter(Boolean).map(value => value.replace(/\\\\/g, '/')))].sort();
}
'''
if helper_anchor not in js:
    raise SystemExit("node commit helper anchor changed")
js = js.replace(helper_anchor, helper + helper_anchor, 1)
old = '''  const commits = [];
  for (let offset = 0; offset + 3 < fields.length; offset += 4) {
    const sha = fields[offset].trim();
    if (!sha) continue;
    const seconds = Number.parseInt(fields[offset + 1].trim(), 10);
    commits.push({
      sha,
      subject: fields[offset + 3].trim(),
      timestamp_ms: Number.isFinite(seconds) ? seconds * 1000 : 0,
      parent_shas: fields[offset + 2].trim().split(/\\s+/).filter(Boolean),
      changed_paths: [],
    });
  }
'''
new = '''  const commits = [];
  let totalChangedPaths = 0;
  for (let offset = 0; offset + 3 < fields.length; offset += 4) {
    const sha = fields[offset].trim();
    if (!sha) continue;
    const seconds = Number.parseInt(fields[offset + 1].trim(), 10);
    const parentShas = fields[offset + 2].trim().split(/\\s+/).filter(Boolean);
    const changedPaths = commitChangedPaths(root, sha, parentShas);
    totalChangedPaths += changedPaths.length;
    if (totalChangedPaths > MAX_COMMIT_PATHS) {
      throw new RepositoryDiscoveryError(
        `observed branch commits touch more than ${MAX_COMMIT_PATHS} path occurrences; ` +
        'refusing a partial Work Graph commit-path observation',
      );
    }
    commits.push({
      sha,
      subject: fields[offset + 3].trim(),
      timestamp_ms: Number.isFinite(seconds) ? seconds * 1000 : 0,
      parent_shas: parentShas,
      changed_paths: changedPaths,
    });
  }
'''
js = replace_once(js, old, new, "node branch commits")
js_path.write_text(js, encoding="utf-8")

print("commit-path observation patch applied")
