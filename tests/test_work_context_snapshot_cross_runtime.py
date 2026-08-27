from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from entroly.repository_intelligence import RepositoryIntelligenceService
from entroly.repository_intelligence.verified_context import verify_context_commitment
from entroly.work_context_snapshot_store import WorkContextSnapshotStore
from entroly.work_graph_store import WorkGraphStore


NODE = shutil.which("node")


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.mark.skipif(NODE is None, reason="Node.js is required for cross-runtime parity")
def test_python_node_context_snapshot_bytes_roundtrip(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(
        repo,
        "src/parity.py",
        "def unicode_parity(value):\n"
        "    label = 'snowman ☃'\n"
        "    return f'{label}:{value}'\n",
    )
    _write(
        repo,
        "src/caller.py",
        "from src.parity import unicode_parity\n\n"
        "def call_parity(value):\n"
        "    return unicode_parity(value)\n",
    )

    service = RepositoryIntelligenceService(repo)
    index, _digest, _generation = service._snapshot()
    target = next(symbol for symbol in index.symbols.values() if symbol.name == "unicode_parity")
    payload = service.context(
        "unrelated request",
        token_budget=512,
        max_hops=1,
        proposal_scores=[{"symbol_id": target.symbol_id, "score": 1.0}],
        proposal_provider="cross-runtime-parity",
    )
    assert verify_context_commitment(payload)

    repo_id = "repo:context-snapshot-cross-runtime"
    py_root = tmp_path / "python-state"
    node_root = tmp_path / "node-state"
    py_store = WorkContextSnapshotStore(WorkGraphStore(repo_id, root=py_root))
    token = py_store.put_json(payload)
    digest = payload["receipt"]["context_sha256"]
    raw = (py_store.context_dir / f"{digest}.json").read_bytes()

    # These two bytes prove the parity path cannot rely on JS object
    # reserialisation: ensure_ascii emits Unicode escapes, while Python keeps a
    # float that JSON.parse/stringify would collapse from 1.0 to 1.
    assert b"\\u2603" in raw
    assert b'"score":1.0' in raw
    assert raw == raw.decode("ascii").encode("ascii")

    root = Path(__file__).resolve().parents[1]
    node_module = root / "entroly-wasm" / "js" / "work_context_snapshot_store.js"
    graph_module = root / "entroly-wasm" / "js" / "work_graph_store.js"
    script = r"""
const fs = require('fs');
const { WorkGraphStore } = require(process.argv[1]);
const {
  WorkContextSnapshotError,
  WorkContextSnapshotStore,
  verifyCanonicalSnapshotBytes,
} = require(process.argv[2]);

const sourcePath = process.argv[3];
const targetRoot = process.argv[4];
const repoId = process.argv[5];
const expected = process.argv[6];
const raw = fs.readFileSync(sourcePath);
const verified = verifyCanonicalSnapshotBytes(raw, expected);
const store = new WorkContextSnapshotStore(new WorkGraphStore(repoId, { root: targetRoot }));
const token = store.putCanonicalBytes(raw, expected);
const reread = store.getCanonicalBytes(token);
if (!raw.equals(reread)) throw new Error('Node changed canonical snapshot bytes');

const tampered = Buffer.from(raw);
const marker = Buffer.from('snowman \\u2603', 'ascii');
const offset = tampered.indexOf(marker);
if (offset < 0) throw new Error('Unicode parity marker missing');
tampered[offset] = tampered[offset] === 0x73 ? 0x74 : 0x73;
let tamperRejected = false;
try { verifyCanonicalSnapshotBytes(tampered, expected); }
catch (error) { tamperRejected = error instanceof WorkContextSnapshotError; }
if (!tamperRejected) throw new Error('tampered snapshot was accepted');

let rewriteRejected = false;
try { verifyCanonicalSnapshotBytes(Buffer.concat([raw, Buffer.from(' ')]), expected); }
catch (error) { rewriteRejected = error instanceof WorkContextSnapshotError; }
if (!rewriteRejected) throw new Error('noncanonical rewrite was accepted');

process.stdout.write(JSON.stringify({
  token,
  commitment: verified.commitment,
  sha256: require('crypto').createHash('sha256').update(reread).digest('hex'),
}));
"""
    source_path = py_store.context_dir / f"{digest}.json"
    completed = subprocess.run(
        [
            str(NODE),
            "-e",
            script,
            str(graph_module),
            str(node_module),
            str(source_path),
            str(node_root),
            repo_id,
            digest,
        ],
        cwd=root / "entroly-wasm",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"Node snapshot parity failed:\n{completed.stdout}\n{completed.stderr}")
    result = json.loads(completed.stdout)
    assert result["token"] == token
    assert result["commitment"] == digest
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()

    # Python must accept the exact bytes written by Node under the same repo ID
    # and semantic context commitment.
    node_store = WorkContextSnapshotStore(WorkGraphStore(repo_id, root=node_root))
    restored = node_store.get_json(token)
    assert restored["receipt"]["context_sha256"] == digest
    assert verify_context_commitment(restored)
    assert restored == py_store.get_json(token)
