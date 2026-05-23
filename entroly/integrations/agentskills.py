"""
agentskills.io Export Adapter
=============================

Exports Entroly's vault-promoted skills to the agentskills.io portable
spec so they can be consumed by any compatible agent runtime.

Spec (simplified, v0.1):
  skills/<skill_id>/
    skill.json      # name, description, entity, trigger, metrics
    procedure.md    # human-readable SOP
    tool.py         # executable implementation
    tests.json      # validation cases

Usage:
    from entroly.integrations.agentskills import export_promoted
    export_promoted(vault_path=".entroly/vault", out_dir="./dist/agentskills")

CLI:
    python -m entroly.integrations.agentskills ./dist/agentskills
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any


SPEC_VERSION = "0.1"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    header = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body


def _load_skill(skill_dir: Path) -> dict[str, Any] | None:
    skill_md = skill_dir / "SKILL.md"
    tool_py = skill_dir / "tool.py"
    metrics_json = skill_dir / "metrics.json"
    if not (skill_md.exists() and tool_py.exists()):
        return None

    meta, procedure = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    if meta.get("status") != "promoted":
        return None

    metrics: dict[str, Any] = {}
    if metrics_json.exists():
        try:
            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}

    tests_dir = skill_dir / "tests"
    tests: list[dict[str, Any]] = []
    if tests_dir.is_dir():
        for t in sorted(tests_dir.glob("*.json")):
            try:
                tests.append(json.loads(t.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue

    return {
        "meta": meta,
        "procedure": procedure,
        "tool_code": tool_py.read_text(encoding="utf-8"),
        "metrics": metrics,
        "tests": tests,
    }


def export_promoted(
    vault_path: str | Path = ".entroly/vault",
    out_dir: str | Path = "./dist/agentskills",
) -> dict[str, Any]:
    """Export all promoted vault skills to an agentskills.io-compatible bundle."""
    vault = Path(vault_path)
    skills_dir = vault / "evolution" / "skills"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    skipped: list[str] = []

    for sdir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        loaded = _load_skill(sdir)
        if loaded is None:
            skipped.append(sdir.name)
            continue

        meta = loaded["meta"]
        target = out / sdir.name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

        skill_json = {
            "spec_version": SPEC_VERSION,
            "id": meta.get("skill_id", sdir.name),
            "name": meta.get("name", sdir.name),
            "entity": meta.get("entity", ""),
            "description": f"Skill for handling {meta.get('entity', sdir.name)} queries",
            "status": meta.get("status", "promoted"),
            "created_at": meta.get("created_at", ""),
            "metrics": {
                "fitness_score": loaded["metrics"].get("fitness_score", 0.0),
                "runs": loaded["metrics"].get("runs", 0),
                "successes": loaded["metrics"].get("successes", 0),
                "failures": loaded["metrics"].get("failures", 0),
            },
            "entrypoint": {
                "runtime": "python",
                "module": "tool",
                "match": "matches",
                "execute": "execute",
            },
            "origin": {
                "runtime": "entroly",
                "synthesis": "structural",
                "token_cost": 0.0,
            },
        }
        (target / "skill.json").write_text(
            json.dumps(skill_json, indent=2), encoding="utf-8"
        )
        (target / "procedure.md").write_text(loaded["procedure"], encoding="utf-8")
        (target / "tool.py").write_text(loaded["tool_code"], encoding="utf-8")
        (target / "tests.json").write_text(
            json.dumps(loaded["tests"], indent=2), encoding="utf-8"
        )

        exported.append(meta.get("skill_id", sdir.name))

    manifest = {
        "spec_version": SPEC_VERSION,
        "exported_at": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "source": "entroly",
        "skills": exported,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return {
        "status": "ok",
        "out_dir": str(out),
        "exported": exported,
        "skipped": skipped,
    }


def _cli() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "./dist/agentskills"
    result = export_promoted(out_dir=out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
