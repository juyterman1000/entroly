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

from entroly.path_safety import resolve_dir_within, resolve_file_within
from entroly.skill_engine import SkillEngine
from entroly.vault import VaultConfig, VaultManager


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


def _load_skill(skill_dir: Path, engine: SkillEngine) -> dict[str, Any] | None:
    spec = engine.load_promoted_skill(skill_dir.name)
    skill_md = resolve_file_within(skill_dir, "SKILL.md")
    if spec is None or skill_md is None:
        return None

    meta, procedure = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    if meta.get("status") != "promoted":
        return None

    return {
        "meta": meta,
        "procedure": procedure,
        "tool_code": spec.tool_code,
        "metrics": spec.metrics,
        "tests": spec.test_cases,
    }


def export_promoted(
    vault_path: str | Path = ".entroly/vault",
    out_dir: str | Path = "./dist/agentskills",
) -> dict[str, Any]:
    """Export all promoted vault skills to an agentskills.io-compatible bundle."""
    vault = Path(vault_path)
    engine = SkillEngine(VaultManager(VaultConfig(base_path=str(vault))))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    skipped: list[str] = []

    for info in engine.list_skills():
        sdir = Path(info["path"])
        loaded = _load_skill(sdir, engine) if info.get("status") == "promoted" else None
        if loaded is None:
            skipped.append(sdir.name)
            # A previous export must not remain executable after revocation.
            old_target = resolve_dir_within(out, sdir.name)
            old_manifest = (
                resolve_file_within(old_target, "skill.json")
                if old_target is not None else None
            )
            if old_manifest is not None:
                try:
                    old_data = json.loads(old_manifest.read_text(encoding="utf-8"))
                    if (
                        old_data.get("id") == sdir.name
                        and old_data.get("origin", {}).get("runtime") == "entroly"
                    ):
                        shutil.rmtree(old_target)
                except (OSError, ValueError, TypeError):
                    pass
            continue

        meta = loaded["meta"]
        target = out / sdir.name
        old_target = resolve_dir_within(out, sdir.name)
        if old_target is not None:
            shutil.rmtree(old_target)
        elif target.exists():
            skipped.append(sdir.name)
            continue
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
