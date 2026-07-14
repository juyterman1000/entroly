#!/usr/bin/env python3
"""Audit Entroly's repository readiness against LobeHub's MCP score model.

This script never claims an external LobeHub result. It separates facts that
can be proved from the repository and local MCP protocol tests from marketplace
state that only LobeHub can confirm after indexing and validation.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WEIGHTS = {
    "claimed": 4,
    "deployMoreThanManual": 12,
    "deployment": 15,
    "license": 8,
    "prompts": 8,
    "readme": 10,
    "resources": 8,
    "tools": 15,
    "validated": 20,
}
REQUIRED = {"deployment", "readme", "tools", "validated"}


@dataclass(frozen=True)
class Evidence:
    check: bool
    source: str
    classification: str
    external: bool = False


def _decorator_kind(node: ast.expr) -> tuple[str | None, str | None]:
    """Return (`tool`/`prompt`/`resource`, optional resource URI)."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None, None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "mcp":
        return None, None
    kind = node.func.attr
    if kind not in {"tool", "prompt", "resource"}:
        return None, None
    uri = None
    if kind == "resource" and node.args and isinstance(node.args[0], ast.Constant):
        if isinstance(node.args[0].value, str):
            uri = node.args[0].value
    return kind, uri


def _mcp_surfaces(server_path: Path) -> dict[str, Any]:
    tree = ast.parse(server_path.read_text(encoding="utf-8"), filename=str(server_path))
    counts = {"tool": 0, "prompt": 0, "resource": 0}
    resources: list[str] = []
    prompts: list[str] = []
    tools: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            kind, uri = _decorator_kind(decorator)
            if kind is None:
                continue
            counts[kind] += 1
            if kind == "tool":
                tools.append(node.name)
            elif kind == "prompt":
                prompts.append(node.name)
            elif uri:
                resources.append(uri)
    return {
        "counts": counts,
        "tools": sorted(tools),
        "prompts": sorted(prompts),
        "resources": sorted(resources),
    }


def collect(root: Path, *, protocol_validated: bool = False) -> dict[str, Any]:
    manifest_path = root / "server.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packages = manifest.get("packages") if isinstance(manifest, dict) else []
    packages = packages if isinstance(packages, list) else []
    non_manual = any(
        isinstance(item, dict) and item.get("registryType") in {"pypi", "npm", "oci", "nuget"}
        for item in packages
    )

    readme = next(
        (path for path in (root / "README.md", root / "PYPI_README.md") if path.is_file() and path.stat().st_size > 0),
        None,
    )
    license_file = next(
        (path for path in (root / "LICENSE", root / "LICENSE.md", root / "LICENSE.txt") if path.is_file()),
        None,
    )
    apache_detected = False
    if license_file:
        license_text = license_file.read_text(encoding="utf-8", errors="replace")[:8192].lower()
        apache_detected = "apache license" in license_text and "version 2.0" in license_text

    surfaces = _mcp_surfaces(root / "entroly" / "server.py")
    counts = surfaces["counts"]

    evidence = {
        "claimed": Evidence(
            False,
            "Only LobeHub can confirm ownership; its detail score currently does not receive claimed state.",
            "external implementation/index state",
            external=True,
        ),
        "deployMoreThanManual": Evidence(
            non_manual,
            f"server.json package registries: {[item.get('registryType') for item in packages if isinstance(item, dict)]}",
            "packaging/discovery",
        ),
        "deployment": Evidence(
            bool(packages),
            f"server.json declares {len(packages)} package deployment option(s)",
            "packaging/discovery",
        ),
        "license": Evidence(
            apache_detected,
            str(license_file.relative_to(root)) if license_file else "No license file found",
            "metadata",
        ),
        "prompts": Evidence(
            counts["prompt"] > 0,
            f"FastMCP prompt decorators: {counts['prompt']}",
            "product capability",
        ),
        "readme": Evidence(
            readme is not None,
            str(readme.relative_to(root)) if readme else "No non-empty README found",
            "documentation",
        ),
        "resources": Evidence(
            counts["resource"] > 0,
            f"FastMCP resource decorators: {counts['resource']}",
            "product capability/security",
        ),
        "tools": Evidence(
            counts["tool"] > 0,
            f"FastMCP tool decorators: {counts['tool']}",
            "product capability",
        ),
        "validated": Evidence(
            protocol_validated,
            (
                "Local stdio initialize/list/get/read protocol tests passed; public LobeHub validation remains external."
                if protocol_validated
                else "Not asserted locally; public LobeHub validation remains external."
            ),
            "runtime validation",
            external=True,
        ),
    }

    score = sum(WEIGHTS[key] for key, item in evidence.items() if item.check)
    required_ready = all(evidence[key].check for key in REQUIRED)
    grade = "A" if required_ready and score >= 80 else "B" if required_ready and score >= 60 else "F"

    return {
        "schema_version": 1,
        "model": {"weights": WEIGHTS, "required": sorted(REQUIRED)},
        "evidence": {key: asdict(value) for key, value in evidence.items()},
        "surfaces": surfaces,
        "local_readiness": {
            "score": score,
            "maximum": 100,
            "grade_if_lobehub_confirmed_same_flags": grade,
            "all_required_ready": required_ready,
            "warning": "This is repository/local-protocol evidence, not the public LobeHub score.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--protocol-validated",
        action="store_true",
        help="Set only after the stdio MCP protocol tests pass in the same run.",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--require-local-completeness",
        action="store_true",
        help="Fail unless deployment, README, tools, prompts, resources, and license are present.",
    )
    args = parser.parse_args()

    report = collect(Path(args.root).resolve(), protocol_validated=args.protocol_validated)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    if args.require_local_completeness:
        required_local = {"deployment", "readme", "tools", "prompts", "resources", "license"}
        missing = [key for key in sorted(required_local) if not report["evidence"][key]["check"]]
        if missing:
            print("Missing local LobeHub evidence: " + ", ".join(missing))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
