#!/usr/bin/env python3
"""Reproducible symbol-extraction audit for every Entroly-mapped language.

This is a conformance benchmark, not a parser popularity count.  Each case has
one small, reviewable source sample and an exact expected symbol set.  A case
passes only when extraction is complete, byte ranges recover the exact source,
and no extra symbol is invented.  Markup, style, and assembly samples are
audited separately because they do not contain declarations in Entroly's code
symbol model.

The optional baseline is loaded from Git without checking it out, so the same
fixtures and installed grammar pack evaluate both implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "language_symbol_coverage.json"
BENCHMARK_MODULE = "benchmarks/language_symbol_coverage.py"
IMPLEMENTATION_COMMIT = "4fecb039c579c7d3e7f534d41bef3c0ede2c5d8a"


@dataclass(frozen=True)
class Case:
    language: str
    path: str
    source: str
    expected_symbols: tuple[str, ...]
    declaration_bearing: bool = True


CASES = (
    Case(
        "ada",
        "sample.adb",
        "procedure Total is\nbegin\n   null;\nend Total;\n",
        ("Total",),
    ),
    Case("asm", "sample.asm", "section .text\n_start:\n    mov rax, 60\n", (), False),
    Case("bash", "sample.sh", "total() { echo 1; }\n", ("total",)),
    Case("c", "sample.c", "int total(int x) { return x; }\n", ("total",)),
    Case("c3", "sample.c3", "fn int total(int x) { return x; }\n", ("total",)),
    Case(
        "c_sharp",
        "sample.cs",
        "public class Cart { public int Total() { return 1; } }\n",
        ("Cart", "Total"),
    ),
    Case(
        "cpp",
        "sample.cpp",
        "struct Cart { int total() { return 1; } };\n",
        ("Cart", "total"),
    ),
    Case("css", "sample.css", ".cart { color: red; }\n", (), False),
    Case(
        "dart",
        "sample.dart",
        "int total(int x) { return x; }\nclass Cart { void add() {} }\n",
        ("Cart", "add", "total"),
    ),
    Case(
        "elixir",
        "sample.ex",
        "defmodule Cart do\n  def total(x), do: x\nend\n",
        ("Cart", "total"),
    ),
    Case(
        "erlang", "sample.erl", "total(X) -> X.\nadd() -> total(1).\n", ("add", "total")
    ),
    Case("fish", "sample.fish", "function total\n    echo 1\nend\n", ("total",)),
    Case("fsharp", "sample.fs", "let total x = x + 1\n", ("total",)),
    Case(
        "go",
        "sample.go",
        "package m\nfunc total() {}\ntype Cart struct{}\n",
        ("Cart", "total"),
    ),
    Case(
        "groovy",
        "sample.groovy",
        "class Cart { def total() { 1 } }\n",
        ("Cart", "total"),
    ),
    Case(
        "haskell",
        "sample.hs",
        "total :: Int -> Int\ntotal x = x\n\ndata Cart = Empty\n",
        ("Cart", "total"),
    ),
    Case("html", "sample.html", '<main class="cart">hello</main>\n', (), False),
    Case(
        "java",
        "sample.java",
        "class Cart { int total() { return 1; } }\n",
        ("Cart", "total"),
    ),
    Case(
        "javascript",
        "sample.js",
        "function total() {}\nclass Cart {}\n",
        ("Cart", "total"),
    ),
    Case(
        "julia",
        "sample.jl",
        "function total(x)\n    x\nend\nstruct Cart\n    x::Int\nend\n",
        ("Cart", "total"),
    ),
    Case(
        "kotlin",
        "sample.kt",
        "fun total(price: Int): Int { return price }\nclass Cart {\n    fun add() {}\n    fun clear() {}\n}\n",
        ("Cart", "add", "clear", "total"),
    ),
    Case("lua", "sample.lua", "function total(x) return x end\n", ("total",)),
    Case("nim", "sample.nim", "proc total(x: int): int = x\n", ("total",)),
    Case("ocaml", "sample.ml", "let total x = x + 1\n", ("total",)),
    Case(
        "php",
        "sample.php",
        "<?php function total($x) { return $x; } class Cart {} ?>\n",
        ("Cart", "total"),
    ),
    Case("proto", "sample.proto", "message Cart {\n  string id = 1;\n}\n", ("Cart",)),
    Case(
        "python",
        "sample.py",
        "def total(x):\n    return x\n\nclass Cart:\n    def add(self):\n        pass\n",
        ("Cart", "add", "total"),
    ),
    Case(
        "r",
        "sample.R",
        "total <- function(x) x + 1\nrunner <- function() total(1)\n",
        ("runner", "total"),
    ),
    Case(
        "ruby",
        "sample.rb",
        "class Cart\n  def total\n    1\n  end\nend\n",
        ("Cart", "total"),
    ),
    Case(
        "rust",
        "sample.rs",
        "pub struct Cart { n: u32 }\nimpl Cart { pub fn total(&self) -> u32 { self.n } }\n",
        ("Cart", "total"),
    ),
    Case(
        "scala",
        "sample.scala",
        "class Cart { def total: Int = 1 }\n",
        ("Cart", "total"),
    ),
    Case(
        "scss", "sample.scss", "$accent: red;\n.cart { color: $accent; }\n", (), False
    ),
    Case(
        "solidity",
        "sample.sol",
        "contract Cart {\n  function total() public {}\n}\n",
        ("Cart", "total"),
    ),
    Case(
        "sql",
        "sample.sql",
        "CREATE FUNCTION add_tax(p int) RETURNS int AS $$ SELECT 1 $$ LANGUAGE SQL;\n",
        ("add_tax",),
    ),
    Case(
        "svelte",
        "sample.svelte",
        "<script>\nfunction total() {}\nclass Cart {}\n</script>\n<div/>\n",
        ("Cart", "total"),
    ),
    Case(
        "swift",
        "sample.swift",
        "struct Cart { func total() -> Int { 1 } }\n",
        ("Cart", "total"),
    ),
    Case(
        "tsx",
        "sample.tsx",
        "export function total(): number { return 1 }\nexport class Cart {}\n",
        ("Cart", "total"),
    ),
    Case(
        "typescript",
        "sample.ts",
        "export function total(): number { return 1 }\nexport class Cart {}\n",
        ("Cart", "total"),
    ),
    Case(
        "v",
        "sample.v",
        "struct Cart {}\nfn total() int { return 1 }\n",
        ("Cart", "total"),
    ),
    Case(
        "vue",
        "sample.vue",
        "<script>\nfunction total() {}\n</script>\n<template><div/></template>\n",
        ("total",),
    ),
    Case(
        "zig",
        "sample.zig",
        "pub fn total() void {}\nfn add() void {}\n",
        ("add", "total"),
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _portable_bytes(path: Path) -> bytes:
    """Return text bytes independent of checkout newline policy."""
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _module_from_source(source: bytes, label: str) -> ModuleType:
    module_name = f"_entroly_tree_sitter_benchmark_{_sha256(source)[:12]}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    if spec is None:
        raise RuntimeError(f"could not create module for {label}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = f"{label}:entroly/tree_sitter_support.py"
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _module_at_ref(ref: str) -> tuple[ModuleType, str]:
    completed = subprocess.run(
        ["git", "show", f"{ref}:entroly/tree_sitter_support.py"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return _module_from_source(completed.stdout, ref), _sha256(completed.stdout)


def _worktree_module() -> tuple[ModuleType, str]:
    source = _portable_bytes(ROOT / "entroly" / "tree_sitter_support.py")
    return _module_from_source(source, "worktree"), _sha256(source)


def _evaluate(module: ModuleType, label: str, source_sha256: str) -> dict[str, Any]:
    case_languages = {case.language for case in CASES}
    mapped_languages = set(module.LANGUAGE_BY_SUFFIX.values())
    if len(case_languages) != len(CASES):
        raise RuntimeError("benchmark cases contain a duplicate language")
    if case_languages != mapped_languages:
        missing = sorted(mapped_languages - case_languages)
        extra = sorted(case_languages - mapped_languages)
        raise RuntimeError(f"benchmark/map drift: missing={missing}, extra={extra}")
    rows: list[dict[str, Any]] = []
    for case in CASES:
        report = module.extract_structural_spans_report(case.source, case.path)
        spans = list(report.items) if report is not None else []
        symbols = tuple(sorted({str(span.name) for span in spans}))
        exact_ranges = all(
            case.source.encode("utf-8")[span.start_byte : span.end_byte].decode("utf-8")
            == span.source
            for span in spans
        )
        complete = report is not None and bool(report.complete)
        syntax_valid = module.validate_structural_syntax(case.source, case.path)
        expected = tuple(sorted(case.expected_symbols))
        passed = (
            syntax_valid is True and complete and exact_ranges and symbols == expected
        )
        rows.append(
            {
                "language": case.language,
                "path": case.path,
                "declaration_bearing": case.declaration_bearing,
                "expected_symbols": list(expected),
                "observed_symbols": list(symbols),
                "syntax_valid": syntax_valid,
                "complete": complete,
                "byte_exact": exact_ranges,
                "backend": report.backend if report is not None else "unavailable",
                "status": (
                    "covered"
                    if passed and case.declaration_bearing
                    else "non_declarative_verified"
                    if passed
                    else "gap"
                ),
            }
        )
    declaration_rows = [row for row in rows if row["declaration_bearing"]]
    covered = [row for row in declaration_rows if row["status"] == "covered"]
    non_declarative = [row for row in rows if not row["declaration_bearing"]]
    non_declarative_verified = [
        row for row in non_declarative if row["status"] == "non_declarative_verified"
    ]
    return {
        "label": label,
        "extractor_sha256": source_sha256,
        "summary": {
            "mapped_languages_audited": len(rows),
            "declaration_bearing_languages": len(declaration_rows),
            "declaration_languages_covered": len(covered),
            "declaration_coverage_pct": round(
                100.0 * len(covered) / len(declaration_rows), 1
            ),
            "non_declarative_languages": len(non_declarative),
            "non_declarative_languages_verified": len(non_declarative_verified),
            "strict_cases_passed": len(covered) + len(non_declarative_verified),
        },
        "cases": rows,
    }


def run(baseline_ref: str | None) -> dict[str, Any]:
    current_module, current_sha = _worktree_module()
    current = _evaluate(current_module, "worktree", current_sha)
    result: dict[str, Any] = {
        "schema": "entroly.language-symbol-coverage.v1",
        "headline_eligible": True,
        "claim_scope": (
            "Exact symbol extraction for one valid representative sample per "
            "mapped language under tree-sitter-language-pack 1.14.3"
        ),
        "sample_size": {
            "mapped_languages": 41,
            "declaration_bearing_languages": 37,
            "non_declarative_languages": 4,
        },
        "benchmark_module": BENCHMARK_MODULE,
        "harness_sha256": _sha256(_portable_bytes(ROOT / BENCHMARK_MODULE)),
        "reproduction_command": (
            "python benchmarks/language_symbol_coverage.py "
            "--baseline-ref 2eeecb8733103fe7234133f48b105f271662b219 "
            "--output benchmarks/results/language_symbol_coverage.json --check"
        ),
        "implementation": {"commit": IMPLEMENTATION_COMMIT},
        "limitations": [
            "One representative sample per language is not complete grammar coverage.",
            "The benchmark measures declarations, traversal, syntax, and byte spans; "
            "it does not measure call resolution, data flow, answer quality, or savings.",
            "Results depend on tree-sitter-language-pack 1.14.3 and must be regenerated "
            "and reviewed for a different grammar pack.",
            "C3, F#, Groovy, Nim, and OCaml remain measured declaration gaps.",
        ],
        "method": {
            "case_contract": "exact symbol set, complete traversal, byte-exact spans",
            "scope": "one representative sample for every language in LANGUAGE_BY_SUFFIX",
            "economic_or_answer_quality_claim": False,
            "case_manifest_sha256": _sha256(
                json.dumps(
                    [
                        {
                            "language": case.language,
                            "path": case.path,
                            "source": case.source,
                            "expected_symbols": case.expected_symbols,
                            "declaration_bearing": case.declaration_bearing,
                        }
                        for case in CASES
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        },
        "environment": {
            "tree_sitter_language_pack": importlib.metadata.version(
                "tree-sitter-language-pack"
            ),
        },
        "current": current,
    }
    if baseline_ref:
        baseline_module, baseline_sha = _module_at_ref(baseline_ref)
        baseline = _evaluate(baseline_module, baseline_ref, baseline_sha)
        current_covered = {
            row["language"] for row in current["cases"] if row["status"] == "covered"
        }
        baseline_covered = {
            row["language"] for row in baseline["cases"] if row["status"] == "covered"
        }
        result["baseline"] = baseline
        result["comparison"] = {
            "newly_covered_declaration_languages": sorted(
                current_covered - baseline_covered
            ),
            "lost_declaration_language_coverage": sorted(
                baseline_covered - current_covered
            ),
            "coverage_delta": len(current_covered) - len(baseline_covered),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run(args.baseline_ref)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    rendered_sha256 = _sha256(rendered.encode("utf-8"))
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar_text = f"{rendered_sha256}  {args.output.name}\n"
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            print(f"benchmark artifact is stale: {args.output}", file=sys.stderr)
            return 1
        if not sidecar.exists() or sidecar.read_text(encoding="ascii") != sidecar_text:
            print(f"benchmark checksum is stale: {sidecar}", file=sys.stderr)
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        sidecar.write_text(sidecar_text, encoding="ascii")
    print(json.dumps(result["current"]["summary"], sort_keys=True))
    if "comparison" in result:
        print(json.dumps(result["comparison"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
