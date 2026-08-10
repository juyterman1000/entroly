#!/usr/bin/env python3
"""Does a sound abstraction of Python code answer anything useful?

Candidate #23 (section 9): treat compressed context as an abstract
interpretation, so the guarantee is universally quantified -- for every query in
the supported fragment the answer is a sound over-approximation, never wrong and
possibly `unknown` -- rather than probabilistic over a query distribution.

Soundness is a proof obligation, not a measurement. What must be measured is
**completeness**: how often the abstraction can answer at all instead of
declaring `unknown`. A guarantee that covers almost nothing is a paper, not a
product, so the preregistered kill threshold is 20%.

The abstract domain, per top-level function:

    params        names and kinds from the signature
    calls         may-call set: every syntactic call target
    raises        may-raise set: every explicit `raise`
    mutates       writes to names declared `global`, and module attribute writes

Soundness discipline -- the part that makes this honest rather than flattering.
An over-approximation may only answer NO when it has seen every possibility. A
function using dynamic dispatch can call or mutate anything, so the analysis
must surrender rather than under-report:

    getattr / setattr / delattr    attribute name not statically known
    eval / exec / compile          arbitrary code
    globals() / locals() / vars()  arbitrary name access
    __import__ / importlib         arbitrary module
    *args / **kwargs forwarding    call target's behaviour not determined here

Any of these poisons the relevant predicate to `unknown`. Reporting a confident
NO in their presence would be exactly the silent-wrongness this design exists to
eliminate.

`raises` additionally needs closure over callees to answer NO: a callee may
raise anything. One level of resolution is attempted; where a callee cannot be
resolved the predicate is `unknown`. Answering "may raise" for everything would
be trivially complete and worthless, so it is not counted as an answer.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

_DYNAMIC_CALL = {"getattr", "setattr", "delattr", "eval", "exec", "compile",
                 "globals", "locals", "vars", "__import__"}


@dataclass
class Abstraction:
    """What one function looks like in the abstract domain."""

    name: str
    file: str
    params: tuple[str, ...] = ()
    has_varargs: bool = False
    calls: set[str] = field(default_factory=set)
    raises: set[str] = field(default_factory=set)
    mutates: set[str] = field(default_factory=set)
    # Why a predicate had to surrender, if it did.
    poison: set[str] = field(default_factory=set)

    def decidable(self, predicate: str) -> bool:
        if predicate == "params":
            # A signature is always readable; *args/**kwargs make the parameter
            # SET open, so the question "what are the parameters" is unanswered.
            return not self.has_varargs
        if predicate == "calls":
            return not (self.poison & {"dynamic_call", "forwarding"})
        if predicate == "mutates":
            return not (self.poison & {"dynamic_attr", "dynamic_call"})
        if predicate == "raises":
            return not (self.poison & {"dynamic_call", "unresolved_callee"})
        raise KeyError(predicate)


def _abstract(node: ast.AST, name: str, rel: str,
              resolvable: set[str]) -> Abstraction:
    out = Abstraction(name=name, file=rel)

    args = node.args  # type: ignore[attr-defined]
    out.params = tuple(a.arg for a in args.args if a.arg not in {"self", "cls"})
    out.has_varargs = bool(args.vararg or args.kwarg)

    declared_global: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Global):
            declared_global.update(sub.names)

    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                out.calls.add(func.id)
                if func.id in _DYNAMIC_CALL:
                    out.poison.add(
                        "dynamic_attr" if func.id in
                        {"getattr", "setattr", "delattr"} else "dynamic_call"
                    )
            elif isinstance(func, ast.Attribute):
                out.calls.add(func.attr)
            else:
                # A call on a computed value: target not statically known.
                out.poison.add("dynamic_call")
            # Forwarding *args/**kwargs means the callee's parameters, and so
            # its behaviour, are not determined at this site.
            if any(isinstance(a, ast.Starred) for a in sub.args) or any(
                k.arg is None for k in sub.keywords
            ):
                out.poison.add("forwarding")

        elif isinstance(sub, ast.Raise):
            exc = sub.exc
            if exc is None:
                out.raises.add("<re-raise>")
            elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                out.raises.add(exc.func.id)
            elif isinstance(exc, ast.Name):
                out.raises.add(exc.id)
            else:
                out.poison.add("dynamic_call")

        elif isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name) and leaf.id in declared_global:
                        out.mutates.add(leaf.id)
                    elif isinstance(leaf, ast.Attribute):
                        out.mutates.add(leaf.attr)

    # `raises` can only answer NO with closure over callees. Any callee we
    # cannot resolve to an abstraction may raise anything.
    if not out.calls <= resolvable:
        out.poison.add("unresolved_callee")
    return out


def _tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO,
                         capture_output=True, text=True, check=False)
    paths = []
    for line in out.stdout.splitlines():
        if set(line.split("/")) & {"tests", "test", "benchmarks", "bench"}:
            continue
        p = REPO / line
        try:
            if p.is_file() and p.stat().st_size <= 400_000:
                paths.append(p)
        except OSError:
            continue
    return paths


def build() -> list[Abstraction]:
    trees: dict[str, ast.Module] = {}
    for path in _tracked():
        try:
            trees[str(path.relative_to(REPO)).replace("\\", "/")] = ast.parse(
                path.read_text(encoding="utf-8", errors="replace")
            )
        except (SyntaxError, ValueError, OSError):
            continue

    # Names we can resolve to a definition anywhere in the corpus. Builtins are
    # deliberately excluded: their behaviour is known but not abstracted here,
    # so claiming resolution would overstate what has been proved.
    resolvable = {
        node.name
        for tree in trees.values()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    out: list[Abstraction] = []
    for rel, tree in trees.items():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(_abstract(node, node.name, rel, resolvable))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "sound_abstraction.json")
    args = ap.parse_args()

    abstractions = build()
    if not abstractions:
        print("no functions abstracted")
        return 1

    predicates = ("params", "calls", "mutates", "raises")
    rates = {
        p: sum(1 for a in abstractions if a.decidable(p)) / len(abstractions)
        for p in predicates
    }
    all_four = sum(
        1 for a in abstractions if all(a.decidable(p) for p in predicates)
    ) / len(abstractions)

    poison = Counter(reason for a in abstractions for reason in a.poison)

    payload = {
        "functions": len(abstractions),
        "decidable_rate": rates,
        "all_predicates_rate": all_four,
        "poison_counts": dict(poison),
        "kill_threshold": 0.20,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  functions abstracted: {len(abstractions)}\n")
    print(f"  {'predicate':<14}{'answerable':>12}")
    for p in predicates:
        print(f"  {p:<14}{rates[p]:>11.1%}")
    print(f"  {'ALL FOUR':<14}{all_four:>11.1%}")
    print("\n  why the analysis had to surrender:")
    for reason, count in poison.most_common():
        print(f"    {reason:<20}{count:>6}  ({count/len(abstractions):.1%} of functions)")
    print("\n  preregistered kill threshold: 20% on the useful predicates")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
