"""
PROVE — Prose Verification via Causal-Weighted Semantic Entropy
=================================================================

Detects hallucinated prose explanations — cases where the LLM describes
code behavior that contradicts what the code actually does.

The problem PROVE solves:
    An LLM can generate code that is correct AND an explanation that is
    wrong. The existing GRAPHS verifier checks the code; PROVE checks
    the explanation. Example:

        Code:   df.groupby("region").sum()
        Claim:  "This averages sales by region"  ← HALLUCINATED

    The code sums, but the explanation says "averages". All symbols are
    real, the code compiles, pyright is happy — but the explanation is
    a semantic hallucination.

Mathematical Foundation (Kuhn, Gal & Farquhar ICLR 2023 — adapted)
------------------------------------------------------------------
Original Semantic Entropy samples N completions, clusters by
bidirectional entailment, and computes H(clusters). This requires
an LLM. We don't have one at verification time.

PROVE adapts the insight using a ZERO-LLM approach:

1. Extract *causal predicates* from the prose: verb-object pairs that
   make falsifiable claims about code behavior.

   "This averages sales by region" → [("averages", "sales"), ("by", "region")]

2. Extract *structural predicates* from the AST: what the code actually
   does via static analysis.

   df.groupby("region").sum() → [("groups_by", "region"), ("sums", "columns")]

3. Compute *predicate alignment* as a Bayesian posterior:

   For each prose predicate p and code predicate c:
     sim(p, c) = token_jaccard(p, c) × verb_synonym_bonus(p.verb, c.verb)

   Alignment score:
     A = (1/|P|) Σ_p max_c sim(p, c)

   Hallucination posterior:
     P(halu | prose, code) = 1 - A

4. Weight by *causal importance*: predicates involving control flow
   (if/else, loops, returns) carry 2× weight because errors there
   propagate to downstream reasoning.

Complexity
----------
Training: None (fully static).
Inference: O(|prose_tokens| × |ast_nodes|) per verification.
Memory: O(|prose| + |code|).

References
----------
- Kuhn, Gal & Farquhar (ICLR 2023): Semantic Entropy
- Shi et al. (ICML 2023): Irrelevant Context Distraction
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any


# ── Verb synonym clusters ────────────────────────────────────────────
# Hand-curated from the ~100 most common code-explanation verbs.
# Each cluster maps surface verbs to canonical code operations.

_VERB_CLUSTERS: dict[str, str] = {}
_VERB_CLUSTER_GROUPS = [
    # Aggregation
    (["sum", "sums", "adds", "totals", "accumulates", "aggregates"], "sum"),
    (["average", "averages", "means", "computes_mean", "mean"], "mean"),
    (["count", "counts", "tallies", "enumerates", "value_counts"], "count"),
    (["max", "maximum", "maximizes", "finds_max", "largest"], "max"),
    (["min", "minimum", "minimizes", "finds_min", "smallest"], "min"),
    # Filtering
    (["filter", "filters", "selects", "chooses", "picks", "keeps", "retains",
      "dropna", "fillna", "where", "query", "isin"], "filter"),
    (["remove", "removes", "drops", "deletes", "discards", "excludes", "drop"], "remove"),
    # Transformation
    (["sort", "sorts", "orders", "arranges", "ranks",
      "sort_values", "sort_index", "sorted", "argsort", "nlargest", "nsmallest"], "sort"),
    (["reverse", "reverses", "inverts", "flips"], "reverse"),
    (["map", "maps", "transforms", "converts", "applies", "apply", "applymap"], "map"),
    (["merge", "merges", "joins", "combines", "concatenates", "concat", "join"], "merge"),
    # I/O
    (["read", "reads", "loads", "opens", "imports", "fetches", "gets",
      "read_csv", "read_json", "read_excel", "read_parquet", "read_sql",
      "open", "load", "fetch", "get", "download"], "read"),
    (["write", "writes", "saves", "stores", "exports", "outputs", "dumps",
      "to_csv", "to_json", "to_excel", "to_parquet", "to_sql",
      "save", "dump", "export"], "write"),
    (["print", "prints", "displays", "shows", "outputs", "logs",
      "info", "describe", "head", "tail"], "print"),
    # Control
    (["return", "returns", "yields", "produces", "gives", "emits"], "return"),
    (["raise", "raises", "throws", "signals"], "raise"),
    (["loop", "loops", "iterates", "traverses", "walks", "scans",
      "iterrows", "itertuples", "items"], "loop"),
    (["check", "checks", "validates", "verifies", "tests", "asserts",
      "assert_frame_equal", "assertTrue", "assertEqual"], "check"),
    # Data
    (["group", "groups", "partitions", "segments", "clusters", "buckets",
      "groupby", "group_by", "pivot", "pivot_table", "crosstab"], "group"),
    (["split", "splits", "divides", "separates", "breaks"], "split"),
    (["append", "appends", "pushes", "extends", "adds_to"], "append"),
    (["insert", "inserts", "puts", "places"], "insert"),
    (["update", "updates", "modifies", "changes", "sets", "assigns",
      "replace", "rename", "set_index", "reset_index"], "update"),
    (["create", "creates", "initializes", "constructs", "builds", "makes",
      "DataFrame", "Series", "array", "zeros", "ones", "empty"], "create"),
]

for _verbs, _canonical in _VERB_CLUSTER_GROUPS:
    for _v in _verbs:
        _VERB_CLUSTERS[_v] = _canonical


# ── Causal weight multipliers ────────────────────────────────────────

# Predicates involving these AST node types carry higher weight because
# errors in control-flow description propagate to reasoning.
_CAUSAL_WEIGHTS = {
    "If":           2.0,
    "For":          1.5,
    "While":        1.5,
    "Return":       2.0,
    "Raise":        2.0,
    "Assert":       1.8,
    "Try":          1.5,
    "With":         1.2,
    "Assign":       1.0,
    "Call":         1.0,
    "Attribute":    0.8,
}


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class Predicate:
    """A falsifiable claim about code behavior."""
    verb: str                   # canonical verb
    object_: str                # what the verb acts on
    raw: str                    # original text fragment
    causal_weight: float = 1.0  # importance multiplier
    source: str = ""            # "prose" or "code"


@dataclass
class AlignmentResult:
    """Per-predicate alignment score."""
    prose_predicate: Predicate
    best_code_match: Predicate | None
    similarity: float           # [0, 1]
    weighted_contribution: float


@dataclass
class ProveResult:
    """Full PROVE verification result."""
    prose: str
    code: str
    prose_predicates: list[Predicate]
    code_predicates: list[Predicate]
    alignments: list[AlignmentResult]
    alignment_score: float      # A ∈ [0, 1], higher = more aligned
    hallucination_risk: float   # P(halu) = 1 - A
    n_unaligned: int            # predicates with sim < 0.3
    verdict: str                # "pass", "warn", "fail"

    def explain(self, max_items: int = 15) -> str:
        lines = [
            "=== PROVE — Prose Verification ===",
            f"verdict: {self.verdict}  "
            f"alignment={self.alignment_score:.3f}  "
            f"P(halu)={self.hallucination_risk:.3f}  "
            f"unaligned={self.n_unaligned}/"
            f"{len(self.prose_predicates)}",
            "",
        ]
        # Show worst alignments first
        sorted_a = sorted(self.alignments, key=lambda a: a.similarity)
        for a in sorted_a[:max_items]:
            tag = "[X]" if a.similarity < 0.3 else "[!]" if a.similarity < 0.6 else "[ok]"
            match_str = (
                f"↔ {a.best_code_match.verb}({a.best_code_match.object_})"
                if a.best_code_match else "↔ (no match)"
            )
            lines.append(
                f"  {tag} sim={a.similarity:.2f} w={a.weighted_contribution:.2f}  "
                f"{a.prose_predicate.verb}({a.prose_predicate.object_})  "
                f"{match_str}"
            )
        return "\n".join(lines)


# ── Predicate extraction: Prose ──────────────────────────────────────


# Regex patterns to extract verb-object pairs from natural language.
# Deliberately simple — we want high recall, tolerate noise.
_PROSE_VERB_OBJ = re.compile(
    r"\b("
    + "|".join(sorted(set(_VERB_CLUSTERS.keys()), key=len, reverse=True))
    + r")\s+(?:the\s+|a\s+|an\s+|all\s+|each\s+)?"
    r"(\w+(?:\s+\w+)?)",
    re.IGNORECASE,
)

# Negation detector — "does NOT sum" inverts the predicate.
_NEGATION_RE = re.compile(
    r"\b(?:not|don'?t|doesn'?t|never|no|won'?t|cannot|can'?t)\s+",
    re.IGNORECASE,
)


def _extract_prose_predicates(text: str) -> list[Predicate]:
    """Extract verb-object predicates from prose text."""
    predicates: list[Predicate] = []
    seen: set[tuple[str, str]] = set()

    for m in _PROSE_VERB_OBJ.finditer(text):
        verb_raw = m.group(1).lower()
        obj_raw = m.group(2).lower().strip()

        # Canonicalize verb
        verb = _VERB_CLUSTERS.get(verb_raw, verb_raw)

        # Check for negation in the ~20 chars before the verb
        prefix = text[max(0, m.start() - 25):m.start()]
        is_negated = bool(_NEGATION_RE.search(prefix))
        if is_negated:
            verb = f"not_{verb}"

        key = (verb, obj_raw)
        if key in seen:
            continue
        seen.add(key)

        predicates.append(Predicate(
            verb=verb,
            object_=obj_raw,
            raw=m.group(0),
            source="prose",
        ))

    return predicates


# ── Predicate extraction: Code (AST) ────────────────────────────────


def _extract_code_predicates(source: str) -> list[Predicate]:
    """Extract structural predicates from Python code via AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    predicates: list[Predicate] = []
    seen: set[tuple[str, str]] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            # Extract function/method calls → verb-object predicates
            func_name = _ast_call_name(node)
            if func_name:
                verb = _VERB_CLUSTERS.get(func_name.lower(), func_name.lower())
                # Object is the first string arg or the calling object
                obj = _ast_first_arg_str(node) or _ast_call_receiver(node) or ""
                w = _CAUSAL_WEIGHTS.get("Call", 1.0)
                key = (verb, obj)
                if key not in seen:
                    seen.add(key)
                    predicates.append(Predicate(
                        verb=verb, object_=obj, raw=func_name,
                        causal_weight=w, source="code",
                    ))

            # Handle chained method calls: df.groupby("x").sum()
            # The outer Call is sum(), the inner Call (in node.func.value)
            # is groupby("x"). We need to walk the chain.
            if isinstance(node.func, ast.Attribute):
                inner = node.func.value
                while isinstance(inner, ast.Call):
                    inner_name = _ast_call_name(inner)
                    if inner_name:
                        inner_verb = _VERB_CLUSTERS.get(
                            inner_name.lower(), inner_name.lower()
                        )
                        inner_obj = (
                            _ast_first_arg_str(inner)
                            or _ast_call_receiver(inner)
                            or ""
                        )
                        inner_key = (inner_verb, inner_obj)
                        if inner_key not in seen:
                            seen.add(inner_key)
                            predicates.append(Predicate(
                                verb=inner_verb, object_=inner_obj,
                                raw=inner_name,
                                causal_weight=_CAUSAL_WEIGHTS.get("Call", 1.0),
                                source="code",
                            ))
                    # Continue walking the chain
                    if isinstance(inner.func, ast.Attribute):
                        inner = inner.func.value
                    else:
                        break

            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            target = _name_of(node.target) or "item"
            iter_name = _name_of(node.iter) or "iterable"
            w = _CAUSAL_WEIGHTS.get("For", 1.5)
            predicates.append(Predicate(
                verb="loop", object_=f"{target}_in_{iter_name}",
                raw=f"for {target} in {iter_name}",
                causal_weight=w, source="code",
            ))
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            w = _CAUSAL_WEIGHTS.get("While", 1.5)
            predicates.append(Predicate(
                verb="loop", object_="while_condition",
                raw="while ...", causal_weight=w, source="code",
            ))
            self.generic_visit(node)

        def visit_If(self, node: ast.If) -> None:
            test_str = _name_of(node.test) or "condition"
            w = _CAUSAL_WEIGHTS.get("If", 2.0)
            predicates.append(Predicate(
                verb="check", object_=test_str,
                raw=f"if {test_str}", causal_weight=w, source="code",
            ))
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return) -> None:
            val = _name_of(node.value) or "result" if node.value else "None"
            w = _CAUSAL_WEIGHTS.get("Return", 2.0)
            predicates.append(Predicate(
                verb="return", object_=val,
                raw=f"return {val}", causal_weight=w, source="code",
            ))
            self.generic_visit(node)

        def visit_Raise(self, node: ast.Raise) -> None:
            exc = _name_of(node.exc) if node.exc else "Exception"
            w = _CAUSAL_WEIGHTS.get("Raise", 2.0)
            predicates.append(Predicate(
                verb="raise", object_=exc,
                raw=f"raise {exc}", causal_weight=w, source="code",
            ))
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                t_name = _name_of(target) or "var"
                v_name = _name_of(node.value) or "value"
                w = _CAUSAL_WEIGHTS.get("Assign", 1.0)
                predicates.append(Predicate(
                    verb="update", object_=t_name,
                    raw=f"{t_name} = {v_name}",
                    causal_weight=w, source="code",
                ))
            self.generic_visit(node)

    Visitor().visit(tree)
    return predicates


def _ast_call_name(node: ast.Call) -> str | None:
    """Extract the function/method name from a Call node."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _ast_call_receiver(node: ast.Call) -> str:
    """Extract the object being called on (e.g., 'df' from df.sum())."""
    if isinstance(node.func, ast.Attribute):
        return _name_of(node.func.value) or ""
    return ""


def _ast_first_arg_str(node: ast.Call) -> str:
    """Extract the first positional argument if it's a string constant."""
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return ""


def _name_of(node: ast.AST | None) -> str:
    """Best-effort name extraction from an AST node."""
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Call):
        return _ast_call_name(node) or ""
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        parts = [_name_of(e) for e in node.elts[:3]]
        return ",".join(p for p in parts if p)
    return ""


# ── Predicate alignment ─────────────────────────────────────────────


def _token_set(s: str) -> set[str]:
    """Lowercase word tokens."""
    return set(re.findall(r"[a-z][a-z0-9_]*", s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _verb_similarity(v1: str, v2: str) -> float:
    """Similarity between two verbs via canonical form.

    Same canonical -> 1.0
    One is negation of the other -> 0.0 (active contradiction)
    One canonical is a substring of the other -> 0.7 (partial match)
    Different canonical -> 0.2 * token_overlap (weak signal)
    """
    c1 = _VERB_CLUSTERS.get(v1, v1)
    c2 = _VERB_CLUSTERS.get(v2, v2)

    # Exact canonical match
    if c1 == c2:
        return 1.0

    # Negation contradiction
    if c1 == f"not_{c2}" or c2 == f"not_{c1}":
        return 0.0

    # Substring containment: sort_values -> sort, groupby -> group
    # This catches compound method names not in the cluster table.
    if c1 in c2 or c2 in c1:
        return 0.7

    # Fallback: token jaccard on the verb surface forms
    return 0.2 * _jaccard(_token_set(v1), _token_set(v2))


def _predicate_similarity(p: Predicate, c: Predicate) -> float:
    """Similarity between a prose predicate and a code predicate.

    sim(p, c) = 0.6 × verb_sim(p.verb, c.verb) + 0.4 × obj_jaccard(p.obj, c.obj)

    Special case: if the verb similarity is exactly 0.0 (active
    contradiction, e.g. negation), cap total similarity at 0.15.
    A negated verb can never be "saved" by matching objects.
    """
    v_sim = _verb_similarity(p.verb, c.verb)
    o_sim = _jaccard(_token_set(p.object_), _token_set(c.object_))
    combined = 0.6 * v_sim + 0.4 * o_sim

    # Hard cap: verb contradiction dominates
    if v_sim == 0.0:
        return min(combined, 0.15)
    return combined


def _align_predicates(
    prose_preds: list[Predicate],
    code_preds: list[Predicate],
) -> list[AlignmentResult]:
    """Align each prose predicate to its best code predicate match."""
    results: list[AlignmentResult] = []

    for pp in prose_preds:
        best_sim = 0.0
        best_match: Predicate | None = None

        for cp in code_preds:
            sim = _predicate_similarity(pp, cp)
            if sim > best_sim:
                best_sim = sim
                best_match = cp

        # Weighted contribution: sim × causal_weight (from best match)
        weight = best_match.causal_weight if best_match else 1.0
        results.append(AlignmentResult(
            prose_predicate=pp,
            best_code_match=best_match,
            similarity=best_sim,
            weighted_contribution=best_sim * weight,
        ))

    return results


# ── Top-level API ────────────────────────────────────────────────────


def prove_verify(
    prose: str,
    code: str,
    fail_threshold: float = 0.35,
    warn_threshold: float = 0.60,
) -> ProveResult:
    """Verify prose explanation against code using PROVE.

    Args:
        prose: Natural language explanation of the code.
        code: Python source code being explained.
        fail_threshold: Alignment below this → verdict="fail".
        warn_threshold: Alignment below this → verdict="warn".

    Returns:
        ProveResult with alignment score and per-predicate breakdown.
    """
    prose_preds = _extract_prose_predicates(prose)
    code_preds = _extract_code_predicates(code)

    if not prose_preds:
        # No falsifiable claims in prose → cannot verify → pass
        return ProveResult(
            prose=prose, code=code,
            prose_predicates=[], code_predicates=code_preds,
            alignments=[], alignment_score=1.0,
            hallucination_risk=0.0, n_unaligned=0,
            verdict="pass",
        )

    alignments = _align_predicates(prose_preds, code_preds)

    # Weighted alignment score
    total_weight = sum(
        a.prose_predicate.causal_weight
        for a in alignments
    )
    if total_weight > 0:
        alignment_score = sum(
            a.similarity * a.prose_predicate.causal_weight
            for a in alignments
        ) / total_weight
    else:
        alignment_score = 0.0

    n_unaligned = sum(1 for a in alignments if a.similarity < 0.3)
    halu_risk = 1.0 - alignment_score

    if alignment_score < fail_threshold:
        verdict = "fail"
    elif alignment_score < warn_threshold:
        verdict = "warn"
    else:
        verdict = "pass"

    return ProveResult(
        prose=prose, code=code,
        prose_predicates=prose_preds,
        code_predicates=code_preds,
        alignments=alignments,
        alignment_score=round(alignment_score, 4),
        hallucination_risk=round(halu_risk, 4),
        n_unaligned=n_unaligned,
        verdict=verdict,
    )
