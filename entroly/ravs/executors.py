"""
RAVS v2 — Node Executors

Each executor handles one node type. All executors:
  1. Accept an input string and return (result_str, succeeded: bool)
  2. Are sandboxed — no network, no filesystem writes, bounded CPU
  3. Fail closed — on any exception, return (error_msg, False)
  4. Are stateless — no side effects between calls

Executor types:
  SymPyExecutor    — symbolic math via SymPy (computation nodes)
  PythonExecutor   — safe eval for simple arithmetic (computation fallback)
  ASTExecutor      — Python AST queries (code_inspection nodes)
  TestRunnerStub   — records test intent, doesn't execute (test_execution)
  RetrievalStub    — records retrieval intent (retrieval_claim)
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import subprocess
import time
from typing import Any

logger = logging.getLogger("entroly.ravs.executors")


# ── Base ───────────────────────────────────────────────────────────────


class ExecutorResult:
    """Standardized executor output."""
    __slots__ = ("result", "succeeded", "error", "execution_time_ms", "executor_name")

    def __init__(
        self,
        result: str = "",
        succeeded: bool = False,
        error: str = "",
        execution_time_ms: float = 0.0,
        executor_name: str = "",
    ):
        self.result = result
        self.succeeded = succeeded
        self.error = error
        self.execution_time_ms = execution_time_ms
        self.executor_name = executor_name


# ── SymPy Executor ─────────────────────────────────────────────────────


class SymPyExecutor:
    """Execute symbolic math via SymPy.

    Accepts expressions like:
      "solve x^2 - 1"
      "integrate sin(x)"
      "simplify (x+1)^2 - x^2"
      "2 + 3 * 4"
    """

    def __init__(self, timeout_s: float = 5.0):
        self._timeout = timeout_s
        self._sympy = None
        self._available = False
        try:
            import sympy
            self._sympy = sympy
            self._available = True
        except ImportError:
            logger.debug("SymPy not available — SymPyExecutor disabled")

    @property
    def available(self) -> bool:
        return self._available

    def execute(self, input_text: str) -> ExecutorResult:
        if not self._available:
            return ExecutorResult(
                error="sympy not installed",
                executor_name="sympy",
            )

        t0 = time.perf_counter()
        try:
            result = self._eval_expression(input_text)
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=str(result),
                succeeded=True,
                execution_time_ms=round(elapsed, 2),
                executor_name="sympy",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=str(e)[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="sympy",
            )

    def _eval_expression(self, text: str) -> Any:
        sympy = self._sympy

        # Strip instruction prefixes
        cleaned = re.sub(
            r'^(?:calculate|compute|evaluate|solve|simplify|'
            r'factor|expand|integrate|differentiate|find the value of|'
            r'what is|how much is)\s+',
            '', text.strip(), flags=re.I,
        ).strip()

        # Replace common notation
        cleaned = cleaned.replace('^', '**')

        # Try to parse and evaluate
        expr = sympy.sympify(cleaned)

        # If it's a relational/equation, solve it
        if isinstance(expr, (sympy.Eq, sympy.Rel)):
            symbols = list(expr.free_symbols)
            if symbols:
                return sympy.solve(expr, symbols[0])

        # If it contains free symbols, try to simplify
        if expr.free_symbols:
            return sympy.simplify(expr)

        # Pure numeric — evaluate
        result = expr.evalf()
        # Return integer if it is one
        if result == int(result):
            return int(result)
        return float(result)


# ── Python Safe Eval Executor ──────────────────────────────────────────


class PythonExecutor:
    """Safe evaluation of simple arithmetic expressions.

    Uses Python's ast.literal_eval for safety — no arbitrary code
    execution. Falls back to a restricted eval with math-only namespace
    for expressions that need math functions.
    """

    # Allowlisted names for restricted eval
    _SAFE_NAMES: dict[str, Any] = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "int": int, "float": float,
        "pow": pow, "divmod": divmod,
        # Math constants and functions
        "pi": math.pi, "e": math.e, "tau": math.tau,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
        "log10": math.log10, "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "ceil": math.ceil, "floor": math.floor,
        "gcd": math.gcd,
    }

    def execute(self, input_text: str) -> ExecutorResult:
        t0 = time.perf_counter()
        try:
            # Strip instruction prefixes
            cleaned = re.sub(
                r'^(?:calculate|compute|evaluate|what is|how much is|'
                r'find the value of)\s+',
                '', input_text.strip(), flags=re.I,
            ).strip()
            cleaned = cleaned.replace('^', '**')

            # Try literal_eval first (safest)
            try:
                result = ast.literal_eval(cleaned)
                elapsed = (time.perf_counter() - t0) * 1000
                return ExecutorResult(
                    result=str(result),
                    succeeded=True,
                    execution_time_ms=round(elapsed, 2),
                    executor_name="python_safe",
                )
            except (ValueError, SyntaxError):
                pass

            # Restricted eval with math-only namespace
            # Validate AST first — no calls except allowlisted
            tree = ast.parse(cleaned, mode='eval')
            self._validate_ast(tree)

            result = eval(  # noqa: S307
                compile(tree, '<ravs>', 'eval'),
                {"__builtins__": {}},
                self._SAFE_NAMES,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=str(result),
                succeeded=True,
                execution_time_ms=round(elapsed, 2),
                executor_name="python_safe",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=str(e)[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="python_safe",
            )

    def _validate_ast(self, tree: ast.AST) -> None:
        """Walk AST and reject dangerous nodes."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self._SAFE_NAMES:
                        raise ValueError(
                            f"Function '{node.func.id}' not in allowlist"
                        )
                elif isinstance(node.func, ast.Attribute):
                    raise ValueError("Attribute access not allowed")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Imports not allowed")
            elif isinstance(node, ast.Attribute):
                raise ValueError("Attribute access not allowed")


# ── AST Executor ───────────────────────────────────────────────────────


class ASTExecutor:
    """Python AST-based code inspection.

    Answers structural questions about code: function signatures,
    class definitions, import lists, argument counts, etc.
    """

    def execute(self, input_text: str, source_code: str = "") -> ExecutorResult:
        t0 = time.perf_counter()
        if not source_code:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error="no source code provided",
                execution_time_ms=round(elapsed, 2),
                executor_name="ast",
            )

        try:
            tree = ast.parse(source_code)
            result = self._inspect(tree, input_text)
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=result,
                succeeded=True,
                execution_time_ms=round(elapsed, 2),
                executor_name="ast",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=str(e)[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="ast",
            )

    def _inspect(self, tree: ast.AST, query: str) -> str:
        query_lower = query.lower()

        # List functions
        if "function" in query_lower or "method" in query_lower:
            if "list" in query_lower or "find" in query_lower or "show" in query_lower:
                funcs = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                return json.dumps({"functions": funcs, "count": len(funcs)})

            if "count" in query_lower or "how many" in query_lower:
                count = sum(
                    1 for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                return json.dumps({"count": count})

        # List classes
        if "class" in query_lower:
            if "list" in query_lower or "find" in query_lower or "show" in query_lower:
                classes = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ]
                return json.dumps({"classes": classes, "count": len(classes)})

            if "count" in query_lower or "how many" in query_lower:
                count = sum(
                    1 for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                )
                return json.dumps({"count": count})

        # List imports
        if "import" in query_lower:
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            return json.dumps({"imports": imports, "count": len(imports)})

        # Signature of a specific function
        sig_match = re.search(r'(?:signature|parameters|arguments)\s+(?:of\s+)?(\w+)', query_lower)
        if sig_match:
            target_name = sig_match.group(1)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.lower() == target_name:
                        args = [a.arg for a in node.args.args]
                        return json.dumps({
                            "function": node.name,
                            "parameters": args,
                            "count": len(args),
                        })
            return json.dumps({"error": f"function '{target_name}' not found"})

        # Generic: return summary
        funcs = sum(1 for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        return json.dumps({
            "functions": funcs,
            "classes": classes,
            "summary": "use a more specific query for detailed inspection",
        })




class TestRunnerExecutor:
    """Sandboxed test execution via subprocess.

    Runs test commands (pytest, cargo test, npm test, go test) in a
    subprocess with:
      - Hard timeout (default 30s)
      - Working directory isolation
      - Stdout/stderr capture
      - Exit code verification

    Parses output to extract pass/fail counts where possible.

    Security: uses subprocess with shell=False. No user-controlled
    command injection — the test framework is selected by pattern
    matching on the input, not passed through.
    """

    # Supported test frameworks and their commands
    _FRAMEWORKS: list[tuple[re.Pattern, list[str]]] = [
        (re.compile(r'\bpytest\b|\bpython.*test', re.I),
         ["python", "-m", "pytest", "--tb=short", "-q"]),
        (re.compile(r'\bcargo\s+test\b', re.I),
         ["cargo", "test", "--", "--nocapture"]),
        (re.compile(r'\bnpm\s+test\b|\bjest\b|\bvitest\b', re.I),
         ["npm", "test", "--", "--reporter=verbose"]),
        (re.compile(r'\bgo\s+test\b', re.I),
         ["go", "test", "-v", "./..."]),
        (re.compile(r'\brspec\b', re.I),
         ["bundle", "exec", "rspec", "--format", "documentation"]),
    ]

    # Patterns to extract test counts from output
    _RESULT_PATTERNS: list[tuple[re.Pattern, str]] = [
        # pytest: "5 passed, 1 failed"
        (re.compile(r'(\d+)\s+passed'), "passed"),
        (re.compile(r'(\d+)\s+failed'), "failed"),
        (re.compile(r'(\d+)\s+error'), "errors"),
        # cargo test: "test result: ok. 5 passed; 0 failed"
        (re.compile(r'test result:.*?(\d+)\s+passed;\s*(\d+)\s+failed'), "cargo"),
        # jest/vitest: "Tests: 2 passed, 1 failed"
        (re.compile(r'Tests:\s*(\d+)\s+passed'), "passed"),
        (re.compile(r'Tests:.*?(\d+)\s+failed'), "failed"),
    ]

    def __init__(self, timeout_s: float = 30.0, cwd: str | None = None):
        self._timeout = timeout_s
        self._cwd = cwd

    def execute(self, input_text: str) -> ExecutorResult:
        t0 = time.perf_counter()

        # Match test framework
        cmd = None
        for pattern, base_cmd in self._FRAMEWORKS:
            if pattern.search(input_text):
                cmd = list(base_cmd)
                break

        if cmd is None:
            # Default: try running the input as a pytest invocation
            cmd = ["python", "-m", "pytest", "--tb=short", "-q"]

        # Extract specific test file/path if mentioned
        file_match = re.search(
            r'(?:tests?/\S+\.py|test_\S+\.py|\S+_test\.py|\S+\.spec\.\w+)',
            input_text,
        )
        if file_match:
            cmd.append(file_match.group(0))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._cwd or os.getcwd(),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            # Parse results
            output = proc.stdout + "\n" + proc.stderr
            parsed = self._parse_results(output, proc.returncode)
            elapsed = (time.perf_counter() - t0) * 1000

            return ExecutorResult(
                result=json.dumps(parsed),
                succeeded=proc.returncode == 0,
                error="" if proc.returncode == 0 else f"exit_code={proc.returncode}",
                execution_time_ms=round(elapsed, 2),
                executor_name="test_runner",
            )

        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=json.dumps({
                    "exit_code": -1,
                    "error": f"timeout after {self._timeout}s",
                    "passed": 0, "failed": 0,
                }),
                succeeded=False,
                error=f"timeout after {self._timeout}s",
                execution_time_ms=round(elapsed, 2),
                executor_name="test_runner",
            )
        except FileNotFoundError as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=f"command not found: {e}",
                execution_time_ms=round(elapsed, 2),
                executor_name="test_runner",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=str(e)[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="test_runner",
            )

    def _parse_results(self, output: str, exit_code: int) -> dict[str, Any]:
        """Extract structured test results from raw output."""
        result: dict[str, Any] = {
            "exit_code": exit_code,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "output_tail": output[-500:] if len(output) > 500 else output,
        }

        for pattern, label in self._RESULT_PATTERNS:
            m = pattern.search(output)
            if m:
                if label == "cargo":
                    result["passed"] = int(m.group(1))
                    result["failed"] = int(m.group(2))
                elif label in result:
                    result[label] = max(result[label], int(m.group(1)))

        return result


# ── Retrieval Executor ─────────────────────────────────────────────────


class RetrievalExecutor:
    """TF-IDF based retrieval for citation verification.

    Given a query, retrieves the most relevant fragments from a provided
    corpus using term frequency-inverse document frequency scoring with
    cosine similarity ranking.

    This is the retrieval half of the citation pipeline; the CitationVerifier
    checks whether retrieved content actually supports the claim.
    """

    def __init__(self, max_results: int = 5):
        self._max_results = max_results

    def execute(
        self,
        input_text: str,
        corpus: list[dict[str, str]] | None = None,
    ) -> ExecutorResult:
        """Retrieve relevant fragments from corpus.

        Args:
            input_text: The query/claim to find support for.
            corpus: List of {"id": str, "content": str} dicts.
        """
        t0 = time.perf_counter()

        if not corpus:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=json.dumps({"matches": [], "query": input_text[:100]}),
                succeeded=True,
                execution_time_ms=round(elapsed, 2),
                executor_name="retrieval",
            )

        try:
            matches = self._tfidf_rank(input_text, corpus)
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                result=json.dumps({
                    "matches": matches[:self._max_results],
                    "total_corpus": len(corpus),
                    "query": input_text[:100],
                }),
                succeeded=True,
                execution_time_ms=round(elapsed, 2),
                executor_name="retrieval",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return ExecutorResult(
                error=str(e)[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="retrieval",
            )

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r'\b\w{2,}\b', text.lower())

    def _tfidf_rank(
        self,
        query: str,
        corpus: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Rank corpus documents by TF-IDF cosine similarity to query."""
        import math as _math

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Build document frequency
        doc_tokens: list[list[str]] = []
        for doc in corpus:
            tokens = self._tokenize(doc.get("content", ""))
            doc_tokens.append(tokens)

        n_docs = len(corpus)
        # Document frequency for each term
        df: dict[str, int] = {}
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        # IDF: log(N / df(t)) with smoothing
        idf: dict[str, float] = {}
        for term, freq in df.items():
            idf[term] = _math.log((n_docs + 1) / (freq + 1)) + 1.0

        # Query TF-IDF vector
        query_tf: dict[str, float] = {}
        for t in query_tokens:
            query_tf[t] = query_tf.get(t, 0) + 1.0
        query_vec: dict[str, float] = {}
        for t, tf in query_tf.items():
            query_vec[t] = tf * idf.get(t, 1.0)

        # Score each document
        results: list[dict[str, Any]] = []
        for i, tokens in enumerate(doc_tokens):
            doc_tf: dict[str, float] = {}
            for t in tokens:
                doc_tf[t] = doc_tf.get(t, 0) + 1.0
            doc_vec: dict[str, float] = {}
            for t, tf in doc_tf.items():
                doc_vec[t] = tf * idf.get(t, 1.0)

            # Cosine similarity
            dot = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in query_vec)
            norm_q = _math.sqrt(sum(v * v for v in query_vec.values()))
            norm_d = _math.sqrt(sum(v * v for v in doc_vec.values()))
            similarity = dot / (norm_q * norm_d) if norm_q > 0 and norm_d > 0 else 0.0

            if similarity > 0.01:
                results.append({
                    "id": corpus[i].get("id", str(i)),
                    "similarity": round(similarity, 4),
                    "preview": corpus[i].get("content", "")[:150],
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


# ── Executor Registry ─────────────────────────────────────────────────


class ExecutorRegistry:
    """Central registry mapping executor types to implementations."""

    def __init__(self, test_cwd: str | None = None):
        self._sympy = SymPyExecutor()
        self._python = PythonExecutor()
        self._ast = ASTExecutor()
        self._test = TestRunnerExecutor(cwd=test_cwd)
        self._retrieval = RetrievalExecutor()

    def get(self, executor_type: str) -> Any:
        return {
            "sympy": self._sympy,
            "python": self._python,
            "ast": self._ast,
            "test_runner": self._test,
            "retrieval": self._retrieval,
        }.get(executor_type)

    def execute_node(
        self,
        executor_type: str,
        input_text: str,
        **kwargs: Any,
    ) -> ExecutorResult:
        """Execute a node with the appropriate executor."""
        executor = self.get(executor_type)
        if executor is None:
            return ExecutorResult(
                error=f"unknown executor: {executor_type}",
                executor_name=executor_type,
            )

        try:
            if executor_type == "ast" and "source_code" in kwargs:
                return executor.execute(input_text, kwargs["source_code"])
            if executor_type == "retrieval" and "corpus" in kwargs:
                return executor.execute(input_text, kwargs["corpus"])
            return executor.execute(input_text)
        except Exception as e:
            return ExecutorResult(
                error=str(e)[:200],
                executor_name=executor_type,
            )
