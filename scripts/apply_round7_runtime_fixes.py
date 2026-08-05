"""Apply the PR #280 bounded-demo repair deterministically.

This helper is temporary and restored from ``origin/main`` by the trusted
repair workflow after Linux and Windows validate the exact same transform.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_cli_demo() -> None:
    path = ROOT / "entroly" / "cli.py"
    text = path.read_text(encoding="utf-8")

    if "Run the bounded no-key demo through the shared simulation path." not in text:
        start = text.index("def cmd_demo(args):\n")
        end = text.index("\ndef _simulation_queries(args) -> list[str]:\n", start)
        replacement = textwrap.dedent(
            '''
            def cmd_demo(args):
                """Run the bounded no-key demo through the shared simulation path.

                ``demo``, ``simulate``, and ``perf`` must measure the same indexed
                surface. A separate auto-index path previously bypassed the file cap
                and could spend minutes ingesting a large repository before printing.
                """
                report = _run_local_simulation(args)
                if getattr(args, "json", False):
                    print(json.dumps(report, indent=2))
                    return

                _print_local_simulation(
                    report,
                    title="Entroly Demo",
                    include_perf=False,
                )
                if report["files_indexed"] == 0:
                    return

                from entroly.value_tracker import estimate_cost

                per_query_saved = report["total_tokens_saved"] // max(
                    len(report["queries"]), 1
                )
                print(
                    f"  {C.BOLD}Per-query savings{C.RESET} "
                    f"(current input rates, {per_query_saved:,} tokens saved/query):"
                )
                for model in ("gpt-4o", "claude-sonnet-4", "gemini-2.5-pro"):
                    cost = estimate_cost(per_query_saved, model)
                    print(
                        f"    {C.CYAN}{model:25s}{C.RESET} "
                        f"${cost:.4f}/query"
                    )
                print(
                    f"  {C.GRAY}Multiply by your actual request volume. "
                    f"We don't know what that is.{C.RESET}"
                )

                recommended = _recommend_quality(
                    _detect_project_type(), report["files_indexed"]
                )
                print(f"""
                {C.GREEN}{C.BOLD}Get started:{C.RESET}
                  {C.CYAN}entroly go{C.RESET}                One command: init + proxy + dashboard
                  {C.CYAN}entroly proxy --quality {recommended}{C.RESET}  Start optimizing
              """)
            '''
        ).lstrip()
        text = text[:start] + replacement + text[end:]

    old_parser = """    # entroly demo (Gap #41)\n    subparsers.add_parser(\n        \"demo\",\n        help=\"Quick-win demo: before/after comparison showing token savings\",\n    )\n"""
    new_parser = """    # entroly demo (bounded local measurement)\n    demo_parser = subparsers.add_parser(\n        \"demo\",\n        help=\"Quick-win demo: before/after comparison showing token savings\",\n    )\n    _add_local_measure_args(demo_parser)\n"""
    if old_parser in text:
        text = text.replace(old_parser, new_parser, 1)
    elif new_parser not in text:
        raise RuntimeError("demo parser block is neither original nor repaired")

    path.write_text(text, encoding="utf-8")


def write_regression_tests() -> None:
    path = ROOT / "tests" / "test_demo_dogfood.py"
    path.write_text(
        '''from __future__ import annotations

import ast
import inspect
import json
from types import SimpleNamespace

from entroly import cli


def test_demo_delegates_to_bounded_shared_simulation(monkeypatch, capsys):
    args = SimpleNamespace(
        max_files=7,
        budget=2048,
        baseline=0,
        query=[],
        json=False,
    )
    report = {
        "files_indexed": 7,
        "repo_tokens_indexed": 20_000,
        "budget": 2048,
        "baseline_tokens_per_query": 20_000,
        "queries": [{"query": "q"}],
        "total_tokens_saved": 12_000,
    }
    observed = {}

    def fake_run(received):
        observed["args"] = received
        return report

    monkeypatch.setattr(cli, "_run_local_simulation", fake_run)
    monkeypatch.setattr(cli, "_print_local_simulation", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_detect_project_type", lambda: "python")
    monkeypatch.setattr(cli, "_recommend_quality", lambda *_: "balanced")

    cli.cmd_demo(args)

    assert observed["args"] is args
    assert "Get started" in capsys.readouterr().out


def test_demo_cannot_reintroduce_direct_auto_indexing():
    tree = ast.parse(inspect.getsource(cli.cmd_demo))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_run_local_simulation" in calls
    assert "auto_index" not in calls


def test_demo_json_output_is_machine_readable(monkeypatch, capsys):
    args = SimpleNamespace(json=True)
    report = {
        "mode": "local_no_llm",
        "files_indexed": 0,
        "queries": [],
        "total_tokens_saved": 0,
    }
    monkeypatch.setattr(cli, "_run_local_simulation", lambda _: report)

    cli.cmd_demo(args)

    assert json.loads(capsys.readouterr().out) == report
''',
        encoding="utf-8",
    )


def main() -> None:
    patch_cli_demo()
    write_regression_tests()


if __name__ == "__main__":
    main()
