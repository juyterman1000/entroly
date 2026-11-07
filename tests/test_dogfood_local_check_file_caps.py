"""The README's two "check that it worked" commands must index the same files.

On pallets/click (155 indexable files) `entroly verify-claims` reported
"114 files, 284,821 tokens" and `entroly simulate` a minute later "94 of 155
files (182,714 tokens)": verify-claims samples up to 120 files, simulate (and
perf and demo, which share its options) up to 100. A new user comparing step 1
with step 2 saw two sizes for one repository and no reason for the gap.
"""

from __future__ import annotations

import entroly.cli as cli
from entroly.cli_parser import build_cli_parser


def _parser():
    return build_cli_parser(
        version="test",
        wrap_agent_names=cli._wrap_agent_names,
        reason_options=cli._EXIT_REASON_OPTIONS,
        benefit_options=cli._EXIT_BENEFIT_OPTIONS,
        surface_options=cli._EXIT_SURFACE_OPTIONS,
        duration_options=cli._EXIT_DURATION_OPTIONS,
    )


def test_local_checks_share_one_file_cap() -> None:
    parser = _parser()
    caps = {
        command: parser.parse_args([command]).max_files
        for command in ("verify-claims", "simulate", "perf", "demo")
    }
    assert len(set(caps.values())) == 1, caps


def test_simulate_without_the_option_uses_the_same_cap() -> None:
    import inspect

    source = inspect.getsource(cli._run_local_simulation)
    verify_default = _parser().parse_args(["verify-claims"]).max_files
    assert f'getattr(args, "max_files", {verify_default})' in source
