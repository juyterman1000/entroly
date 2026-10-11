"""The security scan must read the fragment that gets injected.

`_run_pipeline` scanned `frag.get("preview", frag.get("content", ""))`. On the
hierarchical path the Rust engine attaches both keys: `content` is the whole
fragment and `preview` is its first 100 characters
(entroly-core/src/lib.rs, level3_fragments). Preferring `preview` meant the
scan looked at the opening lines of a file while the entire file was sent to
the model, so a credential below the imports was reported as no issue.

Every other consumer of these fragments already reads `content` first --
`entroly/proxy_transform.py` does -- so this call site was the outlier rather
than the convention.
"""
from __future__ import annotations

from entroly.adaptive_pruner import FragmentGuard

# A credential positioned past the preview window, which is where it sits in a
# real config file: imports and a docstring come first.
CONFIG = (
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n\n"
    '"""Service configuration loader for the billing worker."""\n\n'
    "LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')\n"
    "TIMEOUT_S = 30\n\n"
    'AWS_SECRET_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
)


def _hcc_preview(content: str) -> str:
    """The preview the Rust hierarchical path attaches (lib.rs: first 100)."""
    return content[:100] + "..." if len(content) > 100 else content


def _scan_as_pipeline_does(fragment: dict) -> list[str]:
    """Mirror the selection the pipeline makes between the two keys."""
    content = fragment.get("content") or fragment.get("preview", "")
    return FragmentGuard().scan(content, fragment.get("source", ""))


def test_fixture_is_only_detectable_beyond_the_preview_window() -> None:
    """Guard the guard: the test is meaningless if the preview also matches."""
    guard = FragmentGuard()
    assert guard.scan(CONFIG, "cfg.py"), "fixture must be detectable at all"
    assert not guard.scan(_hcc_preview(CONFIG), "cfg.py"), (
        "fixture must place the credential outside the preview window, "
        "otherwise this suite passes for the wrong reason"
    )


def test_scan_sees_the_whole_fragment_when_a_preview_is_present() -> None:
    fragment = {
        "source": "cfg.py",
        "content": CONFIG,
        "preview": _hcc_preview(CONFIG),
    }
    assert _scan_as_pipeline_does(fragment), (
        "a credential in the injected fragment was not scanned"
    )


def test_preview_is_still_used_when_content_is_absent() -> None:
    """Some producers emit only a preview; scanning it beats scanning nothing."""
    fragment = {"source": "cfg.py", "preview": CONFIG}
    assert _scan_as_pipeline_does(fragment)


def test_empty_content_falls_back_rather_than_scanning_nothing() -> None:
    fragment = {"source": "cfg.py", "content": "", "preview": CONFIG}
    assert _scan_as_pipeline_does(fragment)
