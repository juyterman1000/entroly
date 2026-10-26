"""An installed core that fails to load is not reported as "not installed".

The pre-prompt hook's degraded line (#532) derives its cause from
`fallback_reason(native_status(...))`. For a core that is present but cannot
be imported -- a corrupt or ABI-mismatched binary, a missing shared library --
it read "entroly_core is not installed (<path>.so: undefined symbol ...)",
sending the user to install a package they already have instead of
reinstalling or rebuilding it.

Each case runs in a fresh interpreter with a real `entroly_core` package on
sys.path, so the actual import machinery decides what happened.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

PROBE = textwrap.dedent(
    """
    import json
    from entroly.native_status import CORE_SYMBOLS, fallback_reason, native_status
    status = native_status(CORE_SYMBOLS)
    print(json.dumps({"reason": fallback_reason(status), "import_failed": getattr(status, "import_failed", None)}))
    """
)


def _probe(tmp_path, init_source=None):
    stub_root = tmp_path / "stub"
    if init_source is not None:
        package = stub_root / "entroly_core"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(init_source, encoding="utf-8")
    else:
        stub_root.mkdir()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(stub_root), *sys.path])}
    completed = subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_a_core_that_fails_to_import_is_reported_as_failing_to_load(tmp_path) -> None:
    result = _probe(
        tmp_path,
        'raise ImportError("entroly_core.cpython-313-x86_64-linux-gnu.so: '
        'undefined symbol: _PyThreadState_UncheckedGet")\n',
    )

    assert "not installed" not in result["reason"]
    assert "installed but failed to load" in result["reason"]
    assert "undefined symbol" in result["reason"]
    assert result["import_failed"] is True


def test_a_core_missing_its_extension_module_is_reported_as_failing_to_load(tmp_path) -> None:
    result = _probe(tmp_path, "from .entroly_core import *  # noqa: F403\n")

    assert "failed to load" in result["reason"]
    assert result["import_failed"] is True
