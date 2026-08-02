from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_entry_points_bind_secure_store_without_import_order_dependency() -> None:
    script = r"""
import sys
sys.path.insert(0, __REPO_ROOT__)
from entroly.compression_retrieval_store import (
    CompressionRetrievalStore as direct_legacy_store,
)
import entroly
import entroly.compression_dashboard as dashboard
import entroly.compression_proxy as compression_proxy
import entroly.compression_proxy_direct as compression_proxy_direct
import entroly.compression_verification_loop as verification_loop
import entroly.proxy as provider_proxy
import entroly.session_rescue as session_rescue
from entroly import compression_retrieval_store as legacy_module

secure = entroly.CompressionRetrievalStore
assert direct_legacy_store is secure
assert secure.__module__ == "entroly.compression_retrieval_store_secure"
assert legacy_module.CompressionRetrievalStore is secure
assert compression_proxy.CompressionRetrievalStore is secure
assert compression_proxy_direct.CompressionRetrievalStore is secure
assert dashboard.CompressionRetrievalStore is secure
assert verification_loop.CompressionRetrievalStore is secure
assert provider_proxy.CompressionRetrievalStore is secure
assert session_rescue.CompressionRetrievalStore is secure

store = secure(scope_id="clean-interpreter")
written = store.put(
    original_text="recoverable evidence",
    compressed_text="[omitted]",
    receipt={
        "original_tokens": 8,
        "compressed_tokens": 2,
        "omitted_spans": [{"start_line": 1, "end_line": 1}],
    },
)
assert store.get_receipt(written.receipt_id) is not None
""".replace("__REPO_ROOT__", repr(str(REPO_ROOT)))
    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
