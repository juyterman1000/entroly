from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_all_public_recovery_entrypoints_use_secure_store_in_clean_interpreter() -> None:
    source_root = Path(__file__).resolve().parents[1]
    script = r'''
import json
import sys
sys.path.insert(0, sys.argv[1])
import entroly
import entroly.compression_dashboard as dashboard
import entroly.compression_proxy as proxy_compression
import entroly.compression_proxy_direct as proxy_direct
import entroly.compression_retrieval_store as legacy
import entroly.compression_retrieval_store_secure as secure
import entroly.compression_verification_loop as verification
import entroly.proxy as provider_proxy
import entroly.session_rescue as rescue

surfaces = {
    "package": entroly.CompressionRetrievalStore,
    "legacy": legacy.CompressionRetrievalStore,
    "secure": secure.CompressionRetrievalStore,
    "dashboard": dashboard.CompressionRetrievalStore,
    "compression_proxy": proxy_compression.CompressionRetrievalStore,
    "compression_proxy_direct": proxy_direct.CompressionRetrievalStore,
    "verification": verification.CompressionRetrievalStore,
    "provider_proxy": provider_proxy.CompressionRetrievalStore,
    "session_rescue": rescue.CompressionRetrievalStore,
}
print(json.dumps({name: cls.__module__ for name, cls in surfaces.items()}, sort_keys=True))
assert len(set(surfaces.values())) == 1
assert next(iter(surfaces.values())) is secure.CompressionRetrievalStore
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_root)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    modules = json.loads(completed.stdout)
    assert set(modules.values()) == {"entroly.compression_retrieval_store_secure"}
