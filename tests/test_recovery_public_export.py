from __future__ import annotations

import entroly
from entroly.compression_retrieval_store_secure import CompressionRetrievalStore


def test_package_root_exports_secure_recovery_store() -> None:
    assert entroly.CompressionRetrievalStore is CompressionRetrievalStore
