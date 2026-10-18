"""Tests for PAC-KV integration in EntrolyProxy."""

import json
from unittest.mock import MagicMock

from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig


def test_proxy_pac_kv_alignment_multi_turn():
    config = ProxyConfig()
    engine = MagicMock()
    proxy = PromptCompilerProxy(engine, config=config)
    proxy._cache_aligner = MagicMock()

    # Mock pac_result
    mock_receipt = MagicMock()
    mock_receipt.provider_page_size = 1024
    mock_receipt.prefix_tokens = 1024
    mock_receipt.dynamic_tail_tokens = 50
    mock_receipt.warm_cache_detected = True
    mock_receipt.economic_decision = "cache_locked_tail_compressed"
    mock_receipt.tail_compressed = True
    mock_receipt.as_dict.return_value = {"warm": True}

    mock_pac_result = MagicMock()
    mock_pac_result.strategy = "cache_locked_tail_compressed"
    mock_pac_result.receipt = mock_receipt
    mock_pac_result.dynamic_tail = "compressed user query"

    proxy._cache_aligner.align_prompt.return_value = mock_pac_result

    body = {
        "model": "claude-3-7-sonnet",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "What is Entroly?"},
            {"role": "assistant", "content": "Entroly is an auditable context-control plane."},
            {"role": "user", "content": "How does PAC-KV alignment work?"},
        ]
    }

    body_out, headers = proxy._apply_pac_kv_alignment(
        conversation_id="conv_test_123",
        provider="anthropic",
        body=body,
    )

    assert headers["X-Entroly-PAC-Strategy"] == "cache_locked_tail_compressed"
    assert headers["X-Entroly-PAC-Page-Size"] == "1024"
    assert headers["X-Entroly-PAC-Warm"] == "true"
    assert headers["X-Entroly-PAC-Decision"] == "cache_locked_tail_compressed"


def test_proxy_pac_kv_single_message_noop():
    config = ProxyConfig()
    engine = MagicMock()
    proxy = PromptCompilerProxy(engine, config=config)

    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello!"},
        ]
    }

    body_out, headers = proxy._apply_pac_kv_alignment(
        conversation_id="conv_single_123",
        provider="openai",
        body=body,
    )

    assert headers == {}
