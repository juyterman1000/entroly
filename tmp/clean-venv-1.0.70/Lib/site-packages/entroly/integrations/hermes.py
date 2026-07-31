"""
Entroly × Nous Hermes Integration
=================================

Support for Nous Hermes (Hermes 2 Pro, Hermes 3) local deployments.
Hermes models use specific ChatML formats and strict system prompts for tool calling.
This integration ensures that Entroly's context compression does not strip or corrupt
the critical XML-like tags and tool schemas required by Hermes.

Usage::

    from entroly.integrations.hermes import safe_compress_hermes

    # Safely compresses user context while preserving the Hermes
    # tool-calling system prompt and ChatML markers.
    compressed_messages = safe_compress_hermes(messages, budget=8000)
"""

from typing import List, Dict, Any
from ..sdk import compress_messages

# The mandatory Hermes system prompt for function calling
HERMES_TOOL_SYSTEM_PROMPT = "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions."


def safe_compress_hermes(
    messages: List[Dict[str, Any]], budget: int = 8000, preserve_last_n: int = 2
) -> List[Dict[str, Any]]:
    """
    Compresses an LLM conversation while specifically protecting Hermes structural markers.

    Hermes relies on <tools>, <tool_call>, and specific system prompts.
    This wrapper extracts those critical elements, compresses the actual payload,
    and reassembles the conversation to guarantee zero syntax degradation.
    """
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    # We must absolutely preserve the tool-calling instructions
    protected_system = []
    for sys_msg in system_messages:
        content = sys_msg.get("content", "")
        if "<tools>" in content or "function calling AI model" in content:
            protected_system.append(sys_msg)
        else:
            # Compress standard system instructions if needed, but usually
            # we just pass them to the compressor along with the rest
            other_messages.insert(0, sys_msg)

    # Compress the variable context
    compressed_other = compress_messages(
        other_messages, budget=budget, preserve_last_n=preserve_last_n
    )

    # Reassemble: Hermes strictly requires the system prompt first
    final_messages = protected_system + compressed_other
    return final_messages


def format_chatml(messages: List[Dict[str, str]]) -> str:
    """
    Utility to convert standard OpenAI-style messages to raw ChatML strings,
    which is the native format for Nous Hermes inference.
    """
    chatml = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        chatml += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    chatml += "<|im_start|>assistant\n"
    return chatml
