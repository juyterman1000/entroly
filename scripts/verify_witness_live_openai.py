"""Optional live OpenAI smoke test for WITNESS.

This is intentionally not part of the default pytest suite because it spends
provider tokens. It verifies the two production response modes that usually
break output gateways: non-streaming text and streaming text. Set
OPENAI_API_KEY, then run:

    python scripts/verify_witness_live_openai.py
"""

from __future__ import annotations

import os
import sys

from entroly.witness import WitnessAnalyzer


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY is not set.")
        return

    try:
        import openai
    except Exception as exc:
        print(f"SKIP: openai package is unavailable: {exc}")
        return

    client = openai.OpenAI(timeout=20.0)
    analyzer = WitnessAnalyzer(profile="rag", use_nli=False)
    context = "The verified evidence says the capital of Germany is Berlin."

    try:
        non_stream = client.chat.completions.create(
            model=os.getenv("ENTROLY_WITNESS_LIVE_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Answer with exactly this sentence: Paris is the capital of Germany."},
                {"role": "user", "content": "What is the capital of Germany?"},
            ],
            max_tokens=20,
            temperature=0,
        )
    except Exception as exc:
        print(f"SKIP: OpenAI live call failed before verification: {type(exc).__name__}")
        return
    text = non_stream.choices[0].message.content or ""
    _, rewrite = analyzer.analyze_and_rewrite(context, text, mode="strict")
    if "Paris is the capital of Germany" in rewrite.output:
        raise SystemExit("FAIL: non-streaming unsupported claim was not suppressed")

    chunks: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=os.getenv("ENTROLY_WITNESS_LIVE_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Answer with exactly this sentence: Paris is the capital of Germany."},
                {"role": "user", "content": "What is the capital of Germany?"},
            ],
            max_tokens=20,
            temperature=0,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                chunks.append(delta)
    except Exception as exc:
        print(f"SKIP: OpenAI streaming live call failed before verification: {type(exc).__name__}")
        return
    streamed_text = "".join(chunks)
    _, stream_rewrite = analyzer.analyze_and_rewrite(context, streamed_text, mode="strict")
    if "Paris is the capital of Germany" in stream_rewrite.output:
        raise SystemExit("FAIL: streaming unsupported claim was not suppressed")

    print("PASS: live OpenAI non-streaming and streaming outputs were verified and suppressed.")


if __name__ == "__main__":
    sys.exit(main())
