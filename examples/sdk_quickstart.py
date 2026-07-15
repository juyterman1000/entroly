"""No-key Entroly SDK quickstart.

Run from the repository root with ``python examples/sdk_quickstart.py``.
"""

from __future__ import annotations

from entroly import compress, optimize


def main() -> None:
    log = """\
INFO worker-1 completed batch 40
INFO worker-2 completed batch 41
INFO worker-3 completed batch 42
ERROR worker-4 could not connect to the database
INFO worker-5 waiting for retry
"""
    compact = compress(log, budget=35, content_type="log")

    fragments = [
        {
            "source": "database.py",
            "content": "def connect_database():\n    return open_connection(timeout=5)\n",
        },
        {
            "source": "billing.py",
            "content": "def create_invoice():\n    return Invoice()\n",
        },
    ]
    selected = optimize(
        fragments,
        budget=80,
        query="Where is the database connection created?",
    )

    print("Compressed log:")
    print(compact)
    print("\nSelected task context:")
    print(selected["context_text"])


if __name__ == "__main__":
    main()
