"""`entroly compress` and `entroly recover`.

These exist because the recovery contract was unreachable from a terminal.
A user could compress through the SDK and get a digest back, but had no way to
turn that digest into the original bytes without writing Python. A recovery
guarantee nobody can invoke is a claim, not a feature.

`compress` prints what was kept, what it cost, and the digest that recovers the
original. `recover` verifies the digest AND the byte length before returning
anything, so a corrupt or substituted store fails loudly rather than handing
back content that merely looks plausible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def default_recovery_store_path() -> str:
    """Where recovery material lives unless the caller says otherwise."""
    import os

    base = os.environ.get("ENTROLY_DIR")
    if base:
        return str(Path(base) / "recovery.json")
    from .config import _project_checkpoint_dir

    return str(_project_checkpoint_dir() / "recovery.json")


def _colours() -> Any:
    from .cli import C

    return C


def cmd_compress(args) -> int:
    """Compress a file with the content codecs and print an auditable receipt."""
    from .codec import RecoveryStore
    from .codecs_builtin import default_registry

    C = _colours()
    source = Path(args.path)
    if not source.exists():
        print(f"{C.RED}  No such file: {source}{C.RESET}")
        return 1
    # newline="" so line endings survive verbatim. Path.read_text() translates
    # CRLF to LF on Windows, which made the stored "original" 3,606 bytes for a
    # 3,854-byte file -- recovery then returned something that was faithful to
    # what was compressed but NOT to the file on disk. The recovery contract is
    # about the caller's bytes, not Python's view of them.
    with source.open("r", encoding="utf-8", errors="surrogateescape", newline="") as fh:
        text = fh.read()

    store_path = args.store_path or default_recovery_store_path()
    store = RecoveryStore(store_path)
    reps = default_registry(store).representations(
        text, source_id=str(source), query=getattr(args, "query", "") or ""
    )

    chosen = None
    if reps:
        # Never offer a lossy form that has no way back.
        usable = [r for r in reps if r.recovery is not None or r.text == text]
        chosen = min(usable or reps, key=lambda r: r.token_cost)

    before = max(1, len(text) // 4)
    after = chosen.token_cost if chosen else before
    receipt = {
        "source": str(source),
        "codec": chosen.codec if chosen else "none",
        "representation": chosen.representation_id if chosen else None,
        "tokens_before": before,
        "tokens_after": after,
        "reduction_pct": round(100.0 * (1 - after / before), 1) if before else 0.0,
        "protected_evidence": list(chosen.protected_evidence) if chosen else [],
        "unverified_protected": (
            list(chosen.verify_protected_evidence()) if chosen else []
        ),
        "recovery_digest": (
            chosen.recovery.digest if chosen and chosen.recovery else None
        ),
        "recovery_store": store_path,
        "source_sha256": chosen.source_sha256 if chosen else None,
    }

    if getattr(args, "out_path", None) and chosen:
        with Path(args.out_path).open(
            "w", encoding="utf-8", errors="surrogateescape", newline=""
        ) as fh:
            fh.write(chosen.text)
        receipt["written_to"] = args.out_path

    if getattr(args, "json_output", False):
        print(json.dumps(receipt, indent=2))
        return 0

    if chosen is None or chosen.text == text:
        print()
        print(f"{C.YELLOW}  No codec claimed {source.name}, or it was already minimal.{C.RESET}")
        print(f"  {C.GRAY}Left unchanged -- {before:,} tokens.{C.RESET}")
        print()
        return 0

    print()
    print(f"{C.BOLD}  {source.name}{C.RESET}  {C.GRAY}via {chosen.codec} codec{C.RESET}")
    print(
        f"  {before:,} -> {C.GREEN}{after:,} tokens{C.RESET} "
        f"({receipt['reduction_pct']}% smaller)"
    )
    if chosen.protected_evidence:
        shown = [s[:40] for s in list(chosen.protected_evidence)[:5]]
        extra = len(chosen.protected_evidence) - len(shown)
        suffix = f" (+{extra} more)" if extra > 0 else ""
        print()
        print(f"  {C.BOLD}Kept:{C.RESET} " + ", ".join(shown) + suffix)
    if receipt["unverified_protected"]:
        print(f"  {C.RED}Claimed but absent: {receipt['unverified_protected']}{C.RESET}")
    print()
    if receipt["recovery_digest"]:
        print(f"  {C.BOLD}Recover the original:{C.RESET}")
        print(f"    {C.CYAN}entroly recover {receipt['recovery_digest']}{C.RESET}")
    else:
        print(f"  {C.YELLOW}No recovery reference -- nothing was dropped.{C.RESET}")
    print()
    return 0


def cmd_recover(args) -> int:
    """Return the exact original bytes for a recovery digest."""
    from .codec import RecoveryStore

    C = _colours()
    store_path = args.store_path or default_recovery_store_path()
    store = RecoveryStore(store_path)

    reference = store.reference_for(args.digest)
    if reference is None:
        print(f"{C.RED}  No recovery entry for {args.digest}{C.RESET}")
        print(f"  {C.GRAY}store: {store_path}{C.RESET}")
        return 1

    try:
        original = store.recover(reference)
    except (KeyError, ValueError) as exc:
        print(f"{C.RED}  Recovery failed: {exc}{C.RESET}")
        return 1

    if getattr(args, "out_path", None):
        # surrogateescape mirrors the read: bytes that were not valid UTF-8
        # go back out as the same bytes.
        Path(args.out_path).write_bytes(
            original.encode("utf-8", "surrogateescape")
        )
        print(
            f"{C.GREEN}  Recovered {reference.byte_length:,} bytes -> "
            f"{args.out_path}{C.RESET}"
        )
        # The store already records what it holds, and that is not the same
        # thing for every codec: `json` keeps the complete original, while
        # `code` keeps only the bodies elided from a skeleton. Printing the
        # note is the difference between a user knowing they hold a fragment
        # and believing they have their file back -- recovered code bodies are
        # not even syntactically valid alone, because the imports and function
        # signatures stayed in the compressed skeleton.
        if reference.note:
            # The note alone, without a "combine with the compressed form"
            # hint: that instruction is true for `code`, where the recovery is
            # the elided bodies, and false for `json`, where the recovery is
            # already the whole file. Printing it unconditionally would trade
            # one misleading message for another.
            detail = reference.note
            if reference.item_count:
                # The label, not a bare "item(s)": the number means something
                # different per codec, and printing it unlabelled next to
                # "complete original ..." described a 60-record file as 59.
                detail += f" ({reference.item_count:,} {reference.item_label})"
            print(f"  {C.GRAY}{detail}{C.RESET}")
    else:
        sys.stdout.buffer.write(original.encode("utf-8", "surrogateescape"))
        # stdout carries the bytes and has to stay pipeable, so the note goes
        # to stderr rather than corrupting a redirect.
        if reference.note:
            print(reference.note, file=sys.stderr)
    return 0
