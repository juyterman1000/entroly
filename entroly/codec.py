"""Public codec contract with strict recovery-reference verification."""

from __future__ import annotations

from . import codec_legacy as _legacy

for _name, _value in vars(_legacy).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


def _verify_recovery_reference(self, recovered: bytes | str) -> bool:
    raw = recovered.encode("utf-8") if isinstance(recovered, str) else recovered
    return len(raw) == self.byte_length and content_digest(raw) == self.digest


_legacy.RecoveryReference.verify = _verify_recovery_reference
RecoveryReference = _legacy.RecoveryReference
