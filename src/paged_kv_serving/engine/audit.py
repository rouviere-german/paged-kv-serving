"""Hash-chained audit log for privileged operations.

Only a handful of actions can change what the server computes: swapping the weights and flushing
the prefix cache. Both are invisible in ordinary request logs -- a flushed cache just looks like a
bad day for the hit rate, and swapped weights just look like a different model -- so both are
recorded in a log that cannot be edited quietly.

Each entry commits to the one before it:

    digest_i = SHA-256(canonical_json({index, timestamp, actor, action, detail, previous}))

Changing, reordering or deleting any entry changes every digest after it, so verification is a
single forward pass. Two attacks remain, and each has its own answer:

* **Rebuilding the chain.** Plain SHA-256 is unkeyed: whoever can write the file can edit an entry
  and recompute every digest after it. With a key (``APP_AUDIT_HMAC_KEY``) each digest is an
  HMAC-SHA256 instead, so rewriting history needs a secret that lives in the server's environment,
  not next to the file.
* **Truncating the tail.** A self-contained chain cannot see its own missing tail, so
  :meth:`AuditLog.verify` takes an expected head digest. Every append publishes the new head as an
  ``audit_appended`` log event, so the log pipeline -- which the audit file's writer cannot
  rewrite -- holds the anchor; ``paged-kv-serving audit verify --expect-head`` checks against it.

Details are redacted before they are hashed, not after, so the log can never become the place a
secret ends up in plain text.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paged_kv_serving.core.canonical import canonical_json, sha256_hex
from paged_kv_serving.core.fileio import file_lock
from paged_kv_serving.core.logging import get_logger
from paged_kv_serving.core.redaction import redact_value

__all__ = [
    "GENESIS",
    "KEY_ENV",
    "AuditChainError",
    "AuditEntry",
    "AuditLog",
    "key_from_environment",
    "verify_file",
]

GENESIS = "0" * 64
#: Environment variable holding the HMAC key, read by the server settings and by the CLI.
KEY_ENV = "APP_AUDIT_HMAC_KEY"
_log = get_logger("paged_kv_serving.audit")
_MAX_ACTION = 64
_MAX_ACTOR = 128


class AuditChainError(ValueError):
    """The audit chain is broken: a digest, a link or the expected head does not match."""


@dataclass(frozen=True, slots=True)
class AuditEntry:
    index: int
    timestamp: str
    actor: str
    action: str
    detail: dict[str, Any]
    previous: str
    digest: str

    def payload(self) -> dict[str, Any]:
        """The part of the entry the digest commits to."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "detail": self.detail,
            "previous": self.previous,
        }

    def expected_digest(self, key: bytes | None = None) -> str:
        """SHA-256 of the canonical payload, or its HMAC-SHA256 under ``key``."""
        text = canonical_json(self.payload())
        if key is None:
            return sha256_hex(text)
        return hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()

    def to_json(self) -> str:
        return json.dumps({**self.payload(), "digest": self.digest}, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> AuditEntry:
        data = json.loads(line)
        missing = {"index", "timestamp", "actor", "action", "detail", "previous", "digest"} - set(
            data
        )
        if missing:
            raise AuditChainError(f"audit entry is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(data["detail"], dict):
            raise AuditChainError("audit entry detail must be an object")
        return cls(
            index=int(data["index"]),
            timestamp=str(data["timestamp"]),
            actor=str(data["actor"]),
            action=str(data["action"]),
            detail=dict(data["detail"]),
            previous=str(data["previous"]),
            digest=str(data["digest"]),
        )


def _verify_entries(
    entries: list[AuditEntry], *, expect_head: str | None, key: bytes | None = None
) -> str:
    previous = GENESIS
    for position, entry in enumerate(entries):
        if entry.index != position:
            raise AuditChainError(f"entry {position} claims index {entry.index}")
        if entry.previous != previous:
            raise AuditChainError(f"entry {position} does not link to its predecessor")
        if not hmac.compare_digest(entry.expected_digest(key), entry.digest):
            raise AuditChainError(f"entry {position} has been modified")
        previous = entry.digest
    if expect_head is not None and previous != expect_head:
        raise AuditChainError("the chain head does not match the expected value")
    return previous


class AuditLog:
    """Append-only JSONL chain. Every append re-reads and verifies the file under a lock.

    ``key`` switches the digests to HMAC-SHA256; a keyed log verifies only with the same key.
    """

    __slots__ = ("_key", "clock", "path")

    def __init__(self, path: Path, *, clock: Any = None, key: bytes | None = None) -> None:
        if key is not None and not key:
            raise ValueError("an audit key must not be empty")
        self.path = path
        self.clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._key = key

    @property
    def keyed(self) -> bool:
        return self._key is not None

    def entries(self) -> list[AuditEntry]:
        if not self.path.is_file():
            return []
        lines = [
            line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        return [AuditEntry.from_json(line) for line in lines]

    def verify(self, *, expect_head: str | None = None) -> str:
        """Verify the whole chain and return its head digest (GENESIS when empty)."""
        return _verify_entries(self.entries(), expect_head=expect_head, key=self._key)

    @property
    def head(self) -> str:
        return self.verify()

    def append(
        self, actor: str, action: str, detail: Mapping[str, Any] | None = None
    ) -> AuditEntry:
        """Verify the chain, then append one entry. Raises if the file was tampered with."""
        if not action or len(action) > _MAX_ACTION:
            raise ValueError("action must be a short non-empty string")
        if not actor or len(actor) > _MAX_ACTOR:
            raise ValueError("actor must be a short non-empty string")
        safe_detail = redact_value(dict(detail or {}))
        if not isinstance(safe_detail, dict):  # pragma: no cover - redact_value keeps mappings
            raise TypeError("detail must be a mapping")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path):
            existing = self.entries()
            previous = _verify_entries(existing, expect_head=None, key=self._key)
            entry = AuditEntry(
                index=len(existing),
                timestamp=self.clock().isoformat(),
                actor=actor,
                action=action,
                detail=safe_detail,
                previous=previous,
                digest="",
            )
            digest = entry.expected_digest(self._key)
            entry = AuditEntry(**{**entry.payload(), "digest": digest})
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(entry.to_json() + "\n")
                handle.flush()
        # The anchor: the new head goes to the log stream, which the audit file's writer cannot
        # rewrite. Details stay out of it -- they are already in the chain.
        _log.info(
            "audit_appended",
            index=entry.index,
            action=entry.action,
            actor=entry.actor,
            head=entry.digest,
            keyed=self.keyed,
        )
        return entry


def verify_file(path: Path, *, expect_head: str | None = None, key: bytes | None = None) -> str:
    """Verify the chain in ``path``. Returns the head digest."""
    return AuditLog(path, key=key).verify(expect_head=expect_head)


def key_from_environment() -> bytes | None:
    """The HMAC key from ``APP_AUDIT_HMAC_KEY``, or None when it is unset or empty."""
    value = os.environ.get(KEY_ENV, "")
    return value.encode("utf-8") if value else None


def main(argv: list[str] | None = None) -> int:
    """``paged-kv-serving audit verify <path> [--expect-head HEX]``.

    A keyed log is verified with the key in ``APP_AUDIT_HMAC_KEY``.
    """
    import argparse  # noqa: PLC0415 - CLI-only import

    parser = argparse.ArgumentParser(prog="audit", description="verify a hash-chained audit log")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)
    args = parser.parse_args(argv)
    try:
        head = verify_file(args.path, expect_head=args.expect_head, key=key_from_environment())
    except (AuditChainError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))  # noqa: T201 - CLI output
        return 1
    print(json.dumps({"ok": True, "head": head}))  # noqa: T201 - CLI output
    return 0
