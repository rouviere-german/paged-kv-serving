from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from paged_kv_serving.engine.audit import (
    GENESIS,
    AuditChainError,
    AuditEntry,
    AuditLog,
    main,
    verify_file,
)

pytestmark = pytest.mark.unit


def log(tmp_path: Path) -> AuditLog:
    stamps = iter(range(1_000))
    return AuditLog(
        tmp_path / "audit.jsonl",
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC).replace(second=next(stamps) % 60),
    )


def test_an_empty_log_has_the_genesis_head(tmp_path: Path) -> None:
    assert log(tmp_path).head == GENESIS
    assert log(tmp_path).entries() == []


def test_entries_chain_and_verify(tmp_path: Path) -> None:
    audit = log(tmp_path)
    first = audit.append("key-a", "prefix_cache.flush", {"released_blocks": 3})
    second = audit.append("key-a", "weights.swap", {"name": "v2"})
    assert first.index == 0
    assert first.previous == GENESIS
    assert second.previous == first.digest
    assert audit.verify(expect_head=second.digest) == second.digest
    assert [entry.action for entry in audit.entries()] == [
        "prefix_cache.flush",
        "weights.swap",
    ]


def test_any_single_byte_change_is_detected(tmp_path: Path) -> None:
    audit = log(tmp_path)
    audit.append("key-a", "prefix_cache.flush", {"released_blocks": 3})
    audit.append("key-a", "weights.swap", {"name": "v2"})
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["detail"]["released_blocks"] = 4
    lines[0] = json.dumps(payload, sort_keys=True)
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError, match="entry 0 has been modified"):
        audit.verify()


def test_reordering_or_deleting_an_entry_is_detected(tmp_path: Path) -> None:
    audit = log(tmp_path)
    for index in range(3):
        audit.append("key-a", "prefix_cache.flush", {"n": index})
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    audit.path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError, match="claims index"):
        audit.verify()
    audit.path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError):
        audit.verify()


def test_tail_truncation_needs_an_anchored_head(tmp_path: Path) -> None:
    """A chain cannot see its own missing tail, which is why the head is published separately."""
    audit = log(tmp_path)
    audit.append("key-a", "prefix_cache.flush", {"n": 0})
    head = audit.append("key-a", "prefix_cache.flush", {"n": 1}).digest
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    audit.path.write_text(lines[0] + "\n", encoding="utf-8")
    assert audit.verify() != head  # the truncated chain verifies on its own
    with pytest.raises(AuditChainError, match="head does not match"):
        audit.verify(expect_head=head)


def test_appending_to_a_tampered_log_is_refused(tmp_path: Path) -> None:
    audit = log(tmp_path)
    audit.append("key-a", "prefix_cache.flush", {"n": 0})
    audit.path.write_text(
        audit.path.read_text(encoding="utf-8").replace('"n": 0', '"n": 9'), encoding="utf-8"
    )
    with pytest.raises(AuditChainError):
        audit.append("key-a", "prefix_cache.flush", {"n": 1})


def test_details_are_redacted_before_they_are_hashed(tmp_path: Path) -> None:
    audit = log(tmp_path)
    entry = audit.append("key-a", "weights.swap", {"api_key": "sk-ant-abcdefghijklmnopqrstuv"})
    assert "sk-ant" not in audit.path.read_text(encoding="utf-8")
    assert entry.detail["api_key"] == "[REDACTED]"
    assert audit.verify() == entry.digest


def test_actor_and_action_are_bounded(tmp_path: Path) -> None:
    audit = log(tmp_path)
    with pytest.raises(ValueError, match="action must be"):
        audit.append("key-a", "")
    with pytest.raises(ValueError, match="action must be"):
        audit.append("key-a", "x" * 100)
    with pytest.raises(ValueError, match="actor must be"):
        audit.append("", "flush")


def test_malformed_lines_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"index": 0}\n', encoding="utf-8")
    with pytest.raises(AuditChainError, match="missing fields"):
        verify_file(path)
    path.write_text(
        json.dumps(
            {
                "index": 0,
                "timestamp": "t",
                "actor": "a",
                "action": "b",
                "detail": "not-an-object",
                "previous": GENESIS,
                "digest": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AuditChainError, match="detail must be an object"):
        verify_file(path)


def test_an_entry_knows_its_own_expected_digest() -> None:
    entry = AuditEntry(0, "t", "actor", "action", {}, GENESIS, "wrong")
    assert entry.expected_digest() != entry.digest
    assert json.loads(entry.to_json())["digest"] == "wrong"


def test_the_cli_reports_ok_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    audit = log(tmp_path)
    head = audit.append("key-a", "prefix_cache.flush", {}).digest
    assert main(["verify", str(audit.path), "--expect-head", head]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "head": head}
    assert main(["verify", str(audit.path), "--expect-head", GENESIS]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert main(["verify", str(tmp_path / "missing.jsonl")]) == 0  # an absent log is empty
    capsys.readouterr()
