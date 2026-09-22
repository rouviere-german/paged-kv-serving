"""The offline demo and the command line entry points."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from paged_kv_serving import cli
from paged_kv_serving.core import tracing
from paged_kv_serving.core.security import hash_api_key
from paged_kv_serving.engine.audit import AuditLog
from paged_kv_serving.engine.demo import render_report, run_serving_demo

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo_summary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return run_serving_demo(tmp_path_factory.mktemp("demo"))


def test_the_demo_checks_equivalence_through_the_real_scheduler(
    demo_summary: dict[str, Any],
) -> None:
    equivalence = demo_summary["equivalence"]
    assert equivalence["single_prompt_max_abs_logit_gap"] < 1e-4
    assert equivalence["engine_max_abs_logit_gap"] < 1e-4
    assert equivalence["engine_logit_rows_checked"] > 100
    assert equivalence["engine_preemptions"] >= 1, "the cramped pool must force a preemption"


def test_the_demo_shows_prefix_reuse_admission_and_cancellation(
    demo_summary: dict[str, Any],
) -> None:
    cache = demo_summary["prefix_cache"]
    assert cache["hit_ratio_cache_off"] == 0.0
    assert cache["hit_ratio_cache_on"] > 0.5
    assert cache["prefill_tokens_saved_fraction"] > 0.5

    admission = demo_summary["admission"]
    assert admission["completed"] > 0, "admission must not refuse everything"
    assert admission["rejected"] > 0, "a burst this size must be partly refused"
    assert admission["completed"] + admission["rejected"] == admission["submitted"]

    cancellation = demo_summary["cancellation"]
    assert cancellation["blocks_held_before_abort"] > 0
    assert cancellation["blocks_held_after_abort"] == 0
    assert cancellation["free_blocks_after_cache_flush"] == cancellation["pool_size"]


def test_the_demo_serves_over_a_real_socket(demo_summary: dict[str, Any]) -> None:
    api = demo_summary["api"]
    assert api["healthz"] == 200
    assert api["readyz"] == "ready"
    assert api["unauthenticated_status"] == 401
    assert api["batch_key_asking_for_interactive_status"] == 403
    assert api["stream_status"] == 200
    assert api["streamed_token_events"] == 8
    assert api["hangup_aborted_requests"] == 1, "hanging up mid-stream must abort the request"
    assert api["active_requests_after_hangup"] == 0
    assert api["sequence_blocks_after_hangup"] == 0
    assert api["admin_without_scope_status"] == 403
    assert api["admin_flush_status"] == 200
    assert api["audit_verified"] is True
    assert "engine_ttft_seconds" in api["metrics_exported"]
    burst = api["open_loop_burst"]["classes"]
    assert set(burst) == {"batch", "interactive"}
    assert all(row["completed"] > 0 for row in burst.values())


def test_the_demo_compares_both_schedulers_on_one_schedule(demo_summary: dict[str, Any]) -> None:
    continuous, static = demo_summary["batching"]
    assert continuous["label"] == "continuous"
    assert static["label"] == "static"
    assert continuous["submitted"] == static["submitted"]
    assert static["padding_tokens"] > 0
    report = render_report(demo_summary)
    assert "continuous vs static batching" in report
    assert "goodput_per_second" in report


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_prints_a_report_and_writes_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = {"called": False}

    def fake_demo(workdir: Path) -> dict[str, Any]:
        fake["called"] = True
        return {"workdir": str(workdir)}

    monkeypatch.setattr("paged_kv_serving.engine.demo.run_serving_demo", fake_demo)
    monkeypatch.setattr("paged_kv_serving.engine.demo.render_report", lambda summary: "REPORT")
    assert cli.main(["demo", "--workdir", str(tmp_path)]) == 0
    assert fake["called"]
    assert "REPORT" in capsys.readouterr().out
    written = json.loads((tmp_path / "demo-summary.json").read_text(encoding="utf-8"))
    assert written == {"workdir": str(tmp_path)}


def test_cli_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


def test_cli_build_checkpoint_prints_the_pinnable_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["build-checkpoint", "--out", str(tmp_path / "ckpt"), "--seed", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["manifest_sha256"]) == 64
    assert (tmp_path / "ckpt" / "weights.safetensors").is_file()
    assert payload["parameters"] > 100_000


def test_cli_audit_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("admin", "prefix_cache.flush", {"released_blocks": 3})
    assert cli.main(["audit", "verify", str(path), "--expect-head", log.head]) == 0
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


def test_cli_loadgen_requires_a_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LOADGEN_INTERACTIVE_API_KEY", raising=False)
    assert cli.main(["loadgen", "--count", "1"]) == 2
    assert "LOADGEN_INTERACTIVE_API_KEY" in capsys.readouterr().err


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_verifies_the_checkpoint_and_configures_uvicorn_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    from paged_kv_serving.core.config import get_settings
    from paged_kv_serving.engine.weights import build_random_checkpoint

    checkpoint = build_random_checkpoint(
        tmp_path / "ckpt", n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=64,
        tokenizer_vocab_size=300,
    )  # fmt: skip
    monkeypatch.setenv("APP_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("APP_CHECKPOINT_MANIFEST_SHA256", checkpoint.manifest.sha256())
    monkeypatch.setenv("APP_ENGINE_MAX_CONTEXT_TOKENS", "64")
    monkeypatch.setenv("APP_ENGINE_MAX_PROMPT_TOKENS", "32")
    monkeypatch.setenv("APP_ENGINE_MAX_NEW_TOKENS", "16")
    monkeypatch.setenv("APP_ENGINE_PROFILE_ON_START", "false")
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    captured: dict[str, Any] = {}

    def fake_run(self: uvicorn.Server) -> None:
        captured["config"] = self.config

    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    get_settings.cache_clear()
    try:
        assert cli.main(["serve", "--port", "8123"]) == 0
    finally:
        get_settings.cache_clear()
    config = captured["config"]
    assert config.host == "127.0.0.1"
    assert config.port == 8123
    assert config.server_header is False
    assert config.access_log is False


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_refuses_a_checkpoint_whose_manifest_is_not_the_pinned_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from paged_kv_serving.core.config import get_settings
    from paged_kv_serving.engine.weights import CheckpointIntegrityError, build_random_checkpoint

    build_random_checkpoint(
        tmp_path / "ckpt", n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=64,
        tokenizer_vocab_size=300,
    )  # fmt: skip
    monkeypatch.setenv("APP_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("APP_CHECKPOINT_MANIFEST_SHA256", "f" * 64)
    get_settings.cache_clear()
    try:
        with pytest.raises(CheckpointIntegrityError, match="pinned"):
            cli.main(["serve"])
    finally:
        get_settings.cache_clear()


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["paged_kv_serving", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("paged_kv_serving", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out
