"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog
from fastapi import FastAPI
from hypothesis import HealthCheck, settings

from paged_kv_serving.engine.config import EngineConfig, ModelConfig
from paged_kv_serving.engine.model import LanguageModel, random_weights
from paged_kv_serving.engine.threads import limit_blas_threads
from paged_kv_serving.engine.weights import Checkpoint, build_random_checkpoint

# The same single-threaded BLAS the CLI and the benchmarks use (docs/adr/0006): the matrices here
# are tiny, and thread wake-ups would otherwise dominate every forward pass in the suite.
limit_blas_threads(1)

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

#: Small enough that a few hundred forward passes finish in seconds, large enough that every
#: shape in the engine (several layers, several heads, an even head_dim for RoPE) is exercised.
TINY_MODEL_CONFIG = ModelConfig(
    vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=192
)
TINY_ENGINE_CONFIG = EngineConfig(
    block_size=8,
    num_blocks=64,
    max_batched_tokens=32,
    max_running_seqs=6,
    max_prompt_tokens=64,
    max_new_tokens=16,
    max_context_tokens=128,
)


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tiny_model() -> LanguageModel:
    """A two-layer model held in memory: no files, no manifest, fast enough for property tests."""
    return LanguageModel(TINY_MODEL_CONFIG, random_weights(TINY_MODEL_CONFIG, seed=11))


@pytest.fixture
def engine_config() -> EngineConfig:
    return TINY_ENGINE_CONFIG


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Checkpoint:
    """A written-and-verified checkpoint, shared by the tests that need one on disk."""
    directory = tmp_path_factory.mktemp("checkpoint")
    return build_random_checkpoint(
        directory / "model",
        seed=5,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        max_seq_len=192,
        tokenizer_vocab_size=320,
    )


async def lifespan_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An in-process client that also runs the app's ASGI lifespan.

    httpx's ASGI transport speaks HTTP only, so without this the lifespan never runs and the
    engine loop is never started -- which is exactly the bug a test of the real app should not
    have to work around silently.
    """
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
            yield client
