"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. API keys are configured as SHA-256 digests, so the process
never holds a raw key: it hashes what a caller presents and compares digests. Per-key entitlements
(SLO class, tokens per minute, concurrency, per-request ceilings) are configured the same way, as
digest-keyed entries, so the configuration file is not itself a list of credentials.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")
# <sha256>:<classes|separated>:<tokens_per_minute>:<max_concurrent>:<max_prompt>:<max_new>:<admin>
# optionally followed by :<max parallel samples, 1-8> (default 1)
_KEY_POLICY = re.compile(
    r"^[0-9a-f]{64}:(?:interactive|batch)(?:\|(?:interactive|batch))*:\d{1,9}:\d{1,4}:\d{1,6}"
    r":\d{1,6}:[01](?::[1-8])?$"
)


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(default="paged-kv-serving", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- model checkpoint ---
    checkpoint_dir: Path = Path("var/checkpoint")
    # Pinned digest of manifest.json. Set it and a swapped checkpoint is refused at start-up.
    checkpoint_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Root the admin weight-swap endpoint may load from; names are resolved inside it only.
    weights_root: Path = Path("var/weights")

    # --- engine ---
    engine_block_size: int = Field(default=16, ge=1, le=256)
    engine_num_blocks: int = Field(default=512, ge=1, le=1_000_000)
    engine_max_batched_tokens: int = Field(default=256, ge=1, le=65_536)
    engine_max_running_seqs: int = Field(default=32, ge=1, le=4_096)
    engine_max_prompt_tokens: int = Field(default=512, ge=1, le=1_000_000)
    engine_max_new_tokens: int = Field(default=128, ge=1, le=1_000_000)
    engine_max_context_tokens: int = Field(default=1_024, ge=2, le=1_000_000)
    engine_max_queued_sequences: int = Field(default=1_024, ge=1, le=1_000_000)
    engine_enable_prefix_cache: bool = True
    engine_prefix_cache_scope: Literal["per_key", "global"] = "per_key"
    engine_profile_on_start: bool = True
    # BLAS threads per process. The engine's matrices are small enough that thread wake-ups cost
    # more than they save (docs/adr/0006-single-threaded-blas.md); empty leaves BLAS untouched.
    engine_blas_threads: int | None = Field(default=1, ge=1, le=256)
    # Seconds the shutdown hook waits for in-flight requests before stopping the loop.
    drain_timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    # Hard ceiling on one request's wall time, streaming or not. The request is aborted and its
    # KV blocks freed when it runs out. A request with deadline_ms is aborted at its deadline
    # instead, and deadline_ms may not exceed this limit (400 deadline_too_long).
    request_timeout_seconds: float = Field(default=60.0, gt=0, le=3_600)

    # --- workload drift ---
    # The first drift_reference_size prompt lengths become the reference distribution; the PSI
    # of the most recent drift_window prompts against it is exported as engine_prompt_length_psi.
    drift_reference_size: int = Field(default=256, ge=2, le=1_000_000)
    drift_window: int = Field(default=256, ge=2, le=1_000_000)

    # --- per-key entitlements ---
    # Comma-separated entries, each
    # <api key sha256>:<interactive|batch, | separated>:<tokens/min>:<max concurrent sequences>
    # :<max prompt tokens>:<max new tokens>:<1 for admin, else 0>[:<max parallel samples n>]
    key_policies: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- storage ---
    data_dir: Path = Path("var")
    # Request telemetry: one contract-validated JSONL row per finished request (lengths, timings,
    # a prompt digest; never text) in <data_dir>/telemetry/requests.jsonl, rotated once at the cap.
    telemetry_enabled: bool = True
    telemetry_max_bytes: int = Field(default=8 * 1024 * 1024, ge=4_096, le=1_073_741_824)
    # HMAC key for the admin audit chain. Unset: plain SHA-256, which anybody who can write the
    # file can recompute. Set: rewriting the chain needs this secret as well.
    audit_hmac_key: SecretStr | None = Field(default=None, min_length=32)

    @field_validator(
        "engine_blas_threads", "checkpoint_manifest_sha256", "audit_hmac_key", mode="before"
    )
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("key_policies", mode="before")
    @classmethod
    def _parse_key_policies(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("key_policies")
    @classmethod
    def _check_key_policies(cls, value: list[str]) -> list[str]:
        for entry in value:
            if not _KEY_POLICY.fullmatch(entry):
                raise ValueError(
                    "key_policies entries must be "
                    "<sha256>:<classes>:<tokens/min>:<concurrent>:<prompt>:<new>:<admin>[:<n>]"
                )
        return value

    @field_validator("api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("api_key_sha256 entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
