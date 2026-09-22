from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from paged_kv_serving.core.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_defaults_fail_closed() -> None:
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == []
    assert settings.cors_allow_origins == []
    assert settings.enable_docs is False
    assert settings.metrics_require_auth is True
    assert settings.host == "127.0.0.1"
    assert settings.key_policies == []
    assert settings.engine_prefix_cache_scope == "per_key"
    assert settings.checkpoint_manifest_sha256 is None


def test_environment_variables_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_API_KEY_SHA256", f"{'a' * 64}, {'b' * 64}")
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://app.example.com,http://localhost:3000")
    monkeypatch.setenv("APP_ENGINE_BLAS_THREADS", "")
    monkeypatch.setenv("APP_CHECKPOINT_MANIFEST_SHA256", "")
    policy = f"{'c' * 64}:interactive|batch:6000:4:512:128:1"
    monkeypatch.setenv("APP_KEY_POLICIES", policy)
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == ["a" * 64, "b" * 64]
    assert settings.cors_allow_origins == ["https://app.example.com", "http://localhost:3000"]
    assert settings.engine_blas_threads is None
    assert settings.checkpoint_manifest_sha256 is None
    assert settings.key_policies == [policy]


def test_a_key_policy_may_end_with_a_parallel_sample_limit() -> None:
    entry = f"{'c' * 64}:batch:6000:8:512:128:0:4"
    assert Settings(_env_file=None, key_policies=[entry]).key_policies == [entry]


@pytest.mark.parametrize(
    "entry",
    [
        "short:interactive:1:1:1:1:0",
        f"{'c' * 64}:platinum:1:1:1:1:0",
        f"{'c' * 64}:interactive:1:1:1:1:2",
        f"{'c' * 64}:interactive:-1:1:1:1:0",
        f"{'c' * 64}:interactive:1:1:1:1:0:0",
        f"{'c' * 64}:interactive:1:1:1:1:0:99",
    ],
)
def test_malformed_key_policies_are_rejected(entry: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, key_policies=[entry])


@pytest.mark.parametrize("origins", [["*"], ["https://example.com/"], ["example.com"]])
def test_wildcard_or_malformed_origins_are_rejected(origins: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_allow_origins=origins)


@pytest.mark.parametrize("digest", ["A" * 64, "abc", "g" * 64])
def test_non_sha256_key_hashes_are_rejected(digest: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_key_sha256=[digest])


def test_env_example_documents_every_setting(repo_root: Path) -> None:
    lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")
    }
    expected = {f"APP_{name.upper()}" for name in Settings.model_fields}
    assert documented == expected


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_the_audit_key_is_secret_optional_and_long_enough() -> None:
    assert Settings(_env_file=None, audit_hmac_key="").audit_hmac_key is None
    settings = Settings(_env_file=None, audit_hmac_key="k" * 32)
    assert settings.audit_hmac_key is not None
    assert "k" * 32 not in repr(settings)
    with pytest.raises(ValueError, match="at least 32"):
        Settings(_env_file=None, audit_hmac_key="short")
