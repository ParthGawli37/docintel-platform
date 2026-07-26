import pytest
from pydantic import ValidationError

from docintel.core.config import QdrantMode, Settings, get_settings, reset_settings_cache

REQUIRED_ENV = {
    "NVIDIA_API_KEY": "test-key",
    "NVIDIA_GENERATION_MODEL": "placeholder/generation-model",
    "NVIDIA_EMBEDDING_MODEL": "placeholder/embedding-model",
    "NVIDIA_EMBEDDING_DIMENSIONS": "1024",
}


def test_settings_load_with_required_env(monkeypatch, tmp_path):
    for k, v in REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("RAW_FILES_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HASH_REGISTRY_PATH", str(tmp_path / "cache" / "hash.sqlite"))

    reset_settings_cache()
    settings = get_settings()

    assert settings.nvidia_api_key == "test-key"
    assert settings.qdrant_mode is QdrantMode.LOCAL
    assert (tmp_path / "raw").exists()
    assert (tmp_path / "cache").exists()


def test_missing_required_field_raises(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    reset_settings_cache()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_cloud_qdrant_requires_api_key(monkeypatch, tmp_path):
    for k, v in REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("QDRANT_MODE", "cloud")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("RAW_FILES_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HASH_REGISTRY_PATH", str(tmp_path / "cache" / "hash.sqlite"))

    reset_settings_cache()
    with pytest.raises(ValueError, match="QDRANT_API_KEY"):
        get_settings()


def test_chunk_overlap_must_be_smaller_than_chunk_size(monkeypatch, tmp_path):
    for k, v in REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "100")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "100")
    monkeypatch.setenv("RAW_FILES_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HASH_REGISTRY_PATH", str(tmp_path / "cache" / "hash.sqlite"))

    reset_settings_cache()
    with pytest.raises(ValueError, match="CHUNK_OVERLAP_TOKENS"):
        get_settings()
