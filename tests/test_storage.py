
import pytest

from docintel.core.interfaces import EmbeddingCache, HashRegistry
from docintel.storage.cache_store import SqliteEmbeddingCache
from docintel.storage.hash_registry import SqliteHashRegistry
from docintel.storage.raw_store import LocalRawFileStore

# ---------------------------------------------------------------------------
# SqliteEmbeddingCache
# ---------------------------------------------------------------------------


def test_embedding_cache_satisfies_protocol(tmp_path):
    assert isinstance(SqliteEmbeddingCache(tmp_path / "cache.sqlite"), EmbeddingCache)


@pytest.mark.asyncio
async def test_embedding_cache_miss_returns_none(tmp_path):
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite")
    result = await cache.get("nonexistent-hash", "model-a")
    assert result is None


@pytest.mark.asyncio
async def test_embedding_cache_set_then_get_roundtrip(tmp_path):
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite")
    vector = [0.1, 0.2, 0.3]
    await cache.set("hash-1", "model-a", vector)
    result = await cache.get("hash-1", "model-a")
    assert result == vector


@pytest.mark.asyncio
async def test_embedding_cache_is_scoped_per_model(tmp_path):
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite")
    await cache.set("hash-1", "model-a", [1.0])
    result = await cache.get("hash-1", "model-b")
    assert result is None  # different model -- not a cache hit


@pytest.mark.asyncio
async def test_embedding_cache_persists_across_instances(tmp_path):
    db_path = tmp_path / "cache.sqlite"
    cache_a = SqliteEmbeddingCache(db_path)
    await cache_a.set("hash-1", "model-a", [9.9])
    cache_a.close()

    cache_b = SqliteEmbeddingCache(db_path)
    result = await cache_b.get("hash-1", "model-a")
    assert result == [9.9]


# ---------------------------------------------------------------------------
# SqliteHashRegistry
# ---------------------------------------------------------------------------


def test_hash_registry_satisfies_protocol(tmp_path):
    assert isinstance(SqliteHashRegistry(tmp_path / "hashes.sqlite"), HashRegistry)


@pytest.mark.asyncio
async def test_hash_registry_unknown_source_has_no_hash(tmp_path):
    registry = SqliteHashRegistry(tmp_path / "hashes.sqlite")
    result = await registry.get_hash("kb-1", "unknown.txt")
    assert result is None


@pytest.mark.asyncio
async def test_hash_registry_new_source_has_changed(tmp_path):
    registry = SqliteHashRegistry(tmp_path / "hashes.sqlite")
    assert await registry.has_changed("kb-1", "new.txt", "hash-abc") is True


@pytest.mark.asyncio
async def test_hash_registry_unchanged_source_detected(tmp_path):
    registry = SqliteHashRegistry(tmp_path / "hashes.sqlite")
    await registry.set_hash("kb-1", "doc.txt", "hash-abc")
    assert await registry.has_changed("kb-1", "doc.txt", "hash-abc") is False
    assert await registry.has_changed("kb-1", "doc.txt", "hash-different") is True


@pytest.mark.asyncio
async def test_hash_registry_scoped_per_knowledge_base(tmp_path):
    registry = SqliteHashRegistry(tmp_path / "hashes.sqlite")
    await registry.set_hash("kb-1", "doc.txt", "hash-abc")
    # Same source_uri, different KB -- should not be seen as unchanged.
    assert await registry.has_changed("kb-2", "doc.txt", "hash-abc") is True


# ---------------------------------------------------------------------------
# LocalRawFileStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_store_save_and_list(tmp_path):
    store = LocalRawFileStore(tmp_path / "raw")
    saved_path = await store.save_bytes("kb-1", "report.txt", b"hello world")
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"hello world"

    files = await store.list_files("kb-1")
    assert saved_path in files


@pytest.mark.asyncio
async def test_raw_store_scopes_files_by_knowledge_base(tmp_path):
    store = LocalRawFileStore(tmp_path / "raw")
    await store.save_bytes("kb-1", "a.txt", b"a")
    await store.save_bytes("kb-2", "b.txt", b"b")

    kb1_files = await store.list_files("kb-1")
    kb2_files = await store.list_files("kb-2")
    assert len(kb1_files) == 1
    assert len(kb2_files) == 1
    assert kb1_files[0].name == "a.txt"
    assert kb2_files[0].name == "b.txt"


@pytest.mark.asyncio
async def test_raw_store_delete(tmp_path):
    store = LocalRawFileStore(tmp_path / "raw")
    await store.save_bytes("kb-1", "temp.txt", b"data")
    deleted = await store.delete("kb-1", "temp.txt")
    assert deleted is True
    files = await store.list_files("kb-1")
    assert files == []


@pytest.mark.asyncio
async def test_raw_store_delete_nonexistent_returns_false(tmp_path):
    store = LocalRawFileStore(tmp_path / "raw")
    deleted = await store.delete("kb-1", "does-not-exist.txt")
    assert deleted is False


@pytest.mark.asyncio
async def test_raw_store_copy_from_path(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("copied content")
    store = LocalRawFileStore(tmp_path / "raw")
    dest = await store.copy_from_path("kb-1", source)
    assert dest.read_text() == "copied content"
    assert dest != source
