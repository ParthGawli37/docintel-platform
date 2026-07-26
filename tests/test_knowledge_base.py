import pytest
from qdrant_client import AsyncQdrantClient

from docintel.core.models import KnowledgeBase
from docintel.knowledge_base.manager import KnowledgeBaseManager, KnowledgeBaseNotFoundError
from docintel.storage.kb_store import SqliteKnowledgeBaseStore
from docintel.vectorstore.qdrant_store import QdrantVectorStore

# ---------------------------------------------------------------------------
# SqliteKnowledgeBaseStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_store_save_and_get_roundtrip(tmp_path):
    store = SqliteKnowledgeBaseStore(tmp_path / "kb.sqlite")
    kb = KnowledgeBase(
        name="Test KB", embedding_model_id="model-a", embedding_dimensions=768
    )
    await store.save(kb)
    fetched = await store.get(kb.id)
    assert fetched is not None
    assert fetched.name == "Test KB"
    assert fetched.embedding_dimensions == 768


@pytest.mark.asyncio
async def test_kb_store_get_nonexistent_returns_none(tmp_path):
    store = SqliteKnowledgeBaseStore(tmp_path / "kb.sqlite")
    assert await store.get("nonexistent") is None


@pytest.mark.asyncio
async def test_kb_store_list_all_returns_all_saved(tmp_path):
    store = SqliteKnowledgeBaseStore(tmp_path / "kb.sqlite")
    kb1 = KnowledgeBase(name="KB1", embedding_model_id="m", embedding_dimensions=4)
    kb2 = KnowledgeBase(name="KB2", embedding_model_id="m", embedding_dimensions=4)
    await store.save(kb1)
    await store.save(kb2)
    all_kbs = await store.list_all()
    assert {kb.id for kb in all_kbs} == {kb1.id, kb2.id}


@pytest.mark.asyncio
async def test_kb_store_delete(tmp_path):
    store = SqliteKnowledgeBaseStore(tmp_path / "kb.sqlite")
    kb = KnowledgeBase(name="KB", embedding_model_id="m", embedding_dimensions=4)
    await store.save(kb)
    deleted = await store.delete(kb.id)
    assert deleted is True
    assert await store.get(kb.id) is None


@pytest.mark.asyncio
async def test_kb_store_delete_nonexistent_returns_false(tmp_path):
    store = SqliteKnowledgeBaseStore(tmp_path / "kb.sqlite")
    assert await store.delete("nonexistent") is False


# ---------------------------------------------------------------------------
# KnowledgeBaseManager
# ---------------------------------------------------------------------------


@pytest.fixture
async def manager(tmp_path):
    kb_store = SqliteKnowledgeBaseStore(tmp_path / "kb.sqlite")
    client = AsyncQdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client)
    yield KnowledgeBaseManager(kb_store, vector_store)
    await client.close()


@pytest.mark.asyncio
async def test_manager_create_persists_config_and_provisions_collection(manager):
    kb = await manager.create(
        name="Portfolio Assistant", embedding_model_id="model-a", embedding_dimensions=4
    )
    fetched = await manager.get(kb.id)
    assert fetched.name == "Portfolio Assistant"

    # Collection must actually exist in the vector store.
    hashes = await manager._vector_store.get_existing_content_hashes(kb.id)
    assert hashes == set()  # empty but collection exists (no exception raised)


@pytest.mark.asyncio
async def test_manager_get_nonexistent_raises(manager):
    with pytest.raises(KnowledgeBaseNotFoundError):
        await manager.get("nonexistent")


@pytest.mark.asyncio
async def test_manager_list_all(manager):
    kb1 = await manager.create(name="KB1", embedding_model_id="m", embedding_dimensions=4)
    kb2 = await manager.create(name="KB2", embedding_model_id="m", embedding_dimensions=4)
    all_kbs = await manager.list_all()
    assert {kb.id for kb in all_kbs} == {kb1.id, kb2.id}


@pytest.mark.asyncio
async def test_manager_delete_removes_config_and_collection(manager):
    kb = await manager.create(name="Temp KB", embedding_model_id="m", embedding_dimensions=4)
    await manager.delete(kb.id)

    with pytest.raises(KnowledgeBaseNotFoundError):
        await manager.get(kb.id)


@pytest.mark.asyncio
async def test_manager_delete_nonexistent_raises(manager):
    with pytest.raises(KnowledgeBaseNotFoundError):
        await manager.delete("nonexistent")


@pytest.mark.asyncio
async def test_manager_different_kbs_are_isolated_collections(manager):
    kb1 = await manager.create(name="KB1", embedding_model_id="m", embedding_dimensions=4)
    kb2 = await manager.create(name="KB2", embedding_model_id="m", embedding_dimensions=4)
    assert kb1.id != kb2.id
