from pathlib import Path

import pytest

from docintel.core.interfaces import Chunker, Loader
from docintel.core.models import ChunkedDocument, ProcessedDocument, RawDocument, SourceType
from docintel.ingestion.loaders.base import LoaderPlugin, LoaderRegistry, register_loader


def _make_raw_document(content: str = "hello world") -> RawDocument:
    return RawDocument(
        content=content,
        source_uri="test.txt",
        source_type=SourceType.TXT,
        knowledge_base_id="kb-test",
    )


# ---------------------------------------------------------------------------
# Structural typing
# ---------------------------------------------------------------------------


class FakeChunker:
    async def chunk(self, document: ProcessedDocument) -> ChunkedDocument:
        return ChunkedDocument(processed_document=document, chunks=[])


def test_fake_chunker_satisfies_protocol():
    chunker: Chunker = FakeChunker()
    assert isinstance(chunker, Chunker)


class NotAChunker:
    def frobnicate(self) -> None:
        pass


def test_non_conforming_class_fails_protocol_check():
    assert not isinstance(NotAChunker(), Chunker)


# ---------------------------------------------------------------------------
# Loader plugin registry
# ---------------------------------------------------------------------------


class DummyTxtLoader(LoaderPlugin):
    supported_extensions = (".txt",)

    async def load(self, source, knowledge_base_id: str) -> list[RawDocument]:
        return [_make_raw_document()]


class DummyWebLoader(LoaderPlugin):
    handles_urls = True

    async def load(self, source, knowledge_base_id: str) -> list[RawDocument]:
        return [_make_raw_document(content="web content")]


def test_dummy_loader_satisfies_loader_protocol():
    assert isinstance(DummyTxtLoader(), Loader)


def test_registry_resolves_by_extension():
    reg = LoaderRegistry()
    reg.register(DummyTxtLoader())
    loader = reg.resolve(Path("report.txt"))
    assert isinstance(loader, DummyTxtLoader)


def test_registry_resolves_url_loader():
    reg = LoaderRegistry()
    reg.register(DummyWebLoader())
    loader = reg.resolve("https://example.com/page")
    assert isinstance(loader, DummyWebLoader)


def test_registry_raises_when_no_loader_matches():
    reg = LoaderRegistry()
    reg.register(DummyTxtLoader())
    with pytest.raises(ValueError, match="No registered loader"):
        reg.resolve(Path("report.exotic"))


@pytest.mark.asyncio
async def test_registered_loader_can_load_document():
    reg = LoaderRegistry()
    reg.register(DummyTxtLoader())
    loader = reg.resolve(Path("x.txt"))
    docs = await loader.load(Path("x.txt"), knowledge_base_id="kb-test")
    assert docs[0].content == "hello world"
    assert docs[0].knowledge_base_id == "kb-test"


def test_register_loader_decorator_adds_to_global_registry():
    from docintel.ingestion.loaders.base import registry as global_registry

    before = len(global_registry.all())

    @register_loader
    class TempJsonLoader(LoaderPlugin):
        supported_extensions = (".json_test_fixture",)

        async def load(self, source, knowledge_base_id: str) -> list[RawDocument]:
            return [_make_raw_document()]

    after = len(global_registry.all())
    assert after == before + 1
    resolved = global_registry.resolve(Path("data.json_test_fixture"))
    assert isinstance(resolved, TempJsonLoader)
