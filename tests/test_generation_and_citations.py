from dataclasses import dataclass, field

import pytest

from docintel.citations.builder import DefaultCitationBuilder
from docintel.core.interfaces import LLM, CitationBuilder
from docintel.core.models import Chunk, DocumentMetadata, SearchResult, SourceType
from docintel.generation.nvidia_nemotron import NvidiaNemotronLLM


def _search_result(content: str, source_uri: str = "x.txt", title: str | None = None, score: float = 0.8) -> SearchResult:
    chunk = Chunk(
        document_id="doc-1",
        content=content,
        chunk_index=0,
        metadata=DocumentMetadata(
            source_uri=source_uri,
            source_type=SourceType.TXT,
            content_hash="hash-1",
            knowledge_base_id="kb-1",
            title=title,
        ),
    )
    return SearchResult(chunk=chunk, score=score)


def test_citation_builder_satisfies_protocol():
    assert isinstance(DefaultCitationBuilder(), CitationBuilder)


def test_citation_builder_short_content_not_truncated():
    result = _search_result("short content")
    citations = DefaultCitationBuilder().build([result])
    assert citations[0].excerpt == "short content"
    assert not citations[0].excerpt.endswith("...")


def test_citation_builder_truncates_long_content():
    long_content = "word " * 200
    result = _search_result(long_content)
    builder = DefaultCitationBuilder(excerpt_length=50)
    citations = builder.build([result])
    assert citations[0].excerpt.endswith("...")
    assert len(citations[0].excerpt) <= 54


def test_citation_builder_prefers_rerank_score_when_present():
    result = _search_result("content", score=0.5)
    result_with_rerank = result.model_copy(update={"rerank_score": 0.99})
    citations = DefaultCitationBuilder().build([result_with_rerank])
    assert citations[0].score == 0.99


def test_citation_builder_falls_back_to_score_when_no_rerank():
    result = _search_result("content", score=0.5)
    citations = DefaultCitationBuilder().build([result])
    assert citations[0].score == 0.5


def test_citation_builder_preserves_source_fields():
    result = _search_result("content", source_uri="report.pdf", title="Q3 Report")
    citations = DefaultCitationBuilder().build([result])
    assert citations[0].source_uri == "report.pdf"
    assert citations[0].title == "Q3 Report"
    assert citations[0].document_id == "doc-1"


@dataclass
class _FakeDelta:
    content: str | None = None


@dataclass
class _FakeChoice:
    delta: _FakeDelta = field(default_factory=_FakeDelta)
    finish_reason: str | None = None


@dataclass
class _FakeChunk:
    choices: list[_FakeChoice]


class _FakeStream:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


class _FakeCompletions:
    def __init__(self, chunks: list[_FakeChunk], failures: int = 0) -> None:
        self._chunks = chunks
        self._failures = failures
        self.calls = 0
        self.last_call_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.calls += 1
        self.last_call_kwargs = kwargs
        if self.calls <= self._failures:
            raise RuntimeError("transient upstream failure")
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks: list[_FakeChunk], failures: int = 0) -> None:
        self.completions = _FakeCompletions(chunks, failures=failures)


class _FakeAsyncOpenAI:
    def __init__(self, chunks: list[_FakeChunk], failures: int = 0) -> None:
        self.chat = _FakeChat(chunks, failures=failures)


class _FakeSettings:
    nvidia_generation_model = "fake/nemotron-model"
    nvidia_api_key = "test-key"
    nvidia_api_base_url = "https://example.invalid/v1"


def _token_stream(*tokens: str) -> list[_FakeChunk]:
    chunks = [_FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=t))]) for t in tokens]
    chunks.append(_FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=None), finish_reason="stop")]))
    return chunks


@pytest.mark.asyncio
async def test_nemotron_llm_satisfies_protocol():
    client = _FakeAsyncOpenAI(_token_stream("hi"))
    llm = NvidiaNemotronLLM(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]
    assert isinstance(llm, LLM)


@pytest.mark.asyncio
async def test_nemotron_llm_retries_transient_request_failure():
    client = _FakeAsyncOpenAI(_token_stream("ok"), failures=1)
    llm = NvidiaNemotronLLM(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]

    texts = [chunk.text async for chunk in llm.stream_generate("q", context=[])]

    assert texts == ["ok", ""]
    assert client.chat.completions.calls == 2


@pytest.mark.asyncio
async def test_nemotron_llm_streams_text_in_order():
    client = _FakeAsyncOpenAI(_token_stream("Hello", " ", "world"))
    llm = NvidiaNemotronLLM(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]

    texts = []
    async for chunk in llm.stream_generate("query", context=[]):
        texts.append(chunk.text)

    assert texts == ["Hello", " ", "world", ""]


@pytest.mark.asyncio
async def test_nemotron_llm_attaches_citations_only_on_final_chunk():
    client = _FakeAsyncOpenAI(_token_stream("answer"))
    llm = NvidiaNemotronLLM(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]
    context = [_search_result("some context")]

    generation_chunks = [chunk async for chunk in llm.stream_generate("q", context=context)]

    non_final = generation_chunks[:-1]
    final = generation_chunks[-1]
    assert all(c.citations == [] for c in non_final)
    assert final.is_final is True
    assert len(final.citations) == 1
    assert final.citations[0].source_uri == "x.txt"


@pytest.mark.asyncio
async def test_nemotron_llm_sends_context_and_query_in_prompt():
    client = _FakeAsyncOpenAI(_token_stream("ok"))
    llm = NvidiaNemotronLLM(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]
    context = [_search_result("The sky is blue.", source_uri="facts.txt")]

    async for _ in llm.stream_generate("Why is the sky blue?", context=context):
        pass

    sent_messages = client.chat.completions.last_call_kwargs["messages"]
    user_message = next(m["content"] for m in sent_messages if m["role"] == "user")
    assert "The sky is blue." in user_message
    assert "Why is the sky blue?" in user_message
    assert "facts.txt" in user_message


@pytest.mark.asyncio
async def test_nemotron_llm_uses_custom_system_prompt_when_provided():
    client = _FakeAsyncOpenAI(_token_stream("ok"))
    llm = NvidiaNemotronLLM(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]

    async for _ in llm.stream_generate("q", context=[], system_prompt="Custom instructions."):
        pass

    sent_messages = client.chat.completions.last_call_kwargs["messages"]
    system_message = next(m["content"] for m in sent_messages if m["role"] == "system")
    assert system_message == "Custom instructions."


@pytest.mark.asyncio
async def test_nemotron_llm_calls_with_stream_true():
    client = _FakeAsyncOpenAI(_token_stream("ok"))
    llm = NvidiaNemotronLLM(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]
    async for _ in llm.stream_generate("q", context=[]):
        pass
    assert client.chat.completions.last_call_kwargs["stream"] is True
    assert client.chat.completions.last_call_kwargs["model"] == "fake/nemotron-model"
