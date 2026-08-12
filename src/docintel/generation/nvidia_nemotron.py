"""NVIDIA Nemotron LLM provider with streaming generation and bounded retries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from tenacity import retry, stop_after_attempt, wait_exponential

from docintel.citations.builder import DefaultCitationBuilder
from docintel.core.config import Settings
from docintel.core.interfaces import CitationBuilder
from docintel.core.logging import get_logger
from docintel.core.models import GenerationChunk, SearchResult

logger = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "provided context. If the context does not contain the answer, say "
    "so plainly rather than guessing."
)


def _build_context_block(context: list[SearchResult]) -> str:
    sections = []
    for i, result in enumerate(context, start=1):
        source = result.chunk.metadata.source_uri
        sections.append(f"[Source {i}: {source}]\n{result.chunk.content}")
    return "\n\n".join(sections)


class NvidiaNemotronLLM:
    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
        citation_builder: CitationBuilder | None = None,
    ) -> None:
        self.model_id = settings.nvidia_generation_model
        self._client = client or AsyncOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_api_base_url,
        )
        self._citation_builder = citation_builder or DefaultCitationBuilder()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _create_stream(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncStream[ChatCompletionChunk]:
        """Create the upstream stream with bounded exponential-backoff retries."""
        logger.info("generation_request", model=self.model_id)
        stream = await self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
        )
        return cast(AsyncStream[ChatCompletionChunk], stream)

    async def stream_generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        citations = self._citation_builder.build(context)
        context_block = _build_context_block(context)
        messages = [
            {"role": "system", "content": system_prompt or _DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {query}",
            },
        ]

        logger.info("generation_start", model=self.model_id, context_chunk_count=len(context))
        stream = await self._create_stream(messages)

        async for event in stream:
            delta = event.choices[0].delta if event.choices else None
            text = delta.content if delta and delta.content else ""
            is_final = bool(event.choices and event.choices[0].finish_reason is not None)
            if text or is_final:
                yield GenerationChunk(
                    text=text,
                    citations=citations if is_final else [],
                    is_final=is_final,
                )

        logger.info("generation_complete", model=self.model_id)
