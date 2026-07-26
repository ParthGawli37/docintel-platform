"""
NvidiaNemotronLLM: default LLM implementation, streaming chat completions
from NVIDIA's OpenAI-compatible endpoint via AsyncOpenAI, with
`stream=True` against the confirmed `chat.completions.create` API shape.

model_id comes from Settings.nvidia_generation_model -- required, unset
by default (see .env.example's TODO(user) marker), never guessed here.

NOTE ON TESTING: same caveat as NvidiaEmbedder -- this sandbox cannot
reach api.nvidia.com, so `client` is injectable and tests exercise the
prompt-building/streaming/citation-attachment logic against a fake async
iterator shaped like AsyncStream[ChatCompletionChunk], not a live call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk

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

        stream = cast(
            AsyncStream[ChatCompletionChunk],
            await self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            ),
        )

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
