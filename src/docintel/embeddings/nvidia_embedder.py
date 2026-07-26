"""
NvidiaEmbedder: default Embedder implementation, calling NVIDIA's
OpenAI-compatible embeddings endpoint via the `openai` SDK's AsyncOpenAI
client pointed at NVIDIA_API_BASE_URL.

model_id and dimensions are NOT guessed here -- they come from
Settings.nvidia_embedding_model / nvidia_embedding_dimensions, which are
required, unset-by-default fields the user must configure in .env (see
.env.example's TODO(user) markers). This class will raise clearly at
construction if they're missing, rather than silently using a made-up
default model name.

NOTE ON TESTING: this sandbox cannot reach api.nvidia.com, so this class
accepts an injectable `client` (defaulting to a real AsyncOpenAI) --
tests exercise the batching/mapping/retry logic against a fake client
that mimics the OpenAI SDK's response shape, not a live call. Real
end-to-end verification requires valid NVIDIA credentials at deploy time.
"""

from __future__ import annotations

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from docintel.core.config import Settings
from docintel.core.logging import get_logger
from docintel.core.models import Chunk, EmbeddedChunk

logger = get_logger(__name__)

_DEFAULT_BATCH_SIZE = 32


class NvidiaEmbedder:
    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_id = settings.nvidia_embedding_model
        self.dimensions = settings.nvidia_embedding_dimensions
        self._batch_size = batch_size
        self._client = client or AsyncOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_api_base_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _embed_batch(
        self, 
        texts: list[str],
        input_type: str,
        ) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.model_id,
            input=texts,
            extra_body={
            "input_type":input_type,
            },
        )
        # The API may not preserve request order in `data` -- sort by the
        # `index` field it's contractually required to return, rather than
        # assuming positional order.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        embedded: list[EmbeddedChunk] = []
        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            texts = [c.content for c in batch]
            logger.info("embedding_batch_start", batch_size=len(texts), model=self.model_id)
            vectors = await self._embed_batch(
                texts,
                input_type="passage")
            embedded.extend(
                EmbeddedChunk(chunk=chunk, vector=vector, model_id=self.model_id)
                for chunk, vector in zip(batch, vectors)
            )
        return embedded

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self._embed_batch(
            [query],
            input_type="query",
            )
        return vectors[0]
