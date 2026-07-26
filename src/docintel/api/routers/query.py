"""Query endpoint: hybrid retrieval + streaming generation with citations."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from docintel.api.container import AppContainer
from docintel.api.dependencies import get_container
from docintel.api.schemas import QueryRequest
from docintel.knowledge_base.manager import KnowledgeBaseNotFoundError

router = APIRouter(prefix="/knowledge-bases/{kb_id}", tags=["query"])


async def _sse_stream(container: AppContainer, kb_id: str, query: str, top_k: int) -> AsyncIterator[str]:
    kb = await container.kb_manager.get(kb_id)

    results = await container.hybrid_retriever.retrieve(kb_id, query, top_k)

    async for generation_chunk in container.llm.stream_generate(
        query=query, context=results, system_prompt=kb.system_prompt
    ):
        payload = {
            "text": generation_chunk.text,
            "is_final": generation_chunk.is_final,
            "citations": [c.model_dump() for c in generation_chunk.citations],
        }
        yield f"data: {json.dumps(payload)}\n\n"


@router.post("/query")
async def query_knowledge_base(
    kb_id: str,
    body: QueryRequest,
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    try:
        await container.kb_manager.get(kb_id)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return StreamingResponse(
        _sse_stream(container, kb_id, body.query, body.top_k),
        media_type="text/event-stream",
    )
