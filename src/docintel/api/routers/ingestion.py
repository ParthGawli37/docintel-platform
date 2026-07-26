"""Ingestion endpoints: file upload and URL ingestion into a knowledge base."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from docintel.api.container import AppContainer
from docintel.api.dependencies import get_container
from docintel.api.schemas import IngestResultResponse, IngestUrlRequest
from docintel.knowledge_base.manager import KnowledgeBaseNotFoundError

router = APIRouter(prefix="/knowledge-bases/{kb_id}/ingest", tags=["ingestion"])


@router.post("/file", response_model=IngestResultResponse)
async def ingest_file(
    kb_id: str,
    file: UploadFile,
    container: AppContainer = Depends(get_container),
) -> IngestResultResponse:
    await _require_kb_exists(kb_id, container)

    content = await file.read()
    saved_path = await container.raw_file_store.save_bytes(kb_id, file.filename or "upload", content)

    try:
        loader = container.loader_registry.resolve(saved_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    indexer = container.build_indexer(loader)
    result = await indexer.index_source(saved_path, kb_id)
    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    return IngestResultResponse(
        source=result.source, skipped=result.skipped, chunk_count=result.chunk_count, error=result.error
    )


@router.post("/url", response_model=IngestResultResponse)
async def ingest_url(
    kb_id: str,
    body: IngestUrlRequest,
    container: AppContainer = Depends(get_container),
) -> IngestResultResponse:
    await _require_kb_exists(kb_id, container)

    try:
        loader = container.loader_registry.resolve(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    indexer = container.build_indexer(loader)
    result = await indexer.index_source(body.url, kb_id)
    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    return IngestResultResponse(
        source=result.source, skipped=result.skipped, chunk_count=result.chunk_count, error=result.error
    )


async def _require_kb_exists(kb_id: str, container: AppContainer) -> None:
    try:
        await container.kb_manager.get(kb_id)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
