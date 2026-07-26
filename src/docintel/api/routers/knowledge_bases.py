"""Knowledge base CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from docintel.api.container import AppContainer
from docintel.api.dependencies import get_container
from docintel.api.schemas import CreateKnowledgeBaseRequest, KnowledgeBaseResponse
from docintel.core.models import KnowledgeBase
from docintel.knowledge_base.manager import KnowledgeBaseNotFoundError

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


def _to_response(kb: KnowledgeBase) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        embedding_model_id=kb.embedding_model_id,
        embedding_dimensions=kb.embedding_dimensions,
        system_prompt=kb.system_prompt,
    )


@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(
    body: CreateKnowledgeBaseRequest,
    container: AppContainer = Depends(get_container),
) -> KnowledgeBaseResponse:
    kb = await container.kb_manager.create(
        name=body.name,
        description=body.description,
        embedding_model_id=body.embedding_model_id or container.settings.nvidia_embedding_model,
        embedding_dimensions=body.embedding_dimensions or container.settings.nvidia_embedding_dimensions,
        system_prompt=body.system_prompt,
    )
    return _to_response(kb)


@router.get("", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(
    container: AppContainer = Depends(get_container),
) -> list[KnowledgeBaseResponse]:
    kbs = await container.kb_manager.list_all()
    return [_to_response(kb) for kb in kbs]


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: str, container: AppContainer = Depends(get_container)
) -> KnowledgeBaseResponse:
    try:
        kb = await container.kb_manager.get(kb_id)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(kb)


@router.delete("/{kb_id}", status_code=204)
async def delete_knowledge_base(
    kb_id: str, container: AppContainer = Depends(get_container)
) -> None:
    try:
        await container.kb_manager.delete(kb_id)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
