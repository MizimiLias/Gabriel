"""Persona management endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Response, status

from app.configs.dependencies import get_persona_service
from app.repositories.persona_repository import (
    PersonaMessageDraft,
    PersonaMessageMutation,
    PersonaRecord,
    PersonaSummaryRow,
)
from app.schemas.persona import (
    PersonaCreate,
    PersonaListResponse,
    PersonaMessageResponse,
    PersonaResponse,
    PersonaSummaryResponse,
    PersonaUpdate,
)
from app.services.persona_service import PersonaService

router = APIRouter(prefix="/api/personas", tags=["personas"])


def _message_drafts(payload: PersonaCreate) -> list[PersonaMessageDraft]:
    return [
        PersonaMessageDraft(role=message.role, content=message.content)
        for message in payload.messages
    ]


def _message_mutations(payload: PersonaUpdate) -> list[PersonaMessageMutation]:
    return [
        PersonaMessageMutation(id=message.id, role=message.role, content=message.content)
        for message in payload.messages
    ]


def _persona_response(record: PersonaRecord) -> PersonaResponse:
    return PersonaResponse(
        persona_id=record.persona_id,
        name=record.name,
        description=record.description,
        system_prompt=record.system_prompt,
        messages=[
            PersonaMessageResponse(
                id=message.id,
                position=message.position,
                role=message.role,
                content=message.content,
            )
            for message in record.messages
        ],
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _persona_summary(row: PersonaSummaryRow) -> PersonaSummaryResponse:
    return PersonaSummaryResponse(
        persona_id=row.persona_id,
        name=row.name,
        description=row.description,
        message_count=row.message_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.post("", response_model=PersonaResponse, status_code=status.HTTP_201_CREATED)
async def create_persona(
    payload: PersonaCreate,
    service: Annotated[PersonaService, Depends(get_persona_service)],
) -> PersonaResponse:
    """Create a persona with optional preset dialogue."""

    record = await service.create(
        name=payload.name,
        description=payload.description,
        system_prompt=payload.system_prompt,
        messages=_message_drafts(payload),
    )
    return _persona_response(record)


@router.get("", response_model=PersonaListResponse)
async def list_personas(
    service: Annotated[PersonaService, Depends(get_persona_service)],
) -> PersonaListResponse:
    """Return every persona ordered by name."""

    rows = await service.list_all()
    return PersonaListResponse(items=[_persona_summary(row) for row in rows])


@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(
    persona_id: Annotated[UUID, Path()],
    service: Annotated[PersonaService, Depends(get_persona_service)],
) -> PersonaResponse:
    """Return one persona with its ordered preset dialogue."""

    return _persona_response(await service.get(persona_id))


@router.put("/{persona_id}", response_model=PersonaResponse)
async def update_persona(
    persona_id: Annotated[UUID, Path()],
    payload: PersonaUpdate,
    service: Annotated[PersonaService, Depends(get_persona_service)],
) -> PersonaResponse:
    """Atomically update persona metadata and preset dialogue."""

    record = await service.update(
        persona_id,
        name=payload.name,
        description=payload.description,
        system_prompt=payload.system_prompt,
        messages=_message_mutations(payload),
    )
    return _persona_response(record)


@router.delete("/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_persona(
    persona_id: Annotated[UUID, Path()],
    service: Annotated[PersonaService, Depends(get_persona_service)],
) -> Response:
    """Delete a persona; the database clears session references."""

    await service.delete(persona_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
