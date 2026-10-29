"""Chat service memory, title, and persistence tests."""

from typing import cast
from uuid import uuid4

import pytest

from app.entities.chat_session import SessionSource
from app.exceptions.chat import PersonaNotFoundError
from app.prompts import TITLE_SYSTEM_PROMPT
from app.repositories.chat_repository import ChatRepository
from app.repositories.persona_repository import (
    PersonaMessageDraft,
    PersonaMessageMutation,
    PersonaRecord,
    PersonaRepository,
)
from app.schemas.content import ImageURL, ImageURLPart
from app.services.chat_service import ChatService, StreamDone, StreamStart, StreamToken
from tests.support import (
    FakeChatModel,
    FakeChatRepository,
    FakePersonaRepository,
    make_settings,
    text_parts,
    visible_message_text,
)


def _image_part(url: str) -> ImageURLPart:
    return ImageURLPart(image_url=ImageURL(url=url))


def build_service(
    *,
    repository: FakeChatRepository,
    model: FakeChatModel,
    persona_repository: FakePersonaRepository | None = None,
    context_limit: int = 20,
) -> ChatService:
    persona_kwargs = (
        {}
        if persona_repository is None
        else {"persona_repository": cast(PersonaRepository, persona_repository)}
    )
    return ChatService(
        cast(ChatRepository, repository),
        model.as_base_chat_model(),
        make_settings(chat_context_message_limit=context_limit),
        **persona_kwargs,
    )


async def test_second_turn_receives_persisted_history() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(response="first answer")
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("My name is Ada"))
    model.response = "Your name is Ada"
    await service.send_message(session_id, text_parts("What is my name?"))

    serialized = [(message.type, visible_message_text(message)) for message in model.last_messages]
    assert serialized == [
        ("human", "My name is Ada"),
        ("ai", "first answer"),
        ("human", "What is my name?"),
    ]


async def test_system_reminder_is_appended_and_persisted() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("hello"))

    stored_parts = repository.saved_exchanges[0][1]
    assert len(stored_parts) == 2
    assert stored_parts[0] == {"type": "text", "text": "hello"}
    reminder = str(stored_parts[1]["text"])
    assert reminder.startswith("<system_reminder>Current datetime: ")
    assert "(Asia/Shanghai), Weekday: " in reminder
    assert reminder.endswith("</system_reminder>")
    assert repository.messages[session_id][0].parts == stored_parts


async def test_image_part_reaches_model_and_is_persisted() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model)
    session_id = uuid4()
    image = _image_part("https://example.com/image.png")

    await service.send_message(session_id, [*text_parts("describe this"), image])

    human_message = model.calls[0][0]
    assert isinstance(human_message.content, list)
    assert human_message.content[1] == image.model_dump(mode="json")
    assert repository.saved_exchanges[0][1][1] == image.model_dump(mode="json")


async def test_context_limit_keeps_only_recent_messages() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model, context_limit=2)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("one"))
    await service.send_message(session_id, text_parts("two"))
    await service.send_message(session_id, text_parts("three"))

    assert [visible_message_text(message) for message in model.last_messages] == [
        "two",
        model.response,
        "three",
    ]


async def test_session_without_persona_sends_no_system_message() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("hello"), source=SessionSource.QQ_C2C)

    assert [message.type for message in model.last_messages] == ["human"]


async def test_first_exchange_generates_title_once() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(title_response="Ada's name")
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    reply = await service.send_message(session_id, text_parts("My name is Ada"))
    assert reply.title == "Ada's name"
    assert repository.titles[session_id] == "Ada's name"

    model.title_response = "A different title"
    await service.send_message(session_id, text_parts("What is my name?"))
    assert repository.titles[session_id] == "Ada's name"


async def test_title_generation_failure_uses_first_message() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(fail_title=True)
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    reply = await service.send_message(session_id, text_parts("  Plan a quiet weekend  "))
    assert reply.title == "Plan a quiet weekend"
    assert len(repository.messages[session_id]) == 2


@pytest.mark.parametrize("source", [SessionSource.QQ_C2C, SessionSource.QQ_GROUP])
async def test_non_web_source_skips_title_generation(source: SessionSource) -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    reply = await service.send_message(session_id, text_parts("hello"), source=source)

    assert reply.title is None
    assert repository.titles[session_id] is None
    assert model.title_calls == 0


@pytest.mark.parametrize("source", [SessionSource.QQ_C2C, SessionSource.QQ_GROUP])
async def test_non_web_stream_skips_title_generation(source: SessionSource) -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    events = [
        event async for event in service.stream_message(session_id, text_parts("Hi"), source=source)
    ]

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.title is None
    assert repository.titles[session_id] is None
    assert model.title_calls == 0


async def test_stream_persists_only_after_completion() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(chunks=["Hello", " world"])
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    events = [event async for event in service.stream_message(session_id, text_parts("Hi"))]

    assert isinstance(events[0], StreamStart)
    assert [event.delta for event in events if isinstance(event, StreamToken)] == [
        "Hello",
        " world",
    ]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].title == "Test conversation"
    assert len(repository.messages[session_id]) == 2


async def test_stream_failure_does_not_persist() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel(fail=True)
    service = build_service(repository=repository, model=model)
    session_id = uuid4()

    with pytest.raises(Exception, match="could not complete"):
        _ = [event async for event in service.stream_message(session_id, text_parts("Hi"))]

    assert session_id not in repository.messages


async def test_external_session_identity_is_persisted() -> None:
    repository = FakeChatRepository()
    service = build_service(repository=repository, model=FakeChatModel())
    session_id = uuid4()

    await service.send_message(
        session_id,
        text_parts("hello"),
        source=SessionSource.QQ_C2C,
        external_key="app:user",
    )

    assert repository.sources[session_id] == SessionSource.QQ_C2C
    assert repository.external_keys[session_id] == "app:user"
    assert repository.titles[session_id] is None


def _persona_messages() -> list[PersonaMessageDraft]:
    return [
        PersonaMessageDraft(role="user", content="Who are you?"),
        PersonaMessageDraft(role="assistant", content="I am Ada."),
    ]


async def _create_persona(
    personas: FakePersonaRepository,
    *,
    name: str = "Ada",
    system_prompt: str = "You are Ada.",
) -> PersonaRecord:
    return await personas.create(
        name=name,
        description=None,
        system_prompt=system_prompt,
        messages=_persona_messages(),
    )


async def test_first_web_message_injects_persona() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    persona = await _create_persona(personas)
    model = FakeChatModel()
    service = build_service(repository=repository, model=model, persona_repository=personas)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("hello"), persona_id=persona.persona_id)

    assert [(message.type, visible_message_text(message)) for message in model.calls[0]] == [
        ("system", "You are Ada."),
        ("human", "Who are you?"),
        ("ai", "I am Ada."),
        ("human", "hello"),
    ]
    assert repository.persona_ids[session_id] == persona.persona_id


async def test_first_web_message_without_persona_has_no_system_message() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(
        repository=repository, model=model, persona_repository=FakePersonaRepository()
    )
    session_id = uuid4()

    await service.send_message(session_id, text_parts("hello"))

    assert [(message.type, visible_message_text(message)) for message in model.calls[0]] == [
        ("human", "hello")
    ]
    assert repository.persona_ids[session_id] is None


async def test_existing_session_keeps_its_persona_and_ignores_requested_id() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    first = await _create_persona(personas, name="Ada", system_prompt="You are Ada.")
    second = await _create_persona(personas, name="Bob", system_prompt="You are Bob.")
    model = FakeChatModel()
    service = build_service(repository=repository, model=model, persona_repository=personas)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("first"), persona_id=first.persona_id)
    await service.send_message(session_id, text_parts("second"), persona_id=second.persona_id)

    assert visible_message_text(model.calls[-1][0]) == "You are Ada."
    assert repository.persona_ids[session_id] == first.persona_id


async def test_existing_session_without_persona_ignores_requested_id() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    persona = await _create_persona(personas)
    model = FakeChatModel()
    service = build_service(repository=repository, model=model, persona_repository=personas)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("first"))
    await service.send_message(session_id, text_parts("second"), persona_id=persona.persona_id)

    assert model.calls[-1][0].type == "human"
    assert repository.persona_ids[session_id] is None


async def test_qq_session_with_database_persona_is_injected() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    persona = await _create_persona(personas)
    model = FakeChatModel()
    service = build_service(repository=repository, model=model, persona_repository=personas)
    session_id = uuid4()

    await repository.save_exchange(
        session_id=session_id,
        user_parts=[{"type": "text", "text": "earlier"}],
        assistant_parts=[{"type": "text", "text": "answer"}],
        assistant_message_id=uuid4(),
        source=SessionSource.QQ_GROUP,
        external_key="app:group",
        persona_id=persona.persona_id,
    )

    await service.send_message(session_id, text_parts("hello"), source=SessionSource.QQ_GROUP)

    assert visible_message_text(model.calls[-1][0]) == "You are Ada."


async def test_qq_session_without_persona_has_no_system_message() -> None:
    repository = FakeChatRepository()
    model = FakeChatModel()
    service = build_service(
        repository=repository, model=model, persona_repository=FakePersonaRepository()
    )
    session_id = uuid4()

    await service.send_message(session_id, text_parts("hello"), source=SessionSource.QQ_C2C)

    assert model.calls[-1][0].type == "human"


async def test_persona_edits_are_read_on_the_next_message() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    persona = await _create_persona(personas)
    model = FakeChatModel()
    service = build_service(repository=repository, model=model, persona_repository=personas)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("first"), persona_id=persona.persona_id)
    await personas.update_with_messages(
        persona.persona_id,
        name=persona.name,
        description=None,
        system_prompt="You are Ada, briefly.",
        messages=[
            PersonaMessageMutation(
                id=message.id,
                role=message.role,
                content=message.content,
            )
            for message in persona.messages
        ],
    )
    await service.send_message(session_id, text_parts("second"))

    assert visible_message_text(model.calls[-1][0]) == "You are Ada, briefly."


async def test_deleted_persona_falls_back_to_no_system_message() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    persona = await _create_persona(personas)
    model = FakeChatModel()
    service = build_service(repository=repository, model=model, persona_repository=personas)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("first"), persona_id=persona.persona_id)
    await personas.delete(persona.persona_id)
    await service.send_message(session_id, text_parts("second"))

    assert model.calls[-1][0].type == "human"


async def test_unknown_persona_on_first_message_raises() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    service = build_service(
        repository=repository, model=FakeChatModel(), persona_repository=personas
    )

    with pytest.raises(PersonaNotFoundError):
        await service.send_message(uuid4(), text_parts("hello"), persona_id=uuid4())


async def test_stream_injects_persona_and_persists_only_the_exchange() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    persona = await _create_persona(personas)
    model = FakeChatModel(chunks=["Hi", " there"])
    service = build_service(repository=repository, model=model, persona_repository=personas)
    session_id = uuid4()

    events = [
        event
        async for event in service.stream_message(
            session_id,
            text_parts("hello"),
            source=SessionSource.QQ_C2C,
            persona_id=persona.persona_id,
        )
    ]

    assert isinstance(events[-1], StreamDone)
    assert [(message.type, visible_message_text(message)) for message in model.calls[0]] == [
        ("system", "You are Ada."),
        ("human", "Who are you?"),
        ("ai", "I am Ada."),
        ("human", "hello"),
    ]
    assert repository.persona_ids[session_id] == persona.persona_id
    assert len(repository.messages[session_id]) == 2


async def test_presets_do_not_consume_the_history_limit() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    persona = await _create_persona(personas)
    model = FakeChatModel(response="answer")
    service = build_service(
        repository=repository,
        model=model,
        persona_repository=personas,
        context_limit=1,
    )
    session_id = uuid4()

    await service.send_message(
        session_id, text_parts("one"), source=SessionSource.QQ_C2C, persona_id=persona.persona_id
    )
    await service.send_message(session_id, text_parts("two"), source=SessionSource.QQ_C2C)

    chat_call = model.calls[-1]
    assert visible_message_text(chat_call[0]) == "You are Ada."
    assert [visible_message_text(message) for message in chat_call[1:3]] == [
        "Who are you?",
        "I am Ada.",
    ]
    assert chat_call[3].type == "ai"
    assert visible_message_text(chat_call[4]) == "two"


async def test_title_generation_excludes_persona_presets() -> None:
    repository = FakeChatRepository()
    personas = FakePersonaRepository()
    persona = await _create_persona(personas)
    model = FakeChatModel(title_response="Greeting")
    service = build_service(repository=repository, model=model, persona_repository=personas)
    session_id = uuid4()

    await service.send_message(session_id, text_parts("hello"), persona_id=persona.persona_id)

    title_call = model.calls[1]
    assert visible_message_text(title_call[0]) == TITLE_SYSTEM_PROMPT
    assert all("Who are you?" not in visible_message_text(message) for message in title_call)
