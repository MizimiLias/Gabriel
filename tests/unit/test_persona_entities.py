"""Persona entity metadata tests."""

from app.entities.base import Base
from app.entities.chat_session import ChatSession
from app.entities.persona import Persona
from app.entities.persona_message import PersonaMessage


def test_persona_tables_are_registered() -> None:
    assert Persona.__tablename__ == "personas"
    assert PersonaMessage.__tablename__ == "persona_messages"
    assert "personas" in Base.metadata.tables
    assert "persona_messages" in Base.metadata.tables


def test_chat_session_has_persona_foreign_key() -> None:
    column = ChatSession.__table__.c.persona_id

    assert column.nullable is True
    assert [foreign_key.target_fullname for foreign_key in column.foreign_keys] == [
        "personas.persona_id"
    ]


def test_persona_message_orders_by_position() -> None:
    constraints = {constraint.name for constraint in PersonaMessage.__table__.constraints}

    assert "uq_persona_messages_persona_id_position" in constraints
    assert "ck_persona_messages_valid_role" in constraints
