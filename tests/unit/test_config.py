"""Settings and RAG validation tests."""

import pytest
from pydantic import ValidationError

from tests.support import make_settings


def test_rag_disabled_does_not_require_embedding_credentials() -> None:
    settings = make_settings(rag_enabled=False)

    assert settings.embedding_api_key is None
    assert settings.rag_enabled is False


def test_rag_enabled_requires_embedding_credentials() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_API_KEY"):
        make_settings(
            rag_enabled=True,
            embedding_base_url="https://example.invalid/v1",
            embedding_model="test-embedding",
        )


def test_rag_enabled_accepts_complete_embedding_configuration() -> None:
    settings = make_settings(
        rag_enabled=True,
        embedding_base_url="https://example.invalid/v1",
        embedding_api_key="embedding-key",
        embedding_model="test-embedding",
        embedding_dimensions=1024,
    )

    assert settings.embedding_dimensions == 1024


def test_qq_disabled_does_not_require_credentials() -> None:
    settings = make_settings(qq_bot_enabled=False)

    assert settings.qq_bot_enabled is False
    assert settings.qq_bot_app_id is None
    assert settings.qq_bot_app_secret is None


@pytest.mark.parametrize(
    ("app_id", "app_secret", "error_name"),
    [
        (None, None, "QQ_BOT_APP_ID"),
        ("app-id", None, "QQ_BOT_APP_SECRET"),
        ("", "secret", "QQ_BOT_APP_ID"),
    ],
)
def test_qq_enabled_requires_credentials(
    app_id: str | None, app_secret: str | None, error_name: str
) -> None:
    with pytest.raises(ValidationError, match=error_name):
        make_settings(
            qq_bot_enabled=True,
            qq_bot_app_id=app_id,
            qq_bot_app_secret=app_secret,
        )


def test_qq_enabled_accepts_complete_configuration() -> None:
    settings = make_settings(
        qq_bot_enabled=True,
        qq_bot_app_id="app-id",
        qq_bot_app_secret="secret",
    )

    assert settings.qq_bot_app_id == "app-id"
    assert settings.qq_bot_app_secret is not None
