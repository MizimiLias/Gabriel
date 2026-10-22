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
