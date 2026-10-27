"""Typed application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    """Runtime configuration for the agent service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Gabriel Agent"
    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8086, ge=1, le=65535)
    app_log_level: str = "INFO"
    app_docs_enabled: bool = True

    db_host: str = "localhost"
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str = "gabriel_db"
    db_username: str = "postgres"
    db_password: SecretStr
    db_echo: bool = False
    db_connect_timeout_seconds: float = Field(default=5.0, gt=0)

    chat_base_url: str
    chat_api_key: SecretStr
    chat_model: str
    chat_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    chat_timeout_seconds: float = Field(default=60.0, gt=0)
    chat_max_retries: int = Field(default=2, ge=0, le=10)
    chat_context_message_limit: int = Field(default=200, ge=1, le=1000)

    qq_bot_enabled: bool = False
    qq_bot_app_id: str | None = None
    qq_bot_app_secret: SecretStr | None = None

    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_model: str | None = None
    embedding_dimensions: int = Field(default=1024, gt=0)

    rag_enabled: bool = False
    rag_top_k: int = Field(default=5, ge=1, le=100)
    rag_vector_table: str = "vector_store"
    rag_vector_index_type: Literal["HNSW", "IVFFlat"] = "HNSW"
    rag_vector_distance: Literal["cosine", "l2", "inner_product"] = "cosine"

    @model_validator(mode="after")
    def validate_optional_integrations(self) -> Settings:
        if self.rag_enabled:
            missing_embeddings = [
                name
                for name, value in (
                    ("EMBEDDING_BASE_URL", self.embedding_base_url),
                    ("EMBEDDING_API_KEY", self.embedding_api_key),
                    ("EMBEDDING_MODEL", self.embedding_model),
                )
                if value is None or (isinstance(value, SecretStr) and not value.get_secret_value())
            ]
            if missing_embeddings:
                raise ValueError(f"RAG_ENABLED=true requires: {', '.join(missing_embeddings)}")

        if self.qq_bot_enabled:
            missing_qq = [
                name
                for name, value in (
                    ("QQ_BOT_APP_ID", self.qq_bot_app_id),
                    ("QQ_BOT_APP_SECRET", self.qq_bot_app_secret),
                )
                if value is None
                or (isinstance(value, str) and not value.strip())
                or (isinstance(value, SecretStr) and not value.get_secret_value())
            ]
            if missing_qq:
                raise ValueError(f"QQ_BOT_ENABLED=true requires: {', '.join(missing_qq)}")
        return self

    @property
    def database_url(self) -> URL:
        """Build a safe SQLAlchemy URL from individual database settings."""

        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.db_username,
            password=self.db_password.get_secret_value(),
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )

    @property
    def docs_url(self) -> str | None:
        """Return the OpenAPI UI path when documentation is enabled."""

        return "/docs" if self.app_docs_enabled else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
