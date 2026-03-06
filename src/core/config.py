from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application settings
    app_name: str = Field(default="Atom API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_description: str = Field(
        default="Autonomous Knowledge System for PDF Documents",
        description="Application description"
    )
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")

    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=True, description="Auto-reload on code changes")

    # Database settings (PostgreSQL)
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/atom",
        description="Async database connection URL"
    )
    database_echo: bool = Field(default=False, description="Echo SQL queries")

    # Redis settings (for Dramatiq)
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")

    # CORS settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins"
    )
    cors_methods: list[str] = Field(
        default=["*"], description="CORS allowed methods"
    )
    cors_headers: list[str] = Field(
        default=["*"], description="CORS allowed headers"
    )

    # API settings
    api_prefix: str = Field(default="/api/v1", description="API URL prefix")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        description="Log format"
    )

    # AI Provider settings
    ai_provider: str = Field(default="openai", description="AI provider: openai")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI chat model")
    openai_mini_model: str = Field(default="gpt-4o-mini", description="OpenAI mini model for lightweight tasks")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )

    # Qdrant settings
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant REST port")
    qdrant_collection_name: str = Field(
        default="document_chunks",
        description="Qdrant collection name"
    )

    # LangSmith settings
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(default="atom-agent", description="LangSmith project name")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", description="LangSmith endpoint")

    # Upload settings
    upload_dir: str = Field(default="./uploads", description="Directory for uploaded files")

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def sync_database_url(self) -> str:
        """Return a synchronous database URL for use in Dramatiq workers."""
        return self.database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()