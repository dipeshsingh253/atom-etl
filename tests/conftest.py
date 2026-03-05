import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from src.core.config import Settings, get_settings
from src.db.base import Base
from src.db.session import get_db
from src.main import create_app


# ---------------------------------------------------------------------------
# Test database URL — uses a separate 'atom_test' database via SQLite for tests
# Override with TEST_DATABASE_URL env var if you have PostgreSQL running
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "sqlite+aiosqlite:///./test_atom.db",
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings."""
    test_config = {
        "database_url": TEST_DATABASE_URL,
        "database_echo": False,
        "environment": "testing",
        "debug": True,
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 1,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "qdrant_collection_name": "test_document_chunks",
        "openai_api_key": "test-key",
        "upload_dir": "./test_uploads",
    }
    return Settings(**test_config)


@pytest_asyncio.fixture(scope="session")
async def test_engine(test_settings: Settings):
    """Create test database engine and tables."""
    connect_args = {}
    pool_class = NullPool

    # SQLite needs special handling
    if "sqlite" in test_settings.database_url:
        connect_args = {"check_same_thread": False}

    engine = create_async_engine(
        test_settings.database_url,
        echo=test_settings.database_echo,
        poolclass=pool_class,
        connect_args=connect_args,
    )

    # Import models to register them with Base.metadata
    from src.modules.documents.model import (  # noqa: F401
        Document,
        DocumentTable,
        DocumentVisual,
        TableRow,
        VisualData,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()

    # Clean up test database file if SQLite
    if "sqlite" in test_settings.database_url:
        db_path = test_settings.database_url.replace("sqlite+aiosqlite:///", "")
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing with cleanup."""
    TestSessionLocal = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with TestSessionLocal() as session:
        # Clean up tables before each test
        table_names = [
            "visual_data",
            "document_visuals",
            "table_rows",
            "document_tables",
            "documents",
        ]
        for table_name in table_names:
            try:
                await session.execute(text(f"DELETE FROM {table_name}"))
            except Exception:
                pass
        await session.commit()

    async with TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@pytest_asyncio.fixture
async def client(test_settings: Settings, db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with dependency overrides."""

    def get_test_settings():
        return test_settings

    async def get_test_db():
        yield db_session

    with patch("src.workers.broker.setup_dramatiq"), \
         patch("src.workers.tasks.ingestion_tasks.process_document.send"):

        app = create_app()

        app.dependency_overrides[get_settings] = get_test_settings
        app.dependency_overrides[get_db] = get_test_db

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as test_client:
            yield test_client

        app.dependency_overrides.clear()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DATABASE_ECHO"] = "false"
    os.environ["OPENAI_API_KEY"] = "test-key"
    yield