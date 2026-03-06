import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from loguru import logger

from src.core.config import get_settings
from src.db.session import create_tables
from src.vectorstore.qdrant import init_collection


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting up AtomETL...")
    settings = get_settings()

    # Create upload directory
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.upload_dir, "figures"), exist_ok=True)
    logger.info(f"Upload directory ensured at: {settings.upload_dir}")

    # Create database tables
    await create_tables()
    logger.info("Database tables created successfully")

    # Initialize Qdrant collection
    try:
        await init_collection()
        logger.info("Qdrant collection initialized successfully")
    except Exception as e:
        logger.warning(f"Qdrant initialization failed (may not be running): {e}")

    yield

    logger.info("Shutting down AtomETL...")
    logger.info("Cleanup completed successfully")