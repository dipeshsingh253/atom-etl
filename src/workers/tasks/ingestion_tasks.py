"""Dramatiq task for asynchronous PDF document ingestion."""

import asyncio

import dramatiq
from loguru import logger

from src.workers.broker import get_broker

# Ensure broker is initialized
get_broker()


async def _get_session():
    """Get an async DB session, initializing the engine if needed."""
    from src.db import session as db_session

    if db_session.SessionLocal is None:
        await db_session.init_db()

    return db_session.SessionLocal()


async def _run_ingestion(document_id: str, file_path: str) -> None:
    """Run the ETL pipeline with a fresh async DB session."""
    async with await _get_session() as db:
        try:
            from src.modules.etl.pipeline import run_pipeline
            await run_pipeline(document_id, file_path, db)
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()


@dramatiq.actor(max_retries=1, min_backoff=5000, max_backoff=60000, time_limit=1800000)
def process_document(document_id: str, file_path: str) -> None:
    """Process a PDF document through the ETL pipeline.

    This task runs asynchronously via Dramatiq. It:
    1. Extracts text, tables, and visual elements from the PDF
    2. Stores text chunks in Qdrant (vector database)
    3. Stores structured data in PostgreSQL
    4. Updates document status throughout

    Args:
        document_id: Database ID of the document record.
        file_path: Path to the uploaded PDF file.
    """
    logger.info(f"Starting ingestion task for document {document_id}")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_ingestion(document_id, file_path))
        finally:
            loop.close()

        logger.info(f"Ingestion task completed for document {document_id}")

    except Exception as e:
        logger.error(f"Ingestion task failed for document {document_id}: {e}")
        # Try to mark document as failed
        try:
            fail_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(fail_loop)
            try:
                fail_loop.run_until_complete(_mark_failed(document_id))
            finally:
                fail_loop.close()
        except Exception:
            logger.error("Could not mark document as failed")
        raise


async def _mark_failed(document_id: str) -> None:
    """Mark a document as failed in the database."""
    from src.modules.documents.service import update_document_status

    async with await _get_session() as db:
        await update_document_status(db, document_id, "failed")
