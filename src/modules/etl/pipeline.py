"""ETL pipeline orchestrator — ties all extraction steps together."""

import os

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.modules.documents.model import (
    Document,
    DocumentTable,
    DocumentVisual,
    TableRow,
    VisualData,
)
from src.modules.etl.table_extractor import extract_tables
from src.modules.etl.text_chunker import chunk_pages
from src.modules.etl.text_extractor import extract_text, get_page_count
from src.modules.etl.visual_analyzer import analyze_visuals_batch
from src.modules.etl.visual_extractor import extract_visuals
from src.providers.factory import get_embedding_provider, get_vision_provider
from src.vectorstore.qdrant import store_chunks


async def run_pipeline(document_id: str, file_path: str, db: AsyncSession) -> None:
    """Run the full ETL pipeline for a PDF document.

    Steps:
        1. Update document status to 'processing'
        2. Extract text → chunk → store in Qdrant
        3. Extract tables → store in PostgreSQL
        4. Extract visuals → analyze with vision LLM → store in PostgreSQL
        5. Update document status to 'completed' (or 'failed')

    Args:
        document_id: The document record ID.
        file_path: Path to the uploaded PDF.
        db: Async database session.
    """
    settings = get_settings()

    try:
        # Mark as processing
        await _update_status(db, document_id, "processing")
        logger.info(f"Starting ETL pipeline for document {document_id}")

        # --- Page count ---
        total_pages = get_page_count(file_path)
        await _update_page_count(db, document_id, total_pages)
        logger.info(f"Document has {total_pages} pages")

        # --- Step 1: Text extraction & chunking ---
        logger.info("Step 1: Extracting text...")
        pages = extract_text(file_path)

        logger.info("Step 2: Chunking text...")
        chunks = chunk_pages(pages, document_id)

        logger.info("Step 3: Storing chunks in Qdrant...")
        stored_count = await store_chunks(chunks, document_id)
        logger.info(f"Stored {stored_count} chunks in vector database")

        # --- Step 2: Table extraction ---
        logger.info("Step 4: Extracting tables...")
        tables = extract_tables(file_path)
        await _store_tables(db, document_id, tables)
        logger.info(f"Stored {len(tables)} tables in PostgreSQL")

        # --- Step 3: Visual extraction & analysis ---
        logger.info("Step 5: Extracting visual elements...")
        figures_dir = os.path.join(settings.upload_dir, "figures", document_id)
        visuals = extract_visuals(file_path, figures_dir)

        if visuals:
            logger.info("Step 6: Analyzing visual elements with vision LLM...")
            vision_provider = get_vision_provider()
            analyzed = await analyze_visuals_batch(visuals, vision_provider)
            await _store_visuals(db, document_id, analyzed)
            logger.info(f"Stored visual data from {len(analyzed)} elements")
        else:
            logger.info("No visual elements found to analyze")

        # --- Done ---
        await _update_status(db, document_id, "completed")
        logger.info(f"ETL pipeline completed for document {document_id}")

    except Exception as e:
        logger.error(f"ETL pipeline failed for document {document_id}: {e}")
        try:
            await _update_status(db, document_id, "failed")
        except Exception:
            logger.error("Failed to update document status to 'failed'")
        raise


async def _update_status(db: AsyncSession, document_id: str, status: str) -> None:
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc:
        doc.status = status
        await db.commit()


async def _update_page_count(db: AsyncSession, document_id: str, total_pages: int) -> None:
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc:
        doc.total_pages = total_pages
        await db.commit()


async def _store_tables(
    db: AsyncSession, document_id: str, tables: list[dict]
) -> None:
    """Store extracted tables and their rows in PostgreSQL."""
    for table_data in tables:
        doc_table = DocumentTable(
            document_id=document_id,
            table_name=table_data["table_name"],
            page_number=table_data["page_number"],
            table_description=f"Headers: {', '.join(table_data['headers'])}",
        )
        db.add(doc_table)
        await db.flush()  # Get the generated ID

        for row_idx, row_dict in enumerate(table_data["rows"]):
            for col_name, value in row_dict.items():
                table_row = TableRow(
                    table_id=doc_table.id,
                    row_index=row_idx,
                    column_name=col_name,
                    value=value,
                    page_number=table_data["page_number"],
                )
                db.add(table_row)

    await db.commit()


async def _store_visuals(
    db: AsyncSession, document_id: str, analyzed_visuals: list[dict]
) -> None:
    """Store analyzed visual elements and their data points in PostgreSQL."""
    for visual in analyzed_visuals:
        if not visual.get("is_data_visualization"):
            continue  # Skip non-data visuals

        doc_visual = DocumentVisual(
            document_id=document_id,
            visual_type=visual.get("visual_type", "unknown"),
            title=visual.get("title", ""),
            page_number=visual["page_number"],
        )
        db.add(doc_visual)
        await db.flush()

        for data_point in visual.get("data", []):
            vd = VisualData(
                visual_id=doc_visual.id,
                label=data_point.get("label", ""),
                value=str(data_point.get("value", "")),
                extra_metadata=None,
                page_number=visual["page_number"],
            )
            db.add(vd)

    await db.commit()
