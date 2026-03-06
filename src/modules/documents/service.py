from typing import Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.documents.model import Document, DocumentTable


async def get_table_page_number(db: AsyncSession, table_name: str) -> int | None:
    """Look up the page number of a table by name.

    Args:
        db: Async database session.
        table_name: Partial or full table name to match (case-insensitive).

    Returns:
        Page number or None if not found.
    """
    result = await db.execute(
        select(DocumentTable.page_number)
        .where(DocumentTable.table_name.ilike(f"%{table_name}%"))
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return int(row) if row is not None else None


async def create_document(db: AsyncSession, filename: str) -> Document:
    """Create a new document record with status 'pending'.

    Args:
        db: Async database session.
        filename: Original uploaded filename.

    Returns:
        The created Document instance.
    """
    document = Document(
        filename=filename,
        status="pending",
        source_document=filename,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    logger.info(f"Created document record: {document.id} ({filename})")
    return document


async def get_document(db: AsyncSession, document_id: str) -> Optional[Document]:
    """Retrieve a document by ID.

    Args:
        db: Async database session.
        document_id: The document UUID.

    Returns:
        Document instance or None if not found.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    return result.scalar_one_or_none()


async def update_document_status(
    db: AsyncSession, document_id: str, status: str, total_pages: int | None = None
) -> Optional[Document]:
    """Update a document's processing status.

    Args:
        db: Async database session.
        document_id: The document UUID.
        status: New status value.
        total_pages: Optional page count to set.

    Returns:
        Updated Document instance or None.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document:
        document.status = status
        if total_pages is not None:
            document.total_pages = total_pages
        await db.commit()
        await db.refresh(document)
        logger.info(f"Document {document_id} status updated to '{status}'")
    return document
