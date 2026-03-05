import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.utils import success_response
from src.db.session import get_db
from src.modules.documents import service
from src.modules.documents.schemas import DocumentStatusResponse, IngestResponse
from src.workers.tasks.ingestion_tasks import process_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", status_code=201)
async def ingest_document(
    request: Request,
    file: UploadFile,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Upload a PDF document and start the ingestion pipeline.

    The ingestion runs asynchronously via a background worker.
    Returns immediately with a document_id for status tracking.
    """
    settings = get_settings()

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save uploaded file
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Create document record to get the ID first
    document = await service.create_document(db, file.filename)

    # Save file with document_id as prefix
    file_path = os.path.join(settings.upload_dir, f"{document.id}_{file.filename}")
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Dispatch async ingestion task
    try:
        process_document.send(document.id, file_path)
        logger.info(f"Dispatched ingestion task for document {document.id}")
    except Exception as e:
        logger.warning(f"Failed to dispatch async task, will be retried: {e}")

    response_data = IngestResponse(
        status="ingestion_started",
        document_id=document.id,
    )
    return success_response(
        data=response_data.model_dump(),
        message="Document ingestion started",
        request=request,
    )


@router.get("/{document_id}/status")
async def get_document_status(
    request: Request,
    document_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get the processing status of a document."""
    document = await service.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    response_data = DocumentStatusResponse.model_validate(document)
    return success_response(
        data=response_data.model_dump(mode="json"),
        message="Document status retrieved",
        request=request,
    )
