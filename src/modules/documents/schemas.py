from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Response returned when a document ingestion is started."""

    status: str = Field(..., description="Ingestion status")
    document_id: str = Field(..., description="Unique document identifier")


class DocumentStatusResponse(BaseModel):
    """Response for document status queries."""

    document_id: str = Field(
        ..., validation_alias="id", description="Unique document identifier"
    )
    status: str = Field(..., description="Current processing status")
    filename: str = Field(..., description="Original filename")
    total_pages: Optional[int] = Field(None, description="Number of pages in the document")
    created_at: datetime = Field(..., description="Timestamp when document was uploaded")
    updated_at: datetime = Field(..., description="Timestamp of last status update")

    model_config = {"from_attributes": True}
