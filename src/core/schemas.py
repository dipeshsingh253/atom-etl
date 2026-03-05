from pydantic import BaseModel, Field
from typing import Any, Optional, Dict
from datetime import datetime, timezone


class PaginationMeta(BaseModel):
    page: int
    per_page: int
    total: int
    total_pages: int


class Meta(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: Optional[str] = None
    pagination: Optional[PaginationMeta] = None


class BaseAPIResponse(BaseModel):
    meta: Meta = Field(default_factory=Meta)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseAPIResponse):
    success: bool = True
    data: Any
    message: str = "Request successful"


class APIErrorResponse(BaseAPIResponse):
    success: bool = False
    error: ErrorResponse