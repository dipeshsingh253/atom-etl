from fastapi import Request
from typing import Optional

from src.core.schemas import Meta, SuccessResponse, PaginationMeta

def success_response(
    data: dict,
    message: str = "Request successful",
    request: Optional[Request] = None,
    pagination: Optional[PaginationMeta] = None
) -> SuccessResponse:
    """Create a standardized success response with request ID."""
    meta = Meta(pagination=pagination)
    
    return SuccessResponse(
        data=data,
        message=message,
        meta=meta
    )