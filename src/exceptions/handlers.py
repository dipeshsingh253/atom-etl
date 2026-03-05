from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from loguru import logger

from src.core.schemas import APIErrorResponse, ErrorResponse, Meta

# TODO: I forgot why I added this. Revisit later.
def get_request_id(request: Request) -> str:
    """Get request ID from request state if available."""
    return getattr(request.state, 'request_id', None)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException and return consistent error format."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    error_response = APIErrorResponse(
        error=ErrorResponse(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            details={"status_code": exc.status_code}
        ),
        meta=Meta(request_id=get_request_id(request))
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle Starlette HTTP exceptions."""
    logger.error(f"Starlette HTTP Exception: {exc.status_code} - {exc.detail}")
    
    error_response = APIErrorResponse(
        error=ErrorResponse(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            details={"status_code": exc.status_code}
        ),
        meta=Meta(request_id=get_request_id(request))
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors and return consistent format."""
    logger.error(f"Validation Error: {exc.errors()}")
    
    # Format validation errors
    validation_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        validation_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    error_response = APIErrorResponse(
        error=ErrorResponse(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details={
                "validation_errors": validation_errors,
                "total_errors": len(validation_errors)
            }
        ),
        meta=Meta(request_id=get_request_id(request))
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {str(exc)}")
    
    error_response = APIErrorResponse(
        error=ErrorResponse(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details={"error_type": type(exc).__name__}
        ),
        meta=Meta(request_id=get_request_id(request))
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers for the FastAPI application."""
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(StarletteHTTPException)(starlette_http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    app.exception_handler(Exception)(general_exception_handler)