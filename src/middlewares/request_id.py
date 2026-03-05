import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from loguru import logger


async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to add a unique request ID to each request for tracking."""
    header_name = getattr(request_id_middleware, 'header_name', 'X-Request-ID')
    
    # Generate or get existing request ID
    request_id = request.headers.get(header_name) or str(uuid.uuid4())
    
    # Store request ID in request state for access in route handlers
    request.state.request_id = request_id
    
    # Log incoming request with request ID
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Request started")
    
    # Process the request
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers[header_name] = request_id
    
    # Log response with request ID
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Request completed (status: {response.status_code})")
    
    return response

def setup_request_id_middleware(app: FastAPI, header_name: str = "X-Request-ID") -> None:
    """Setup the request ID middleware for the FastAPI application."""
    request_id_middleware.header_name = header_name
    app.middleware("http")(request_id_middleware)
    logger.info("Request ID middleware configured successfully")