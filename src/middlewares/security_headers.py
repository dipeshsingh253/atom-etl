from typing import Callable

from fastapi import FastAPI, Request, Response


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    response = await call_next(request)
    
    # Skip security headers for documentation routes
    docs_routes = ["/docs", "/redoc", "/openapi.json"]
    if any(request.url.path.startswith(route) for route in docs_routes):
        return response
    
    # Add security headers for other routes
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=()"
    )
    
    # Content Security Policy (basic example)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-src 'none'; "
        "object-src 'none'"
    )
    
    return response


def setup_security_headers(app: FastAPI) -> None:
    app.middleware("http")(security_headers_middleware)