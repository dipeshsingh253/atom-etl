from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.modules.documents.router import router as documents_router
from src.modules.query.router import router as query_router


def setup_routers() -> APIRouter:
    """
    Create and configure all application routers.

    Returns:
        APIRouter: Main router with all sub-routers included
    """
    settings = get_settings()

    # Create main router
    main_router = APIRouter()

    # Include module routers with API prefix
    api_router = APIRouter()
    api_router.include_router(documents_router, tags=["documents"])
    api_router.include_router(query_router, tags=["query"])

    # Include API routes with prefix
    main_router.include_router(api_router, prefix=settings.api_prefix)
    
    # Health check endpoint
    @main_router.get("/health", tags=["health"])
    async def health_check():
        return JSONResponse(
            content={
                "status": "healthy",
                "app_name": settings.app_name,
                "app_version": settings.app_version,
                "environment": settings.environment
            }
        )
    
    # Root endpoint
    @main_router.get("/", tags=["root"])
    async def root():
        response_content = {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "health_url": "/health"
        }
        
        # Only include docs URLs in development environment
        if settings.environment == "development":
            response_content.update({
                "docs_url": "/docs",
                "redoc_url": "/redoc"
            })
        
        return JSONResponse(content=response_content)
    
    return main_router