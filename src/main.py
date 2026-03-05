from fastapi import FastAPI
from loguru import logger

from src.core.config import get_settings
from src.core.events import lifespan
from src.core.logging import setup_logging
from src.core.router import setup_routers
from src.exceptions.handlers import setup_exception_handlers
from src.middlewares.cors import setup_cors
from src.middlewares.request_id import setup_request_id_middleware
from src.middlewares.security_headers import setup_security_headers
from src.workers.broker import setup_dramatiq
from src.core.constants import ENV_DEVELOPMENT


def create_app() -> FastAPI:
    settings = get_settings()
    
    # Setup logging first
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Create FastAPI application
    # Only enable docs in development environment
    docs_url = "/docs" if settings.environment == ENV_DEVELOPMENT else None
    redoc_url = "/redoc" if settings.environment == ENV_DEVELOPMENT else None
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "none",
            "operationsSorter": "method",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
        } if settings.environment == ENV_DEVELOPMENT else None
    )
    
    # Setup Dramatiq broker
    setup_dramatiq()
    
    # Setup middlewares
    setup_request_id_middleware(app)  # Add first for request tracking
    setup_cors(app)
    setup_security_headers(app)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Setup all routers
    main_router = setup_routers()
    app.include_router(main_router)

    # TODO: Need to implement more effecient logging strategy
    logger.info("Application setup completed successfully")
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
        log_config=None  # Disable uvicorn's default log config
    )