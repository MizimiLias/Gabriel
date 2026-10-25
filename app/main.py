"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.configs.database import dispose_engine
from app.configs.logging import configure_logging
from app.configs.settings import get_settings
from app.controllers.chat_controller import router as chat_router
from app.controllers.health_controller import router as health_router
from app.exceptions.handlers import register_exception_handlers
from app.integrations.registry import build_connector_manager
from app.middleware import RequestContextMiddleware


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage application-level resource lifecycles."""

    connector_manager = build_connector_manager(get_settings())
    await connector_manager.start()
    try:
        yield
    finally:
        await connector_manager.stop()
        await dispose_engine()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()
    configure_logging(settings.app_log_level)

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url=settings.docs_url,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(RequestContextMiddleware)
    register_exception_handlers(application)
    application.include_router(chat_router)
    application.include_router(health_router)
    return application


app = create_app()
