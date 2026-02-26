"""AumOS Video Engine service entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_video_engine.settings import Settings

logger = get_logger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup and shutdown."""
    logger.info(
        "Starting aumos-video-engine",
        version="0.1.0",
        gpu_enabled=settings.gpu_enabled,
        privacy_engine_url=settings.privacy_engine_url,
    )

    # Initialize database connection pool
    init_database(settings.database)

    # TODO: Initialize Kafka publisher
    # TODO: Initialize Redis connection
    # TODO: Initialize MinIO client
    # TODO: Pre-warm SVD model if GPU_ENABLED

    yield

    logger.info("Shutting down aumos-video-engine")
    # TODO: Close Kafka, Redis, MinIO connections


app: FastAPI = create_app(
    service_name="aumos-video-engine",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        # HealthCheck(name="postgres", check_fn=check_db_health),
        # HealthCheck(name="kafka", check_fn=check_kafka_health),
        # HealthCheck(name="privacy-engine", check_fn=check_privacy_engine_health),
    ],
)

# Import and register all API routers
from aumos_video_engine.api.router import router  # noqa: E402

app.include_router(router, prefix="/api/v1")
