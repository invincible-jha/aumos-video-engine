"""FastAPI dependency injection for aumos-video-engine.

All service instances are constructed here with their adapter dependencies,
enabling clean separation of concerns and testability.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session
from aumos_common.events import EventPublisher

from aumos_video_engine.adapters.generators.stable_video_diffusion import (
    StableVideoDiffusionGenerator,
)
from aumos_video_engine.adapters.privacy_client import PrivacyEngineClient
from aumos_video_engine.adapters.privacy_enforcer import LocalPrivacyEnforcer
from aumos_video_engine.adapters.repositories import JobRepository, SceneTemplateRepository
from aumos_video_engine.adapters.storage import VideoStorageAdapter
from aumos_video_engine.adapters.temporal_engine import OpticalFlowTemporalEngine
from aumos_video_engine.core.services import (
    BatchService,
    GenerationService,
    PrivacyEnforcementService,
    SceneCompositionService,
    TemporalCoherenceService,
)
from aumos_video_engine.settings import Settings


@lru_cache
def get_settings() -> Settings:
    """Return cached service settings singleton."""
    return Settings()


async def get_event_publisher() -> EventPublisher:
    """Return a Kafka EventPublisher instance."""
    settings = get_settings()
    return EventPublisher(bootstrap_servers=settings.kafka_bootstrap_servers)


async def get_generation_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    publisher: Annotated[EventPublisher, Depends(get_event_publisher)],
) -> GenerationService:
    """Build and return a GenerationService with all injected adapters."""
    settings = get_settings()
    return GenerationService(
        frame_generator=StableVideoDiffusionGenerator(
            model_id=settings.svd_model_id,
            cache_dir=settings.svd_cache_dir,
            gpu_enabled=settings.gpu_enabled,
            cuda_device=settings.cuda_device,
        ),
        temporal_engine=OpticalFlowTemporalEngine(),
        privacy_enforcer=PrivacyEngineClient(
            base_url=settings.privacy_engine_url,
            fallback=LocalPrivacyEnforcer(),
        ),
        job_repository=JobRepository(session),
        storage_adapter=VideoStorageAdapter(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            storage_prefix=settings.storage_prefix,
        ),
        event_publisher=publisher,
        min_coherence_score=settings.temporal_coherence_min_score,
        coherence_window_frames=settings.temporal_coherence_window_frames,
    )


async def get_scene_composition_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    publisher: Annotated[EventPublisher, Depends(get_event_publisher)],
) -> SceneCompositionService:
    """Build and return a SceneCompositionService."""
    from aumos_video_engine.adapters.generators.blenderproc_scene import BlenderProcSceneComposer

    return SceneCompositionService(
        scene_composer=BlenderProcSceneComposer(),
        template_repository=SceneTemplateRepository(session),
        event_publisher=publisher,
    )


async def get_temporal_coherence_service() -> TemporalCoherenceService:
    """Build and return a TemporalCoherenceService."""
    settings = get_settings()
    return TemporalCoherenceService(
        temporal_engine=OpticalFlowTemporalEngine(),
        min_coherence_score=settings.temporal_coherence_min_score,
        window_frames=settings.temporal_coherence_window_frames,
    )


async def get_privacy_enforcement_service(
    publisher: Annotated[EventPublisher, Depends(get_event_publisher)],
) -> PrivacyEnforcementService:
    """Build and return a PrivacyEnforcementService."""
    settings = get_settings()
    return PrivacyEnforcementService(
        privacy_enforcer=PrivacyEngineClient(
            base_url=settings.privacy_engine_url,
            fallback=LocalPrivacyEnforcer(),
        ),
        event_publisher=publisher,
    )


async def get_batch_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    publisher: Annotated[EventPublisher, Depends(get_event_publisher)],
) -> BatchService:
    """Build and return a BatchService."""
    settings = get_settings()
    generation_service = await get_generation_service(session, publisher)
    return BatchService(
        generation_service=generation_service,
        job_repository=JobRepository(session),
        event_publisher=publisher,
        max_batch_size=settings.max_batch_size,
    )


# Type aliases for clean endpoint signatures
CurrentTenant = Annotated[TenantContext, Depends(get_current_tenant)]
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
