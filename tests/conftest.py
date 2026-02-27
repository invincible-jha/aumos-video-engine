"""Shared pytest fixtures for aumos-video-engine tests.

Provides mock infrastructure adapters, tenant contexts, and test frames
used across all test modules.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from aumos_video_engine.core.models import (
    JobStatus,
    JobType,
    SceneTemplate,
    VideoDomain,
    VideoGenerationJob,
)


# ── Tenant / Auth fixtures ─────────────────────────────────────────


@pytest.fixture()
def tenant_id() -> uuid.UUID:
    """Return a stable tenant UUID for tests."""
    return uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


@pytest.fixture()
def tenant(tenant_id: uuid.UUID) -> MagicMock:
    """Return a mock TenantContext."""
    ctx = MagicMock()
    ctx.tenant_id = tenant_id
    return ctx


@pytest.fixture()
def other_tenant_id() -> uuid.UUID:
    """Return a different tenant UUID to test cross-tenant isolation."""
    return uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


@pytest.fixture()
def other_tenant(other_tenant_id: uuid.UUID) -> MagicMock:
    """Return a mock TenantContext for a second tenant."""
    ctx = MagicMock()
    ctx.tenant_id = other_tenant_id
    return ctx


# ── Frame fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def single_frame() -> np.ndarray:
    """Return a single 64x64 RGB uint8 test frame."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture()
def frame_sequence() -> list[np.ndarray]:
    """Return a sequence of 8 64x64 RGB uint8 test frames."""
    return [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(8)]


@pytest.fixture()
def static_frame_sequence() -> list[np.ndarray]:
    """Return a sequence of identical frames (maximum coherence)."""
    base = np.ones((64, 64, 3), dtype=np.uint8) * 128
    return [base.copy() for _ in range(8)]


# ── ORM model fixtures ─────────────────────────────────────────────


@pytest.fixture()
def job_id() -> uuid.UUID:
    """Return a stable job UUID."""
    return uuid.UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture()
def template_id() -> uuid.UUID:
    """Return a stable template UUID."""
    return uuid.UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture()
def pending_job(job_id: uuid.UUID, tenant_id: uuid.UUID) -> VideoGenerationJob:
    """Return a VideoGenerationJob in PENDING status."""
    job = MagicMock(spec=VideoGenerationJob)
    job.id = job_id
    job.tenant_id = tenant_id
    job.status = JobStatus.PENDING
    job.job_type = JobType.GENERATE
    job.domain = VideoDomain.MANUFACTURING
    job.prompt = "Assembly line defect detection"
    job.num_frames = 8
    job.fps = 24
    job.resolution = "640x480"
    job.duration_seconds = Decimal("0.333")
    job.temporal_coherence_score = None
    job.privacy_enforced = False
    job.output_uri = None
    job.error_message = None
    job.model_config_json = {"seed": 42, "guidance_scale": 3.0}
    job.scene_template_id = None
    job.created_at = MagicMock()
    job.created_at.isoformat.return_value = "2026-01-01T00:00:00"
    job.updated_at = MagicMock()
    job.updated_at.isoformat.return_value = "2026-01-01T00:01:00"
    return job


@pytest.fixture()
def completed_job(pending_job: VideoGenerationJob) -> VideoGenerationJob:
    """Return a VideoGenerationJob in COMPLETED status."""
    pending_job.status = JobStatus.COMPLETED
    pending_job.temporal_coherence_score = Decimal("0.8500")
    pending_job.privacy_enforced = True
    pending_job.output_uri = "s3://vid-bucket/tenant-aaa/11111111.mp4"
    return pending_job


@pytest.fixture()
def scene_template(template_id: uuid.UUID, tenant_id: uuid.UUID) -> SceneTemplate:
    """Return a mock SceneTemplate."""
    template = MagicMock(spec=SceneTemplate)
    template.id = template_id
    template.tenant_id = tenant_id
    template.name = "Manufacturing Floor Alpha"
    template.domain = VideoDomain.MANUFACTURING
    template.description = "Simulated assembly line with robot arm"
    template.scene_config = {
        "lighting": "overhead_industrial",
        "camera_path": "dolly_forward",
        "environment": "factory_floor",
    }
    template.objects = [
        {"id": "robot_arm_1", "model_path": "models/robot_arm.obj", "position": [0, 0, 0]},
    ]
    template.is_public = False
    template.version = 1
    template.created_at = MagicMock()
    template.created_at.isoformat.return_value = "2026-01-01T00:00:00"
    return template


# ── Adapter mock fixtures ──────────────────────────────────────────


@pytest.fixture()
def mock_frame_generator(frame_sequence: list[np.ndarray]) -> AsyncMock:
    """Return a mock FrameGeneratorProtocol."""
    generator = AsyncMock()
    generator.generate_frames = AsyncMock(return_value=frame_sequence)
    generator.is_available = AsyncMock(return_value=True)
    return generator


@pytest.fixture()
def mock_temporal_engine(frame_sequence: list[np.ndarray]) -> MagicMock:
    """Return a mock TemporalEngineProtocol that always returns coherent results."""
    engine = MagicMock()
    engine.score_coherence = MagicMock(return_value=0.85)
    engine.enforce_coherence = MagicMock(return_value=frame_sequence)
    engine.synthesize_motion = MagicMock(
        return_value=[np.ones((64, 64, 3), dtype=np.uint8) * 128]
    )
    return engine


@pytest.fixture()
def mock_privacy_enforcer(frame_sequence: list[np.ndarray]) -> AsyncMock:
    """Return a mock PrivacyEnforcerProtocol."""
    enforcer = AsyncMock()
    enforcer.enforce_frame = AsyncMock(
        return_value=(frame_sequence[0], {"faces": 1, "plates": 0})
    )
    enforcer.enforce_batch = AsyncMock(
        return_value=(frame_sequence, {"faces": 2, "plates": 1})
    )
    return enforcer


@pytest.fixture()
def mock_job_repository(pending_job: VideoGenerationJob) -> AsyncMock:
    """Return a mock JobRepository."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=pending_job)
    repo.get_by_id = AsyncMock(return_value=pending_job)
    repo.update = AsyncMock(return_value=pending_job)
    repo.list_by_status = AsyncMock(return_value=[pending_job])
    repo.list_by_domain = AsyncMock(return_value=[pending_job])
    repo.count_pending = AsyncMock(return_value=1)
    return repo


@pytest.fixture()
def mock_template_repository(scene_template: SceneTemplate) -> AsyncMock:
    """Return a mock SceneTemplateRepository."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=scene_template)
    repo.get_by_id = AsyncMock(return_value=scene_template)
    repo.get_accessible = AsyncMock(return_value=scene_template)
    repo.list_for_tenant = AsyncMock(return_value=[scene_template])
    return repo


@pytest.fixture()
def mock_storage_adapter() -> AsyncMock:
    """Return a mock VideoStorageAdapter."""
    adapter = AsyncMock()
    adapter.upload_video = AsyncMock(return_value="s3://vid-bucket/tenant-aaa/11111111.mp4")
    return adapter


@pytest.fixture()
def mock_event_publisher() -> AsyncMock:
    """Return a mock EventPublisher."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock(return_value=None)
    return publisher


# ── Service fixtures ───────────────────────────────────────────────


@pytest.fixture()
def generation_service(
    mock_frame_generator: AsyncMock,
    mock_temporal_engine: MagicMock,
    mock_privacy_enforcer: AsyncMock,
    mock_job_repository: AsyncMock,
    mock_storage_adapter: AsyncMock,
    mock_event_publisher: AsyncMock,
) -> Any:
    """Return a GenerationService with all mock adapters injected."""
    from aumos_video_engine.core.services import GenerationService

    return GenerationService(
        frame_generator=mock_frame_generator,
        temporal_engine=mock_temporal_engine,
        privacy_enforcer=mock_privacy_enforcer,
        job_repository=mock_job_repository,
        storage_adapter=mock_storage_adapter,
        event_publisher=mock_event_publisher,
        min_coherence_score=0.7,
        coherence_window_frames=4,
    )
