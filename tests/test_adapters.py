"""Tests for repository and Kafka adapter implementations.

Covers JobRepository, SceneTemplateRepository, and VideoEventPublisher
with mocked SQLAlchemy sessions and Kafka publishers.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_video_engine.adapters.kafka import VideoEventPublisher
from aumos_video_engine.adapters.repositories import (
    JobRepository,
    SceneTemplateRepository,
)
from aumos_video_engine.core.models import JobStatus, VideoDomain


# ── JobRepository ──────────────────────────────────────────────────


class TestJobRepository:
    """Tests for JobRepository query methods."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Return a mock AsyncSession."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest.fixture()
    def job_repo(self, mock_session: AsyncMock) -> JobRepository:
        """Return a JobRepository with a mocked session."""
        return JobRepository(session=mock_session)

    @pytest.mark.asyncio()
    async def test_update_applies_field_changes(
        self,
        job_repo: JobRepository,
        mock_session: AsyncMock,
        pending_job: MagicMock,
    ) -> None:
        """update() must setattr each field in the updates dict."""
        updates = {"status": JobStatus.RUNNING, "error_message": None}
        mock_session.refresh = AsyncMock(side_effect=lambda obj: None)

        await job_repo.update(pending_job, updates)

        assert pending_job.status == JobStatus.RUNNING
        assert pending_job.error_message is None
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_list_by_status_executes_filtered_query(
        self,
        job_repo: JobRepository,
        mock_session: AsyncMock,
        pending_job: MagicMock,
    ) -> None:
        """list_by_status() must execute a SELECT filtered by status."""
        scalar_result = MagicMock()
        scalar_result.all.return_value = [pending_job]
        execute_result = MagicMock()
        execute_result.scalars.return_value = scalar_result
        mock_session.execute = AsyncMock(return_value=execute_result)

        result = await job_repo.list_by_status(status=JobStatus.PENDING, limit=10)
        assert result == [pending_job]
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_list_by_domain_executes_filtered_query(
        self,
        job_repo: JobRepository,
        mock_session: AsyncMock,
        pending_job: MagicMock,
    ) -> None:
        """list_by_domain() must execute a SELECT filtered by domain."""
        scalar_result = MagicMock()
        scalar_result.all.return_value = [pending_job]
        execute_result = MagicMock()
        execute_result.scalars.return_value = scalar_result
        mock_session.execute = AsyncMock(return_value=execute_result)

        result = await job_repo.list_by_domain(
            domain=VideoDomain.MANUFACTURING, limit=50, offset=0
        )
        assert result == [pending_job]

    @pytest.mark.asyncio()
    async def test_count_pending_returns_integer(
        self,
        job_repo: JobRepository,
        mock_session: AsyncMock,
    ) -> None:
        """count_pending() must return an integer count."""
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = 7
        mock_session.execute = AsyncMock(return_value=scalar_result)

        count = await job_repo.count_pending()
        assert count == 7

    @pytest.mark.asyncio()
    async def test_count_pending_returns_zero_when_none(
        self,
        job_repo: JobRepository,
        mock_session: AsyncMock,
    ) -> None:
        """count_pending() must return 0 when database returns None."""
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=scalar_result)

        count = await job_repo.count_pending()
        assert count == 0


# ── SceneTemplateRepository ────────────────────────────────────────


class TestSceneTemplateRepository:
    """Tests for SceneTemplateRepository tenant-scoped query methods."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Return a mock AsyncSession."""
        session = AsyncMock()
        return session

    @pytest.fixture()
    def template_repo(self, mock_session: AsyncMock) -> SceneTemplateRepository:
        """Return a SceneTemplateRepository with a mocked session."""
        return SceneTemplateRepository(session=mock_session)

    @pytest.mark.asyncio()
    async def test_list_for_tenant_returns_accessible_templates(
        self,
        template_repo: SceneTemplateRepository,
        mock_session: AsyncMock,
        scene_template: MagicMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """list_for_tenant() must return templates owned by or public to tenant."""
        scalar_result = MagicMock()
        scalar_result.all.return_value = [scene_template]
        execute_result = MagicMock()
        execute_result.scalars.return_value = scalar_result
        mock_session.execute = AsyncMock(return_value=execute_result)

        result = await template_repo.list_for_tenant(tenant_id=tenant_id)
        assert result == [scene_template]

    @pytest.mark.asyncio()
    async def test_list_for_tenant_with_domain_filter(
        self,
        template_repo: SceneTemplateRepository,
        mock_session: AsyncMock,
        scene_template: MagicMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """list_for_tenant() must apply domain filter when provided."""
        scalar_result = MagicMock()
        scalar_result.all.return_value = [scene_template]
        execute_result = MagicMock()
        execute_result.scalars.return_value = scalar_result
        mock_session.execute = AsyncMock(return_value=execute_result)

        result = await template_repo.list_for_tenant(
            tenant_id=tenant_id, domain=VideoDomain.MANUFACTURING
        )
        assert result == [scene_template]
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_accessible_returns_none_for_inaccessible(
        self,
        template_repo: SceneTemplateRepository,
        mock_session: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """get_accessible() must return None if template is not accessible."""
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=scalar_result)

        result = await template_repo.get_accessible(
            template_id=uuid.uuid4(), tenant_id=tenant_id
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_accessible_returns_template_when_accessible(
        self,
        template_repo: SceneTemplateRepository,
        mock_session: AsyncMock,
        scene_template: MagicMock,
        tenant_id: uuid.UUID,
        template_id: uuid.UUID,
    ) -> None:
        """get_accessible() must return the template when owned or public."""
        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = scene_template
        mock_session.execute = AsyncMock(return_value=scalar_result)

        result = await template_repo.get_accessible(
            template_id=template_id, tenant_id=tenant_id
        )
        assert result is scene_template


# ── VideoEventPublisher ────────────────────────────────────────────


class TestVideoEventPublisher:
    """Tests for VideoEventPublisher structured event helpers."""

    @pytest.fixture()
    def inner_publisher(self) -> AsyncMock:
        """Return a mock base EventPublisher."""
        publisher = AsyncMock()
        publisher.publish = AsyncMock(return_value=None)
        return publisher

    @pytest.fixture()
    def video_publisher(self, inner_publisher: AsyncMock) -> VideoEventPublisher:
        """Return a VideoEventPublisher wrapping the mock publisher."""
        return VideoEventPublisher(publisher=inner_publisher)

    @pytest.mark.asyncio()
    async def test_publish_job_created_uses_lifecycle_topic(
        self,
        video_publisher: VideoEventPublisher,
        inner_publisher: AsyncMock,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> None:
        """publish_job_created must publish to Topics.VIDEO_LIFECYCLE."""
        from aumos_common.events import Topics

        await video_publisher.publish_job_created(
            tenant_id=tenant_id,
            job_id=job_id,
            domain="manufacturing",
            num_frames=25,
            fps=24,
        )
        topic = inner_publisher.publish.call_args[0][0]
        assert topic == Topics.VIDEO_LIFECYCLE

    @pytest.mark.asyncio()
    async def test_publish_job_created_event_type(
        self,
        video_publisher: VideoEventPublisher,
        inner_publisher: AsyncMock,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> None:
        """publish_job_created event payload must have event_type='job_created'."""
        await video_publisher.publish_job_created(
            tenant_id=tenant_id,
            job_id=job_id,
            domain="traffic",
            num_frames=10,
            fps=30,
        )
        payload = inner_publisher.publish.call_args[0][1]
        assert payload["event_type"] == "job_created"
        assert payload["tenant_id"] == str(tenant_id)
        assert payload["job_id"] == str(job_id)

    @pytest.mark.asyncio()
    async def test_publish_job_completed_includes_output_uri(
        self,
        video_publisher: VideoEventPublisher,
        inner_publisher: AsyncMock,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> None:
        """publish_job_completed payload must include output_uri and coherence_score."""
        await video_publisher.publish_job_completed(
            tenant_id=tenant_id,
            job_id=job_id,
            output_uri="s3://bucket/video.mp4",
            coherence_score=0.88,
            privacy_enforced=True,
        )
        payload = inner_publisher.publish.call_args[0][1]
        assert payload["event_type"] == "job_completed"
        assert payload["output_uri"] == "s3://bucket/video.mp4"
        assert payload["coherence_score"] == 0.88
        assert payload["privacy_enforced"] is True

    @pytest.mark.asyncio()
    async def test_publish_job_failed_includes_error(
        self,
        video_publisher: VideoEventPublisher,
        inner_publisher: AsyncMock,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> None:
        """publish_job_failed payload must include the error string."""
        await video_publisher.publish_job_failed(
            tenant_id=tenant_id,
            job_id=job_id,
            error="CUDA out of memory",
        )
        payload = inner_publisher.publish.call_args[0][1]
        assert payload["event_type"] == "job_failed"
        assert payload["error"] == "CUDA out of memory"

    @pytest.mark.asyncio()
    async def test_publish_privacy_enforced_uses_privacy_audit_topic(
        self,
        video_publisher: VideoEventPublisher,
        inner_publisher: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """publish_privacy_enforced must publish to Topics.PRIVACY_AUDIT."""
        from aumos_common.events import Topics

        await video_publisher.publish_privacy_enforced(
            tenant_id=tenant_id,
            job_id="job-abc",
            num_frames=10,
            detection_counts={"faces": 3, "plates": 1},
        )
        topic = inner_publisher.publish.call_args[0][0]
        assert topic == Topics.PRIVACY_AUDIT

    def test_build_event_includes_correlation_id(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> None:
        """_build_event must include a correlation_id UUID field."""
        event = VideoEventPublisher._build_event(
            event_type="test_event",
            tenant_id=tenant_id,
            job_id=job_id,
        )
        assert "correlation_id" in event
        # Verify correlation_id is a valid UUID string
        uuid.UUID(event["correlation_id"])  # raises if not valid

    def test_build_event_merges_extra_fields(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> None:
        """_build_event must merge extra fields into the base payload."""
        event = VideoEventPublisher._build_event(
            event_type="job_created",
            tenant_id=tenant_id,
            job_id=job_id,
            extra={"domain": "manufacturing", "num_frames": 25},
        )
        assert event["domain"] == "manufacturing"
        assert event["num_frames"] == 25
