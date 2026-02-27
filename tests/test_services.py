"""Tests for core business logic services.

Covers GenerationService, SceneCompositionService, TemporalCoherenceService,
PrivacyEnforcementService, and BatchService.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest

from aumos_video_engine.core.models import (
    JobStatus,
    JobType,
    VideoDomain,
    VideoGenerationJob,
)
from aumos_video_engine.core.services import (
    BatchService,
    GenerationService,
    PrivacyEnforcementService,
    SceneCompositionService,
    TemporalCoherenceService,
)


# ── GenerationService.create_job ──────────────────────────────────


class TestGenerationServiceCreateJob:
    """Tests for GenerationService.create_job."""

    @pytest.mark.asyncio()
    async def test_create_job_returns_pending_job(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        pending_job: MagicMock,
    ) -> None:
        """create_job must return a VideoGenerationJob with PENDING status."""
        result = await generation_service.create_job(
            tenant=tenant,
            prompt="Assembly line",
            num_frames=8,
            fps=24,
            resolution="640x480",
            domain=VideoDomain.MANUFACTURING,
            model_config={"seed": 42},
        )
        assert result.status == JobStatus.PENDING

    @pytest.mark.asyncio()
    async def test_create_job_publishes_lifecycle_event(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """create_job must publish a job_created event to VIDEO_LIFECYCLE topic."""
        await generation_service.create_job(
            tenant=tenant,
            prompt="Traffic intersection",
            num_frames=25,
            fps=24,
            resolution="1280x720",
            domain=VideoDomain.TRAFFIC,
            model_config={},
        )
        mock_event_publisher.publish.assert_called_once()
        call_args = mock_event_publisher.publish.call_args
        payload = call_args[0][1]
        assert payload["event_type"] == "job_created"

    @pytest.mark.asyncio()
    async def test_create_job_sets_scene_compose_type_when_template_provided(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
        pending_job: MagicMock,
        template_id: uuid.UUID,
    ) -> None:
        """When scene_template_id is set, job_type must be SCENE_COMPOSE."""
        pending_job.job_type = JobType.SCENE_COMPOSE
        mock_job_repository.create = AsyncMock(return_value=pending_job)

        result = await generation_service.create_job(
            tenant=tenant,
            prompt="Scene",
            num_frames=8,
            fps=24,
            resolution="640x480",
            domain=VideoDomain.MANUFACTURING,
            model_config={},
            scene_template_id=str(template_id),
        )
        assert result.job_type == JobType.SCENE_COMPOSE

    @pytest.mark.asyncio()
    async def test_create_job_calculates_duration(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
        pending_job: MagicMock,
    ) -> None:
        """create_job must set duration_seconds = num_frames / fps."""
        pending_job.duration_seconds = Decimal("1.042")  # 25 / 24
        mock_job_repository.create = AsyncMock(return_value=pending_job)

        result = await generation_service.create_job(
            tenant=tenant,
            prompt="Test",
            num_frames=25,
            fps=24,
            resolution="1280x720",
            domain=VideoDomain.CUSTOM,
            model_config={},
        )
        assert result.duration_seconds is not None

    @pytest.mark.asyncio()
    async def test_create_job_invalid_resolution_raises(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
    ) -> None:
        """create_job must raise ValidationError for invalid resolution format."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError):
            await generation_service.create_job(
                tenant=tenant,
                prompt="Test",
                num_frames=25,
                fps=24,
                resolution="not-valid",
                domain=VideoDomain.CUSTOM,
                model_config={},
            )


# ── GenerationService.execute_job ─────────────────────────────────


class TestGenerationServiceExecuteJob:
    """Tests for GenerationService.execute_job."""

    @pytest.mark.asyncio()
    async def test_execute_job_completes_successfully(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
        pending_job: MagicMock,
        job_id: uuid.UUID,
    ) -> None:
        """execute_job must update status to COMPLETED and set output_uri."""
        completed = MagicMock()
        completed.status = JobStatus.COMPLETED
        completed.output_uri = "s3://vid-bucket/tenant-aaa/11111111.mp4"

        running = MagicMock()
        running.status = JobStatus.RUNNING
        running.tenant_id = tenant.tenant_id
        running.resolution = "640x480"
        running.num_frames = 8
        running.fps = 24
        running.prompt = "Test"
        running.model_config_json = {}
        running.privacy_enforced = False

        mock_job_repository.get_by_id = AsyncMock(return_value=pending_job)
        mock_job_repository.update = AsyncMock(side_effect=[running, completed])

        result = await generation_service.execute_job(job_id=job_id, tenant=tenant)
        assert result.status == JobStatus.COMPLETED

    @pytest.mark.asyncio()
    async def test_execute_job_raises_not_found_for_missing_job(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
    ) -> None:
        """execute_job must raise NotFoundError if job does not exist."""
        from aumos_common.errors import NotFoundError

        mock_job_repository.get_by_id = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await generation_service.execute_job(
                job_id=uuid.uuid4(), tenant=tenant
            )

    @pytest.mark.asyncio()
    async def test_execute_job_raises_validation_error_for_wrong_tenant(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        other_tenant: MagicMock,
        mock_job_repository: AsyncMock,
        pending_job: MagicMock,
        job_id: uuid.UUID,
    ) -> None:
        """execute_job must raise ValidationError if job belongs to different tenant."""
        from aumos_common.errors import ValidationError

        # pending_job.tenant_id belongs to `tenant`, not `other_tenant`
        mock_job_repository.get_by_id = AsyncMock(return_value=pending_job)

        with pytest.raises(ValidationError):
            await generation_service.execute_job(
                job_id=job_id, tenant=other_tenant
            )

    @pytest.mark.asyncio()
    async def test_execute_job_enforces_privacy_by_default(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
        mock_privacy_enforcer: AsyncMock,
        pending_job: MagicMock,
        job_id: uuid.UUID,
    ) -> None:
        """execute_job must call privacy enforcer when privacy_enforced is False."""
        running = MagicMock()
        running.status = JobStatus.RUNNING
        running.tenant_id = tenant.tenant_id
        running.resolution = "640x480"
        running.num_frames = 8
        running.fps = 24
        running.prompt = "Surveillance test"
        running.model_config_json = {}
        running.privacy_enforced = False

        completed = MagicMock()
        completed.status = JobStatus.COMPLETED

        mock_job_repository.get_by_id = AsyncMock(return_value=pending_job)
        mock_job_repository.update = AsyncMock(side_effect=[running, completed])

        await generation_service.execute_job(job_id=job_id, tenant=tenant)
        mock_privacy_enforcer.enforce_batch.assert_called_once()

    @pytest.mark.asyncio()
    async def test_execute_job_enforces_coherence_when_below_threshold(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
        mock_temporal_engine: MagicMock,
        pending_job: MagicMock,
        job_id: uuid.UUID,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """execute_job must call enforce_coherence when score is below threshold."""
        # First call returns low score, second call returns passing score
        mock_temporal_engine.score_coherence = MagicMock(side_effect=[0.5, 0.85])

        running = MagicMock()
        running.status = JobStatus.RUNNING
        running.tenant_id = tenant.tenant_id
        running.resolution = "640x480"
        running.num_frames = 8
        running.fps = 24
        running.prompt = "Test"
        running.model_config_json = {}
        running.privacy_enforced = False

        completed = MagicMock()
        completed.status = JobStatus.COMPLETED

        mock_job_repository.get_by_id = AsyncMock(return_value=pending_job)
        mock_job_repository.update = AsyncMock(side_effect=[running, completed])

        await generation_service.execute_job(job_id=job_id, tenant=tenant)
        mock_temporal_engine.enforce_coherence.assert_called_once()

    @pytest.mark.asyncio()
    async def test_execute_job_publishes_completed_event(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
        mock_event_publisher: AsyncMock,
        pending_job: MagicMock,
        job_id: uuid.UUID,
    ) -> None:
        """execute_job must publish job_completed event on success."""
        running = MagicMock()
        running.status = JobStatus.RUNNING
        running.tenant_id = tenant.tenant_id
        running.resolution = "640x480"
        running.num_frames = 8
        running.fps = 24
        running.prompt = "Test"
        running.model_config_json = {}
        running.privacy_enforced = False

        completed = MagicMock()
        completed.status = JobStatus.COMPLETED

        mock_job_repository.get_by_id = AsyncMock(return_value=pending_job)
        mock_job_repository.update = AsyncMock(side_effect=[running, completed])

        await generation_service.execute_job(job_id=job_id, tenant=tenant)

        published_event_types = [
            c[0][1]["event_type"]
            for c in mock_event_publisher.publish.call_args_list
        ]
        assert "job_completed" in published_event_types

    @pytest.mark.asyncio()
    async def test_execute_job_publishes_failed_event_on_error(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
        mock_frame_generator: AsyncMock,
        mock_event_publisher: AsyncMock,
        pending_job: MagicMock,
        job_id: uuid.UUID,
    ) -> None:
        """execute_job must publish job_failed event and re-raise on exception."""
        running = MagicMock()
        running.status = JobStatus.RUNNING
        running.tenant_id = tenant.tenant_id
        running.resolution = "640x480"
        running.num_frames = 8
        running.fps = 24
        running.prompt = "Test"
        running.model_config_json = {}
        running.privacy_enforced = False

        failed = MagicMock()
        failed.status = JobStatus.FAILED

        mock_job_repository.get_by_id = AsyncMock(return_value=pending_job)
        mock_job_repository.update = AsyncMock(side_effect=[running, failed])
        mock_frame_generator.generate_frames = AsyncMock(side_effect=RuntimeError("GPU OOM"))

        with pytest.raises(RuntimeError, match="GPU OOM"):
            await generation_service.execute_job(job_id=job_id, tenant=tenant)

        published_event_types = [
            c[0][1]["event_type"]
            for c in mock_event_publisher.publish.call_args_list
        ]
        assert "job_failed" in published_event_types


# ── GenerationService.get_job ──────────────────────────────────────


class TestGenerationServiceGetJob:
    """Tests for GenerationService.get_job."""

    @pytest.mark.asyncio()
    async def test_get_job_returns_owned_job(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        pending_job: MagicMock,
        mock_job_repository: AsyncMock,
        job_id: uuid.UUID,
    ) -> None:
        """get_job must return the job when it belongs to the tenant."""
        mock_job_repository.get_by_id = AsyncMock(return_value=pending_job)
        result = await generation_service.get_job(job_id, tenant)
        assert result is pending_job

    @pytest.mark.asyncio()
    async def test_get_job_raises_not_found_for_wrong_tenant(
        self,
        generation_service: GenerationService,
        other_tenant: MagicMock,
        pending_job: MagicMock,
        mock_job_repository: AsyncMock,
        job_id: uuid.UUID,
    ) -> None:
        """get_job must raise NotFoundError if job belongs to different tenant."""
        from aumos_common.errors import NotFoundError

        mock_job_repository.get_by_id = AsyncMock(return_value=pending_job)
        with pytest.raises(NotFoundError):
            await generation_service.get_job(job_id, other_tenant)

    @pytest.mark.asyncio()
    async def test_get_job_raises_not_found_when_missing(
        self,
        generation_service: GenerationService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
    ) -> None:
        """get_job must raise NotFoundError when job does not exist."""
        from aumos_common.errors import NotFoundError

        mock_job_repository.get_by_id = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await generation_service.get_job(uuid.uuid4(), tenant)


# ── GenerationService._parse_resolution ───────────────────────────


class TestParseResolution:
    """Tests for GenerationService._parse_resolution static method."""

    def test_parse_valid_resolution(self) -> None:
        """_parse_resolution must return (width, height) for valid input."""
        width, height = GenerationService._parse_resolution("1280x720")
        assert width == 1280
        assert height == 720

    def test_parse_small_resolution(self) -> None:
        """_parse_resolution must handle small resolutions like 64x64."""
        width, height = GenerationService._parse_resolution("64x64")
        assert width == 64
        assert height == 64

    def test_parse_case_insensitive(self) -> None:
        """_parse_resolution must handle uppercase X separator."""
        width, height = GenerationService._parse_resolution("1920X1080")
        assert width == 1920
        assert height == 1080

    def test_parse_invalid_format_raises(self) -> None:
        """_parse_resolution must raise ValidationError for non-WxH format."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError):
            GenerationService._parse_resolution("1920-1080")

    def test_parse_non_numeric_raises(self) -> None:
        """_parse_resolution must raise ValidationError for non-integer dimensions."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError):
            GenerationService._parse_resolution("widthxheight")


# ── TemporalCoherenceService ───────────────────────────────────────


class TestTemporalCoherenceService:
    """Tests for TemporalCoherenceService score/enforce methods."""

    @pytest.fixture()
    def coherence_service(self, mock_temporal_engine: MagicMock) -> TemporalCoherenceService:
        """Return a TemporalCoherenceService with mocked engine."""
        return TemporalCoherenceService(
            temporal_engine=mock_temporal_engine,
            min_coherence_score=0.7,
            window_frames=4,
        )

    def test_score_delegates_to_engine(
        self,
        coherence_service: TemporalCoherenceService,
        mock_temporal_engine: MagicMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """score() must delegate to the temporal engine's score_coherence."""
        score = coherence_service.score(frame_sequence)
        mock_temporal_engine.score_coherence.assert_called_once()
        assert score == 0.85

    def test_enforce_skips_when_score_meets_threshold(
        self,
        coherence_service: TemporalCoherenceService,
        mock_temporal_engine: MagicMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """enforce() must return frames unchanged when score >= threshold."""
        mock_temporal_engine.score_coherence = MagicMock(return_value=0.9)
        result_frames, result_score = coherence_service.enforce(frame_sequence)
        mock_temporal_engine.enforce_coherence.assert_not_called()
        assert result_score == 0.9

    def test_enforce_calls_engine_when_score_below_threshold(
        self,
        coherence_service: TemporalCoherenceService,
        mock_temporal_engine: MagicMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """enforce() must call enforce_coherence when score < threshold."""
        mock_temporal_engine.score_coherence = MagicMock(side_effect=[0.5, 0.8])
        result_frames, result_score = coherence_service.enforce(frame_sequence)
        mock_temporal_engine.enforce_coherence.assert_called_once()
        assert result_score == 0.8

    def test_enforce_respects_custom_min_score_override(
        self,
        coherence_service: TemporalCoherenceService,
        mock_temporal_engine: MagicMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """enforce() must use provided min_score override rather than service default."""
        mock_temporal_engine.score_coherence = MagicMock(return_value=0.6)
        # With min_score=0.5, score=0.6 should NOT trigger enforcement
        result_frames, result_score = coherence_service.enforce(
            frame_sequence, min_score=0.5
        )
        mock_temporal_engine.enforce_coherence.assert_not_called()

    def test_synthesize_transition_delegates_to_engine(
        self,
        coherence_service: TemporalCoherenceService,
        mock_temporal_engine: MagicMock,
        single_frame: np.ndarray,
    ) -> None:
        """synthesize_transition() must call synthesize_motion on the engine."""
        coherence_service.synthesize_transition(
            start_frame=single_frame,
            end_frame=single_frame,
            num_intermediate=3,
        )
        mock_temporal_engine.synthesize_motion.assert_called_once_with(
            start_frame=single_frame,
            end_frame=single_frame,
            num_intermediate=3,
        )


# ── PrivacyEnforcementService ──────────────────────────────────────


class TestPrivacyEnforcementService:
    """Tests for PrivacyEnforcementService.enforce_frames."""

    @pytest.fixture()
    def privacy_service(
        self,
        mock_privacy_enforcer: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> PrivacyEnforcementService:
        """Return a PrivacyEnforcementService with mocked dependencies."""
        return PrivacyEnforcementService(
            privacy_enforcer=mock_privacy_enforcer,
            event_publisher=mock_event_publisher,
        )

    @pytest.mark.asyncio()
    async def test_enforce_frames_returns_processed_frames_and_counts(
        self,
        privacy_service: PrivacyEnforcementService,
        tenant: MagicMock,
        frame_sequence: list[np.ndarray],
        mock_privacy_enforcer: AsyncMock,
    ) -> None:
        """enforce_frames must return (processed_frames, detection_counts)."""
        result_frames, counts = await privacy_service.enforce_frames(
            frames=frame_sequence,
            tenant=tenant,
            job_id="test-job-1",
        )
        assert isinstance(result_frames, list)
        assert isinstance(counts, dict)
        assert "faces" in counts

    @pytest.mark.asyncio()
    async def test_enforce_frames_publishes_privacy_audit_event(
        self,
        privacy_service: PrivacyEnforcementService,
        tenant: MagicMock,
        frame_sequence: list[np.ndarray],
        mock_event_publisher: AsyncMock,
    ) -> None:
        """enforce_frames must publish a video_privacy_enforced event."""
        await privacy_service.enforce_frames(
            frames=frame_sequence,
            tenant=tenant,
            job_id="test-job-2",
        )
        mock_event_publisher.publish.assert_called_once()
        payload = mock_event_publisher.publish.call_args[0][1]
        assert payload["event_type"] == "video_privacy_enforced"
        assert payload["job_id"] == "test-job-2"

    @pytest.mark.asyncio()
    async def test_enforce_frames_passes_blur_and_redact_flags(
        self,
        privacy_service: PrivacyEnforcementService,
        tenant: MagicMock,
        frame_sequence: list[np.ndarray],
        mock_privacy_enforcer: AsyncMock,
    ) -> None:
        """enforce_frames must pass blur_faces and redact_plates to the enforcer."""
        await privacy_service.enforce_frames(
            frames=frame_sequence,
            tenant=tenant,
            job_id="job-123",
            blur_faces=True,
            redact_plates=False,
            remove_pii=False,
        )
        mock_privacy_enforcer.enforce_batch.assert_called_once_with(
            frames=frame_sequence,
            blur_faces=True,
            redact_plates=False,
            remove_pii=False,
        )


# ── SceneCompositionService ────────────────────────────────────────


class TestSceneCompositionService:
    """Tests for SceneCompositionService template management and composition."""

    @pytest.fixture()
    def scene_service(
        self,
        mock_template_repository: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> SceneCompositionService:
        """Return a SceneCompositionService with mocked dependencies."""
        scene_composer = AsyncMock()
        scene_composer.compose_scene = AsyncMock(
            return_value=[np.ones((64, 64, 3), dtype=np.uint8) for _ in range(5)]
        )
        return SceneCompositionService(
            scene_composer=scene_composer,
            template_repository=mock_template_repository,
            event_publisher=mock_event_publisher,
        )

    @pytest.mark.asyncio()
    async def test_list_templates_returns_template_list(
        self,
        scene_service: SceneCompositionService,
        tenant: MagicMock,
        scene_template: MagicMock,
    ) -> None:
        """list_templates must return a list of SceneTemplate records."""
        result = await scene_service.list_templates(tenant=tenant)
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_list_templates_passes_domain_filter(
        self,
        scene_service: SceneCompositionService,
        tenant: MagicMock,
        mock_template_repository: AsyncMock,
    ) -> None:
        """list_templates must pass domain filter through to the repository."""
        await scene_service.list_templates(
            tenant=tenant, domain=VideoDomain.MANUFACTURING
        )
        mock_template_repository.list_for_tenant.assert_called_once_with(
            tenant_id=tenant.tenant_id,
            domain=VideoDomain.MANUFACTURING,
        )

    @pytest.mark.asyncio()
    async def test_get_template_returns_accessible_template(
        self,
        scene_service: SceneCompositionService,
        tenant: MagicMock,
        scene_template: MagicMock,
        template_id: uuid.UUID,
    ) -> None:
        """get_template must return the template when accessible to tenant."""
        result = await scene_service.get_template(
            template_id=template_id, tenant=tenant
        )
        assert result is scene_template

    @pytest.mark.asyncio()
    async def test_get_template_raises_not_found_when_inaccessible(
        self,
        scene_service: SceneCompositionService,
        tenant: MagicMock,
        mock_template_repository: AsyncMock,
        template_id: uuid.UUID,
    ) -> None:
        """get_template must raise NotFoundError when template is inaccessible."""
        from aumos_common.errors import NotFoundError

        mock_template_repository.get_accessible = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await scene_service.get_template(
                template_id=template_id, tenant=tenant
            )

    @pytest.mark.asyncio()
    async def test_create_template_persists_and_returns(
        self,
        scene_service: SceneCompositionService,
        tenant: MagicMock,
        scene_template: MagicMock,
    ) -> None:
        """create_template must create and return the new SceneTemplate."""
        result = await scene_service.create_template(
            tenant=tenant,
            name="Traffic Intersection",
            domain=VideoDomain.TRAFFIC,
            scene_config={"lighting": "daylight"},
            objects=[],
        )
        assert result is scene_template

    @pytest.mark.asyncio()
    async def test_compose_from_template_returns_frames(
        self,
        scene_service: SceneCompositionService,
        tenant: MagicMock,
        template_id: uuid.UUID,
    ) -> None:
        """compose_from_template must return a list of rendered frames."""
        frames = await scene_service.compose_from_template(
            template_id=template_id,
            tenant=tenant,
            num_frames=5,
            fps=24,
            resolution=(640, 480),
        )
        assert isinstance(frames, list)
        assert len(frames) == 5

    @pytest.mark.asyncio()
    async def test_compose_from_template_publishes_event(
        self,
        scene_service: SceneCompositionService,
        tenant: MagicMock,
        template_id: uuid.UUID,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """compose_from_template must publish a scene_composed lifecycle event."""
        await scene_service.compose_from_template(
            template_id=template_id,
            tenant=tenant,
            num_frames=5,
            fps=24,
            resolution=(640, 480),
        )
        mock_event_publisher.publish.assert_called_once()
        payload = mock_event_publisher.publish.call_args[0][1]
        assert payload["event_type"] == "scene_composed"


# ── BatchService ──────────────────────────────────────────────────


class TestBatchService:
    """Tests for BatchService batch submission and status retrieval."""

    @pytest.fixture()
    def batch_service(
        self,
        generation_service: GenerationService,
        mock_job_repository: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> BatchService:
        """Return a BatchService with mocked dependencies."""
        return BatchService(
            generation_service=generation_service,
            job_repository=mock_job_repository,
            event_publisher=mock_event_publisher,
            max_batch_size=5,
        )

    @pytest.mark.asyncio()
    async def test_submit_batch_creates_one_job_per_config(
        self,
        batch_service: BatchService,
        tenant: MagicMock,
    ) -> None:
        """submit_batch must create exactly one job per config entry."""
        configs = [
            {
                "prompt": f"Video {i}",
                "num_frames": 8,
                "fps": 24,
                "resolution": "640x480",
                "domain": "custom",
                "model_config": {},
            }
            for i in range(3)
        ]
        jobs = await batch_service.submit_batch(tenant=tenant, job_configs=configs)
        assert len(jobs) == 3

    @pytest.mark.asyncio()
    async def test_submit_batch_raises_validation_error_when_over_limit(
        self,
        batch_service: BatchService,
        tenant: MagicMock,
    ) -> None:
        """submit_batch must raise ValidationError when batch exceeds max_batch_size."""
        from aumos_common.errors import ValidationError

        configs = [
            {"prompt": f"Video {i}", "num_frames": 8, "fps": 24,
             "resolution": "640x480", "domain": "custom", "model_config": {}}
            for i in range(6)  # max_batch_size is 5
        ]
        with pytest.raises(ValidationError, match="exceeds maximum"):
            await batch_service.submit_batch(tenant=tenant, job_configs=configs)

    @pytest.mark.asyncio()
    async def test_submit_batch_publishes_batch_submitted_event(
        self,
        batch_service: BatchService,
        tenant: MagicMock,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """submit_batch must publish a batch_submitted lifecycle event."""
        await batch_service.submit_batch(
            tenant=tenant,
            job_configs=[
                {"prompt": "Test", "num_frames": 8, "fps": 24,
                 "resolution": "640x480", "domain": "custom", "model_config": {}}
            ],
        )
        published_events = [
            c[0][1]["event_type"]
            for c in mock_event_publisher.publish.call_args_list
        ]
        assert "batch_submitted" in published_events

    @pytest.mark.asyncio()
    async def test_get_batch_status_returns_status_map(
        self,
        batch_service: BatchService,
        tenant: MagicMock,
        pending_job: MagicMock,
        mock_job_repository: AsyncMock,
        job_id: uuid.UUID,
    ) -> None:
        """get_batch_status must return a dict mapping job_id str to status str."""
        status_map = await batch_service.get_batch_status(
            job_ids=[job_id], tenant=tenant
        )
        assert isinstance(status_map, dict)
        assert str(job_id) in status_map

    @pytest.mark.asyncio()
    async def test_get_batch_status_returns_not_found_for_missing_job(
        self,
        batch_service: BatchService,
        tenant: MagicMock,
        mock_job_repository: AsyncMock,
    ) -> None:
        """get_batch_status must return 'not_found' for non-existent job IDs."""
        from aumos_common.errors import NotFoundError

        mock_job_repository.get_by_id = AsyncMock(return_value=None)
        missing_id = uuid.uuid4()
        status_map = await batch_service.get_batch_status(
            job_ids=[missing_id], tenant=tenant
        )
        assert status_map[str(missing_id)] == "not_found"
