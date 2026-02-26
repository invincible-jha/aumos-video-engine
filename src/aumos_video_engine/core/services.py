"""Core business logic services for aumos-video-engine.

All services are framework-agnostic and depend only on Protocol interfaces,
making them independently testable without infrastructure.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from aumos_common.auth import TenantContext
from aumos_common.errors import NotFoundError, ValidationError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_video_engine.core.interfaces import (
    FrameGeneratorProtocol,
    PrivacyEnforcerProtocol,
    SceneComposerProtocol,
    TemporalEngineProtocol,
)
from aumos_video_engine.core.models import (
    JobStatus,
    JobType,
    SceneTemplate,
    VideoDomain,
    VideoGenerationJob,
)

logger = get_logger(__name__)


class GenerationService:
    """Orchestrates the end-to-end synthetic video generation pipeline.

    Coordinates frame generation, temporal coherence enforcement,
    privacy enforcement, and storage upload for a single job.
    """

    def __init__(
        self,
        frame_generator: FrameGeneratorProtocol,
        temporal_engine: TemporalEngineProtocol,
        privacy_enforcer: PrivacyEnforcerProtocol,
        job_repository: Any,
        storage_adapter: Any,
        event_publisher: EventPublisher,
        min_coherence_score: float = 0.7,
        coherence_window_frames: int = 8,
    ) -> None:
        """Initialize GenerationService with injected adapters.

        Args:
            frame_generator: Adapter for generating video frames (SVD or similar).
            temporal_engine: Adapter for coherence scoring and enforcement.
            privacy_enforcer: Adapter for per-frame PII redaction.
            job_repository: Repository for persisting VideoGenerationJob records.
            storage_adapter: MinIO/S3 adapter for uploading completed videos.
            event_publisher: Kafka publisher for lifecycle events.
            min_coherence_score: Minimum acceptable coherence score (0.0–1.0).
            coherence_window_frames: Sliding window size for coherence evaluation.
        """
        self._frame_generator = frame_generator
        self._temporal_engine = temporal_engine
        self._privacy_enforcer = privacy_enforcer
        self._job_repo = job_repository
        self._storage = storage_adapter
        self._publisher = event_publisher
        self._min_coherence_score = min_coherence_score
        self._coherence_window_frames = coherence_window_frames

    async def create_job(
        self,
        tenant: TenantContext,
        prompt: str,
        num_frames: int,
        fps: int,
        resolution: str,
        domain: VideoDomain,
        model_config: dict[str, Any],
        enforce_privacy: bool = True,
        scene_template_id: str | None = None,
    ) -> VideoGenerationJob:
        """Create and persist a new generation job record.

        Args:
            tenant: Current tenant context for RLS scoping.
            prompt: Text description for video content.
            num_frames: Number of frames to generate.
            fps: Target frames per second.
            resolution: Output resolution string (e.g., "1280x720").
            domain: Target generation domain.
            model_config: Model-specific parameters.
            enforce_privacy: Whether to apply per-frame privacy enforcement.
            scene_template_id: Optional scene template ID for scene_compose jobs.

        Returns:
            Newly created VideoGenerationJob with PENDING status.
        """
        width, height = self._parse_resolution(resolution)
        duration_seconds = Decimal(num_frames) / Decimal(fps)

        job = VideoGenerationJob(
            tenant_id=tenant.tenant_id,
            job_type=JobType.SCENE_COMPOSE if scene_template_id else JobType.GENERATE,
            status=JobStatus.PENDING,
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            resolution=resolution,
            duration_seconds=duration_seconds,
            domain=domain,
            model_config_json=model_config,
            scene_template_id=scene_template_id,
            privacy_enforced=False,
        )
        job = await self._job_repo.create(job)

        await self._publisher.publish(
            Topics.VIDEO_LIFECYCLE,
            {
                "event_type": "job_created",
                "tenant_id": str(tenant.tenant_id),
                "job_id": str(job.id),
                "domain": domain.value,
                "num_frames": num_frames,
            },
        )

        logger.info(
            "Video generation job created",
            job_id=str(job.id),
            tenant_id=str(tenant.tenant_id),
            domain=domain.value,
            num_frames=num_frames,
            fps=fps,
            resolution=resolution,
        )
        return job

    async def execute_job(
        self,
        job_id: uuid.UUID,
        tenant: TenantContext,
        reference_image: bytes | None = None,
    ) -> VideoGenerationJob:
        """Execute a video generation job end-to-end.

        Runs frame generation → temporal coherence → privacy enforcement → storage upload.
        Updates job status at each stage.

        Args:
            job_id: UUID of the VideoGenerationJob to execute.
            tenant: Tenant context for authorization and RLS.
            reference_image: Optional conditioning image bytes for SVD.

        Returns:
            Updated VideoGenerationJob with COMPLETED status and output_uri set.

        Raises:
            NotFoundError: If the job does not exist.
            ValidationError: If the job cannot be executed (wrong status or tenant).
        """
        job = await self._job_repo.get_by_id(job_id)
        if not job:
            raise NotFoundError(f"VideoGenerationJob {job_id} not found")
        if str(job.tenant_id) != str(tenant.tenant_id):
            raise ValidationError(f"Job {job_id} does not belong to tenant {tenant.tenant_id}")

        # Mark running
        job = await self._job_repo.update(job, {"status": JobStatus.RUNNING})
        await self._publisher.publish(
            Topics.VIDEO_LIFECYCLE,
            {"event_type": "job_started", "tenant_id": str(tenant.tenant_id), "job_id": str(job_id)},
        )

        try:
            width, height = self._parse_resolution(job.resolution)
            logger.info("Generating frames", job_id=str(job_id), num_frames=job.num_frames)

            # Step 1: Generate raw frames
            frames = await self._frame_generator.generate_frames(
                prompt=job.prompt or "",
                num_frames=job.num_frames,
                fps=job.fps,
                resolution=(width, height),
                model_config=job.model_config_json,
                reference_image=reference_image,
            )

            # Step 2: Temporal coherence enforcement
            coherence_score = self._temporal_engine.score_coherence(
                frames=frames,
                window_size=self._coherence_window_frames,
            )
            logger.info(
                "Coherence scored",
                job_id=str(job_id),
                score=coherence_score,
                min_required=self._min_coherence_score,
            )

            if coherence_score < self._min_coherence_score:
                frames = self._temporal_engine.enforce_coherence(
                    frames=frames,
                    min_score=self._min_coherence_score,
                    window_size=self._coherence_window_frames,
                )
                coherence_score = self._temporal_engine.score_coherence(
                    frames=frames,
                    window_size=self._coherence_window_frames,
                )

            # Step 3: Per-frame privacy enforcement
            privacy_enforced = False
            if job.privacy_enforced is False:  # default enforce
                frames, detection_counts = await self._privacy_enforcer.enforce_batch(
                    frames=frames,
                    blur_faces=True,
                    redact_plates=True,
                    remove_pii=False,
                )
                privacy_enforced = True
                logger.info(
                    "Privacy enforcement applied",
                    job_id=str(job_id),
                    detections=detection_counts,
                )

            # Step 4: Encode and upload video
            output_uri = await self._storage.upload_video(
                frames=frames,
                fps=job.fps,
                job_id=str(job_id),
                tenant_id=str(tenant.tenant_id),
            )

            # Step 5: Mark completed
            job = await self._job_repo.update(
                job,
                {
                    "status": JobStatus.COMPLETED,
                    "temporal_coherence_score": Decimal(str(round(coherence_score, 4))),
                    "privacy_enforced": privacy_enforced,
                    "output_uri": output_uri,
                },
            )

            await self._publisher.publish(
                Topics.VIDEO_LIFECYCLE,
                {
                    "event_type": "job_completed",
                    "tenant_id": str(tenant.tenant_id),
                    "job_id": str(job_id),
                    "output_uri": output_uri,
                    "coherence_score": coherence_score,
                },
            )

            logger.info(
                "Video generation completed",
                job_id=str(job_id),
                output_uri=output_uri,
                coherence_score=coherence_score,
            )
            return job

        except Exception as exc:
            logger.error("Video generation failed", job_id=str(job_id), error=str(exc))
            job = await self._job_repo.update(
                job,
                {"status": JobStatus.FAILED, "error_message": str(exc)},
            )
            await self._publisher.publish(
                Topics.VIDEO_LIFECYCLE,
                {
                    "event_type": "job_failed",
                    "tenant_id": str(tenant.tenant_id),
                    "job_id": str(job_id),
                    "error": str(exc),
                },
            )
            raise

    async def get_job(self, job_id: uuid.UUID, tenant: TenantContext) -> VideoGenerationJob:
        """Retrieve a job by ID for the given tenant.

        Args:
            job_id: UUID of the VideoGenerationJob.
            tenant: Current tenant context.

        Returns:
            The VideoGenerationJob record.

        Raises:
            NotFoundError: If job not found or belongs to a different tenant.
        """
        job = await self._job_repo.get_by_id(job_id)
        if not job or str(job.tenant_id) != str(tenant.tenant_id):
            raise NotFoundError(f"VideoGenerationJob {job_id} not found")
        return job

    @staticmethod
    def _parse_resolution(resolution: str) -> tuple[int, int]:
        """Parse resolution string WxH into (width, height).

        Args:
            resolution: Resolution string, e.g., "1280x720".

        Returns:
            Tuple (width, height) as integers.

        Raises:
            ValidationError: If the string format is invalid.
        """
        try:
            width_str, height_str = resolution.lower().split("x")
            return int(width_str), int(height_str)
        except (ValueError, AttributeError) as exc:
            raise ValidationError(
                f"Invalid resolution format '{resolution}'. Expected WIDTHxHEIGHT, e.g. 1280x720."
            ) from exc


class SceneCompositionService:
    """Orchestrates BlenderProc 3D scene composition for domain-specific video generation.

    Manages scene templates and delegates rendering to the SceneComposerProtocol
    adapter, producing frame sequences that are then passed to the GenerationService
    pipeline for coherence enforcement and privacy protection.
    """

    def __init__(
        self,
        scene_composer: SceneComposerProtocol,
        template_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialize SceneCompositionService.

        Args:
            scene_composer: Adapter implementing SceneComposerProtocol.
            template_repository: Repository for SceneTemplate ORM records.
            event_publisher: Kafka publisher for lifecycle events.
        """
        self._composer = scene_composer
        self._template_repo = template_repository
        self._publisher = event_publisher

    async def list_templates(
        self,
        tenant: TenantContext,
        domain: VideoDomain | None = None,
    ) -> list[SceneTemplate]:
        """List available scene templates for the given tenant.

        Returns both tenant-private templates and public (is_public=True) templates.

        Args:
            tenant: Current tenant context.
            domain: Optional domain filter.

        Returns:
            List of SceneTemplate records.
        """
        return await self._template_repo.list_for_tenant(
            tenant_id=tenant.tenant_id,
            domain=domain,
        )

    async def get_template(
        self,
        template_id: uuid.UUID,
        tenant: TenantContext,
    ) -> SceneTemplate:
        """Retrieve a scene template by ID.

        Args:
            template_id: UUID of the SceneTemplate.
            tenant: Current tenant context.

        Returns:
            The SceneTemplate record.

        Raises:
            NotFoundError: If template not found or not accessible.
        """
        template = await self._template_repo.get_accessible(
            template_id=template_id,
            tenant_id=tenant.tenant_id,
        )
        if not template:
            raise NotFoundError(f"SceneTemplate {template_id} not found")
        return template

    async def compose_from_template(
        self,
        template_id: uuid.UUID,
        tenant: TenantContext,
        num_frames: int,
        fps: int,
        resolution: tuple[int, int],
        overrides: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Render a scene from a template into video frames.

        Args:
            template_id: UUID of the SceneTemplate to use.
            tenant: Current tenant context.
            num_frames: Number of frames to render.
            fps: Target frames per second.
            resolution: Output resolution (width, height).
            overrides: Optional parameter overrides for scene_config and objects.

        Returns:
            List of rendered frames as numpy arrays (H, W, 3) RGB uint8.
        """
        template = await self.get_template(template_id, tenant)
        scene_config = {**template.scene_config, **(overrides or {})}

        logger.info(
            "Composing scene from template",
            template_id=str(template_id),
            tenant_id=str(tenant.tenant_id),
            domain=template.domain.value,
            num_frames=num_frames,
        )

        frames = await self._composer.compose_scene(
            scene_config=scene_config,
            objects=template.objects,
            num_frames=num_frames,
            fps=fps,
            resolution=resolution,
        )

        await self._publisher.publish(
            Topics.VIDEO_LIFECYCLE,
            {
                "event_type": "scene_composed",
                "tenant_id": str(tenant.tenant_id),
                "template_id": str(template_id),
                "num_frames": len(frames),
            },
        )
        return frames

    async def create_template(
        self,
        tenant: TenantContext,
        name: str,
        domain: VideoDomain,
        scene_config: dict[str, Any],
        objects: list[dict[str, Any]],
        description: str | None = None,
        is_public: bool = False,
    ) -> SceneTemplate:
        """Create and persist a new scene template.

        Args:
            tenant: Current tenant context (template will be tenant-scoped).
            name: Human-readable template name.
            domain: Target domain for this template.
            scene_config: BlenderProc scene configuration dict.
            objects: List of 3D object descriptors.
            description: Optional description.
            is_public: If True, make available to all tenants.

        Returns:
            Newly created SceneTemplate.
        """
        template = SceneTemplate(
            tenant_id=tenant.tenant_id,
            name=name,
            domain=domain,
            scene_config=scene_config,
            objects=objects,
            description=description,
            is_public=is_public,
        )
        template = await self._template_repo.create(template)
        logger.info(
            "Scene template created",
            template_id=str(template.id),
            name=name,
            domain=domain.value,
        )
        return template


class TemporalCoherenceService:
    """Standalone service for temporal coherence analysis and enforcement.

    Wraps the TemporalEngineProtocol adapter with logging and validation.
    Can be used independently of GenerationService for post-processing workflows.
    """

    def __init__(
        self,
        temporal_engine: TemporalEngineProtocol,
        min_coherence_score: float = 0.7,
        window_frames: int = 8,
    ) -> None:
        """Initialize TemporalCoherenceService.

        Args:
            temporal_engine: Adapter implementing TemporalEngineProtocol.
            min_coherence_score: Default minimum score threshold.
            window_frames: Default sliding window size.
        """
        self._engine = temporal_engine
        self._min_score = min_coherence_score
        self._window_frames = window_frames

    def score(self, frames: list[Any]) -> float:
        """Compute temporal coherence score for a frame sequence.

        Args:
            frames: List of RGB uint8 numpy arrays (H, W, 3).

        Returns:
            Coherence score in [0.0, 1.0].
        """
        score = self._engine.score_coherence(frames=frames, window_size=self._window_frames)
        logger.debug("Temporal coherence scored", score=score, num_frames=len(frames))
        return score

    def enforce(
        self,
        frames: list[Any],
        min_score: float | None = None,
    ) -> tuple[list[Any], float]:
        """Enforce temporal coherence on a frame sequence.

        Args:
            frames: Input frames as RGB uint8 numpy arrays.
            min_score: Override the minimum score threshold (uses service default if None).

        Returns:
            Tuple of (processed_frames, final_coherence_score).
        """
        threshold = min_score if min_score is not None else self._min_score
        current_score = self._engine.score_coherence(frames=frames, window_size=self._window_frames)

        if current_score >= threshold:
            logger.debug("Frames already meet coherence threshold", score=current_score)
            return frames, current_score

        logger.info(
            "Enforcing temporal coherence",
            current_score=current_score,
            min_required=threshold,
            num_frames=len(frames),
        )
        enforced_frames = self._engine.enforce_coherence(
            frames=frames,
            min_score=threshold,
            window_size=self._window_frames,
        )
        final_score = self._engine.score_coherence(
            frames=enforced_frames,
            window_size=self._window_frames,
        )
        logger.info("Coherence enforcement completed", final_score=final_score)
        return enforced_frames, final_score

    def synthesize_transition(
        self,
        start_frame: Any,
        end_frame: Any,
        num_intermediate: int,
    ) -> list[Any]:
        """Generate smooth transition frames between two keyframes.

        Args:
            start_frame: Starting RGB uint8 frame (H, W, 3).
            end_frame: Ending RGB uint8 frame (H, W, 3).
            num_intermediate: Number of frames to synthesize.

        Returns:
            List of intermediate frames.
        """
        return self._engine.synthesize_motion(
            start_frame=start_frame,
            end_frame=end_frame,
            num_intermediate=num_intermediate,
        )


class PrivacyEnforcementService:
    """Service for per-frame video privacy enforcement.

    Wraps the PrivacyEnforcerProtocol adapter with audit logging and
    configuration management. Supports both local enforcement and
    delegation to the remote privacy-engine service.
    """

    def __init__(
        self,
        privacy_enforcer: PrivacyEnforcerProtocol,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialize PrivacyEnforcementService.

        Args:
            privacy_enforcer: Adapter implementing PrivacyEnforcerProtocol.
            event_publisher: Kafka publisher for privacy enforcement audit events.
        """
        self._enforcer = privacy_enforcer
        self._publisher = event_publisher

    async def enforce_frames(
        self,
        frames: list[Any],
        tenant: TenantContext,
        job_id: str,
        blur_faces: bool = True,
        redact_plates: bool = True,
        remove_pii: bool = False,
    ) -> tuple[list[Any], dict[str, int]]:
        """Apply per-frame privacy enforcement to a video frame sequence.

        Args:
            frames: List of RGB uint8 numpy arrays (H, W, 3).
            tenant: Current tenant context for audit events.
            job_id: Job identifier for correlation.
            blur_faces: Whether to blur detected faces.
            redact_plates: Whether to redact license plates.
            remove_pii: Whether to remove other PII.

        Returns:
            Tuple of (processed_frames, detection_counts).
        """
        logger.info(
            "Applying privacy enforcement",
            job_id=job_id,
            tenant_id=str(tenant.tenant_id),
            num_frames=len(frames),
            blur_faces=blur_faces,
            redact_plates=redact_plates,
        )

        processed_frames, detection_counts = await self._enforcer.enforce_batch(
            frames=frames,
            blur_faces=blur_faces,
            redact_plates=redact_plates,
            remove_pii=remove_pii,
        )

        await self._publisher.publish(
            Topics.PRIVACY_AUDIT,
            {
                "event_type": "video_privacy_enforced",
                "tenant_id": str(tenant.tenant_id),
                "job_id": job_id,
                "num_frames": len(frames),
                "detection_counts": detection_counts,
                "blur_faces": blur_faces,
                "redact_plates": redact_plates,
            },
        )

        logger.info(
            "Privacy enforcement completed",
            job_id=job_id,
            detections=detection_counts,
        )
        return processed_frames, detection_counts


class BatchService:
    """Service for large-scale parallel video generation.

    Manages submission, tracking, and results aggregation for batch
    generation jobs across multiple videos.
    """

    def __init__(
        self,
        generation_service: GenerationService,
        job_repository: Any,
        event_publisher: EventPublisher,
        max_batch_size: int = 50,
    ) -> None:
        """Initialize BatchService.

        Args:
            generation_service: Service for individual video generation.
            job_repository: Repository for VideoGenerationJob records.
            event_publisher: Kafka publisher for batch lifecycle events.
            max_batch_size: Maximum number of videos per batch submission.
        """
        self._generation_service = generation_service
        self._job_repo = job_repository
        self._publisher = event_publisher
        self._max_batch_size = max_batch_size

    async def submit_batch(
        self,
        tenant: TenantContext,
        job_configs: list[dict[str, Any]],
    ) -> list[VideoGenerationJob]:
        """Submit a batch of video generation jobs.

        Creates individual job records for each config. Execution is
        handled asynchronously by background workers.

        Args:
            tenant: Current tenant context.
            job_configs: List of job configuration dicts. Each must contain:
                - prompt, num_frames, fps, resolution, domain, model_config.

        Returns:
            List of created VideoGenerationJob records (all PENDING status).

        Raises:
            ValidationError: If batch size exceeds max_batch_size.
        """
        if len(job_configs) > self._max_batch_size:
            raise ValidationError(
                f"Batch size {len(job_configs)} exceeds maximum of {self._max_batch_size}"
            )

        batch_id = str(uuid.uuid4())
        jobs: list[VideoGenerationJob] = []

        for config in job_configs:
            job = await self._generation_service.create_job(
                tenant=tenant,
                prompt=config.get("prompt", ""),
                num_frames=config.get("num_frames", 25),
                fps=config.get("fps", 24),
                resolution=config.get("resolution", "1280x720"),
                domain=VideoDomain(config.get("domain", "custom")),
                model_config=config.get("model_config", {}),
                enforce_privacy=config.get("enforce_privacy", True),
                scene_template_id=config.get("scene_template_id"),
            )
            jobs.append(job)

        await self._publisher.publish(
            Topics.VIDEO_LIFECYCLE,
            {
                "event_type": "batch_submitted",
                "tenant_id": str(tenant.tenant_id),
                "batch_id": batch_id,
                "num_jobs": len(jobs),
                "job_ids": [str(j.id) for j in jobs],
            },
        )

        logger.info(
            "Batch submitted",
            batch_id=batch_id,
            tenant_id=str(tenant.tenant_id),
            num_jobs=len(jobs),
        )
        return jobs

    async def get_batch_status(
        self,
        job_ids: list[uuid.UUID],
        tenant: TenantContext,
    ) -> dict[str, str]:
        """Retrieve status summary for a set of batch job IDs.

        Args:
            job_ids: List of VideoGenerationJob UUIDs.
            tenant: Current tenant context.

        Returns:
            Dict mapping job_id (str) to status (str).
        """
        status_map: dict[str, str] = {}
        for job_id in job_ids:
            try:
                job = await self._generation_service.get_job(job_id, tenant)
                status_map[str(job_id)] = job.status.value
            except NotFoundError:
                status_map[str(job_id)] = "not_found"
        return status_map
