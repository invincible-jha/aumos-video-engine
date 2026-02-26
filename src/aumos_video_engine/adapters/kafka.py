"""Kafka event publisher for aumos-video-engine lifecycle events.

Extends the aumos-common EventPublisher with video-engine-specific
event schema helpers and topic routing.
"""

from __future__ import annotations

import uuid
from typing import Any

from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class VideoEventPublisher:
    """Publisher for video generation lifecycle events.

    Wraps aumos-common EventPublisher with structured event builders
    for the VIDEO_LIFECYCLE and PRIVACY_AUDIT topics.
    """

    def __init__(self, publisher: EventPublisher) -> None:
        """Initialize VideoEventPublisher.

        Args:
            publisher: aumos-common EventPublisher instance.
        """
        self._publisher = publisher

    async def publish_job_created(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        domain: str,
        num_frames: int,
        fps: int,
    ) -> None:
        """Publish a job_created lifecycle event.

        Args:
            tenant_id: Tenant UUID.
            job_id: New job UUID.
            domain: Video domain name.
            num_frames: Number of frames for this job.
            fps: Target frames per second.
        """
        await self._publisher.publish(
            Topics.VIDEO_LIFECYCLE,
            self._build_event(
                event_type="job_created",
                tenant_id=tenant_id,
                job_id=job_id,
                extra={"domain": domain, "num_frames": num_frames, "fps": fps},
            ),
        )
        logger.debug("Published job_created event", job_id=str(job_id))

    async def publish_job_completed(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        output_uri: str,
        coherence_score: float,
        privacy_enforced: bool,
    ) -> None:
        """Publish a job_completed lifecycle event.

        Args:
            tenant_id: Tenant UUID.
            job_id: Completed job UUID.
            output_uri: MinIO/S3 URI of the output video.
            coherence_score: Final temporal coherence score.
            privacy_enforced: Whether privacy enforcement was applied.
        """
        await self._publisher.publish(
            Topics.VIDEO_LIFECYCLE,
            self._build_event(
                event_type="job_completed",
                tenant_id=tenant_id,
                job_id=job_id,
                extra={
                    "output_uri": output_uri,
                    "coherence_score": coherence_score,
                    "privacy_enforced": privacy_enforced,
                },
            ),
        )
        logger.debug("Published job_completed event", job_id=str(job_id))

    async def publish_job_failed(
        self,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        error: str,
    ) -> None:
        """Publish a job_failed lifecycle event.

        Args:
            tenant_id: Tenant UUID.
            job_id: Failed job UUID.
            error: Error description.
        """
        await self._publisher.publish(
            Topics.VIDEO_LIFECYCLE,
            self._build_event(
                event_type="job_failed",
                tenant_id=tenant_id,
                job_id=job_id,
                extra={"error": error},
            ),
        )
        logger.warning("Published job_failed event", job_id=str(job_id), error=error)

    async def publish_privacy_enforced(
        self,
        tenant_id: uuid.UUID,
        job_id: str,
        num_frames: int,
        detection_counts: dict[str, int],
    ) -> None:
        """Publish a video_privacy_enforced audit event.

        Args:
            tenant_id: Tenant UUID.
            job_id: Job identifier string.
            num_frames: Number of frames processed.
            detection_counts: Detection counts per entity type.
        """
        await self._publisher.publish(
            Topics.PRIVACY_AUDIT,
            {
                "event_type": "video_privacy_enforced",
                "tenant_id": str(tenant_id),
                "job_id": job_id,
                "num_frames": num_frames,
                "detection_counts": detection_counts,
            },
        )

    @staticmethod
    def _build_event(
        event_type: str,
        tenant_id: uuid.UUID,
        job_id: uuid.UUID,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a standardized lifecycle event payload.

        Args:
            event_type: Event type string.
            tenant_id: Tenant UUID.
            job_id: Job UUID.
            extra: Additional fields to merge into event.

        Returns:
            Event payload dict.
        """
        event: dict[str, Any] = {
            "event_type": event_type,
            "tenant_id": str(tenant_id),
            "job_id": str(job_id),
            "correlation_id": str(uuid.uuid4()),
        }
        if extra:
            event.update(extra)
        return event
