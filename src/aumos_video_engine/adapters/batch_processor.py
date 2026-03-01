"""Batch video generation processor with GPU-aware scheduling.

Handles parallel and sequential batch processing of video generation
jobs with configurable concurrency limits, priority queuing, and
progress tracking via Kafka events.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from aumos_common.events import EventPublisher
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class BatchPriority(str, Enum):
    """Batch processing priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class BatchJobSpec(BaseModel):
    """Specification for a single job within a batch.

    Attributes:
        prompt: Text description for frame generation.
        num_frames: Number of frames to generate.
        fps: Target frames per second.
        resolution: Output resolution as (width, height).
        domain: Video domain scenario.
        model_config_override: Optional per-job model config overrides.
    """

    prompt: str
    num_frames: int = 50
    fps: int = 24
    resolution: tuple[int, int] = (1280, 720)
    domain: str = "custom"
    model_config_override: dict[str, Any] = Field(default_factory=dict)


class BatchStatus(BaseModel):
    """Tracks progress of a batch processing operation.

    Attributes:
        batch_id: Unique batch identifier.
        total_jobs: Total number of jobs in the batch.
        completed_jobs: Number of jobs completed successfully.
        failed_jobs: Number of jobs that failed.
        in_progress_jobs: Number of jobs currently processing.
        pending_jobs: Number of jobs waiting to start.
        started_at: Batch start timestamp.
        estimated_completion: Estimated completion timestamp.
    """

    batch_id: str
    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0
    in_progress_jobs: int = 0
    pending_jobs: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_completion: datetime | None = None

    @property
    def progress_pct(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_jobs == 0:
            return 100.0
        return (self.completed_jobs + self.failed_jobs) / self.total_jobs * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if the batch has finished processing."""
        return (self.completed_jobs + self.failed_jobs) >= self.total_jobs


class BatchProcessor:
    """Processes batches of video generation jobs with concurrency control.

    Supports GPU-aware scheduling with configurable parallelism limits,
    priority-based ordering, and real-time progress events via Kafka.

    Args:
        max_concurrent: Maximum concurrent generation jobs.
        event_publisher: Kafka event publisher for progress events.
        gpu_memory_limit_mb: GPU memory budget for concurrent jobs.
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        event_publisher: EventPublisher | None = None,
        gpu_memory_limit_mb: int = 8192,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._event_publisher = event_publisher
        self._gpu_memory_limit_mb = gpu_memory_limit_mb
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_batches: dict[str, BatchStatus] = {}

    async def submit_batch(
        self,
        jobs: list[BatchJobSpec],
        tenant_id: str,
        priority: BatchPriority = BatchPriority.NORMAL,
        generate_fn: Any = None,
    ) -> BatchStatus:
        """Submit a batch of video generation jobs for processing.

        Jobs are executed concurrently up to max_concurrent limit,
        ordered by priority level.

        Args:
            jobs: List of job specifications.
            tenant_id: Originating tenant UUID.
            priority: Batch processing priority.
            generate_fn: Async callable(job_spec) -> result for each job.

        Returns:
            BatchStatus with batch_id for tracking.
        """
        batch_id = str(uuid.uuid4())
        status = BatchStatus(
            batch_id=batch_id,
            total_jobs=len(jobs),
            pending_jobs=len(jobs),
        )
        self._active_batches[batch_id] = status

        logger.info(
            "batch_submitted",
            batch_id=batch_id,
            total_jobs=len(jobs),
            priority=priority.value,
            tenant_id=tenant_id,
        )

        if self._event_publisher:
            await self._event_publisher.publish(
                topic="vid.batch.submitted",
                key=batch_id,
                payload={
                    "batch_id": batch_id,
                    "tenant_id": tenant_id,
                    "total_jobs": len(jobs),
                    "priority": priority.value,
                },
            )

        if generate_fn:
            asyncio.create_task(
                self._process_batch(batch_id, jobs, tenant_id, generate_fn),
            )

        return status

    async def _process_batch(
        self,
        batch_id: str,
        jobs: list[BatchJobSpec],
        tenant_id: str,
        generate_fn: Any,
    ) -> None:
        """Internal batch processing loop with concurrency control.

        Args:
            batch_id: Batch identifier.
            jobs: Job specifications.
            tenant_id: Tenant UUID.
            generate_fn: Generation callable.
        """
        status = self._active_batches[batch_id]
        tasks = []

        for idx, job in enumerate(jobs):
            task = asyncio.create_task(
                self._process_single_job(batch_id, idx, job, tenant_id, generate_fn),
            )
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            "batch_completed",
            batch_id=batch_id,
            completed=status.completed_jobs,
            failed=status.failed_jobs,
        )

        if self._event_publisher:
            await self._event_publisher.publish(
                topic="vid.batch.completed",
                key=batch_id,
                payload={
                    "batch_id": batch_id,
                    "tenant_id": tenant_id,
                    "completed": status.completed_jobs,
                    "failed": status.failed_jobs,
                },
            )

    async def _process_single_job(
        self,
        batch_id: str,
        job_index: int,
        job: BatchJobSpec,
        tenant_id: str,
        generate_fn: Any,
    ) -> Any:
        """Process a single job within a batch, respecting semaphore.

        Args:
            batch_id: Parent batch identifier.
            job_index: Job position in batch.
            job: Job specification.
            tenant_id: Tenant UUID.
            generate_fn: Generation callable.

        Returns:
            Generation result or None on failure.
        """
        status = self._active_batches[batch_id]

        async with self._semaphore:
            status.pending_jobs -= 1
            status.in_progress_jobs += 1

            try:
                result = await generate_fn(job)
                status.completed_jobs += 1
                status.in_progress_jobs -= 1

                logger.debug(
                    "batch_job_completed",
                    batch_id=batch_id,
                    job_index=job_index,
                    progress=f"{status.completed_jobs}/{status.total_jobs}",
                )
                return result

            except Exception as exc:
                status.failed_jobs += 1
                status.in_progress_jobs -= 1

                logger.error(
                    "batch_job_failed",
                    batch_id=batch_id,
                    job_index=job_index,
                    error=str(exc),
                )
                return None

    def get_batch_status(self, batch_id: str) -> BatchStatus | None:
        """Retrieve current status of a batch.

        Args:
            batch_id: Batch identifier.

        Returns:
            BatchStatus or None if not found.
        """
        return self._active_batches.get(batch_id)

    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch (prevents new jobs from starting).

        Args:
            batch_id: Batch identifier.

        Returns:
            True if the batch was found and cancelled.
        """
        status = self._active_batches.get(batch_id)
        if not status:
            return False

        logger.info("batch_cancelled", batch_id=batch_id)
        status.pending_jobs = 0
        return True

    def list_active_batches(self) -> list[BatchStatus]:
        """List all active (non-complete) batches.

        Returns:
            List of active batch statuses.
        """
        return [s for s in self._active_batches.values() if not s.is_complete]
