"""SQLAlchemy repositories for aumos-video-engine ORM models.

Extends BaseRepository from aumos-common with video-engine-specific
query methods. All tenant isolation is handled by RLS set at session time.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import BaseRepository
from aumos_common.observability import get_logger

from aumos_video_engine.core.models import (
    JobStatus,
    SceneTemplate,
    VideoDomain,
    VideoGenerationJob,
)

logger = get_logger(__name__)


class JobRepository(BaseRepository[VideoGenerationJob]):
    """Repository for VideoGenerationJob records.

    RLS is enforced at the session level by aumos-common — all queries
    automatically filter to the current tenant's rows.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize JobRepository.

        Args:
            session: Async SQLAlchemy session with RLS tenant context set.
        """
        super().__init__(session, VideoGenerationJob)

    async def list_by_status(
        self,
        status: JobStatus,
        limit: int = 100,
    ) -> list[VideoGenerationJob]:
        """List jobs filtered by status.

        Args:
            status: Job status to filter on.
            limit: Maximum number of results to return.

        Returns:
            List of VideoGenerationJob records.
        """
        result = await self._session.execute(
            select(VideoGenerationJob)
            .where(VideoGenerationJob.status == status)
            .order_by(VideoGenerationJob.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_domain(
        self,
        domain: VideoDomain,
        limit: int = 100,
        offset: int = 0,
    ) -> list[VideoGenerationJob]:
        """List jobs filtered by domain.

        Args:
            domain: Video domain to filter on.
            limit: Maximum records to return.
            offset: Number of records to skip.

        Returns:
            List of VideoGenerationJob records.
        """
        result = await self._session.execute(
            select(VideoGenerationJob)
            .where(VideoGenerationJob.domain == domain)
            .order_by(VideoGenerationJob.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update(
        self,
        job: VideoGenerationJob,
        updates: dict[str, Any],
    ) -> VideoGenerationJob:
        """Apply field updates to a VideoGenerationJob.

        Args:
            job: The ORM instance to update.
            updates: Dict of field names to new values.

        Returns:
            Updated VideoGenerationJob instance.
        """
        for field, value in updates.items():
            setattr(job, field, value)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def count_pending(self) -> int:
        """Count PENDING jobs for the current tenant.

        Returns:
            Number of pending jobs.
        """
        from sqlalchemy import func

        result = await self._session.execute(
            select(func.count(VideoGenerationJob.id)).where(
                VideoGenerationJob.status == JobStatus.PENDING
            )
        )
        count = result.scalar_one_or_none()
        return int(count) if count is not None else 0


class SceneTemplateRepository(BaseRepository[SceneTemplate]):
    """Repository for SceneTemplate records.

    Supports both tenant-private and public (is_public=True) template access.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize SceneTemplateRepository.

        Args:
            session: Async SQLAlchemy session with RLS tenant context set.
        """
        super().__init__(session, SceneTemplate)

    async def list_for_tenant(
        self,
        tenant_id: uuid.UUID,
        domain: VideoDomain | None = None,
    ) -> list[SceneTemplate]:
        """List templates accessible to a specific tenant.

        Returns both tenant-private templates (matching tenant_id) and
        public templates (is_public=True) visible to all tenants.

        Args:
            tenant_id: The requesting tenant's UUID.
            domain: Optional domain filter.

        Returns:
            List of accessible SceneTemplate records.
        """
        from sqlalchemy import or_

        conditions = [
            or_(
                SceneTemplate.tenant_id == tenant_id,
                SceneTemplate.is_public.is_(True),
            )
        ]
        if domain is not None:
            conditions.append(SceneTemplate.domain == domain)

        result = await self._session.execute(
            select(SceneTemplate)
            .where(*conditions)
            .order_by(SceneTemplate.name.asc())
        )
        return list(result.scalars().all())

    async def get_accessible(
        self,
        template_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> SceneTemplate | None:
        """Get a template if accessible to the given tenant.

        A template is accessible if it belongs to the tenant OR is public.

        Args:
            template_id: UUID of the SceneTemplate.
            tenant_id: UUID of the requesting tenant.

        Returns:
            SceneTemplate if accessible, None otherwise.
        """
        from sqlalchemy import or_

        result = await self._session.execute(
            select(SceneTemplate).where(
                SceneTemplate.id == template_id,
                or_(
                    SceneTemplate.tenant_id == tenant_id,
                    SceneTemplate.is_public.is_(True),
                ),
            )
        )
        return result.scalar_one_or_none()
