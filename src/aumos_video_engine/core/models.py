"""SQLAlchemy ORM models for aumos-video-engine.

Table prefix: vid_
All models extend AumOSModel for id, tenant_id, created_at, updated_at.
"""

from __future__ import annotations

import enum
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, Column, Enum, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from aumos_common.database import AumOSModel, TenantMixin


class JobType(str, enum.Enum):
    """Type of video generation operation."""

    GENERATE = "generate"
    SCENE_COMPOSE = "scene_compose"
    PRIVACY_ENFORCE = "privacy_enforce"


class JobStatus(str, enum.Enum):
    """Lifecycle status for a video generation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoDomain(str, enum.Enum):
    """Domain scenario for video generation."""

    MANUFACTURING = "manufacturing"
    SURVEILLANCE = "surveillance"
    TRAFFIC = "traffic"
    CUSTOM = "custom"


class VideoGenerationJob(AumOSModel, TenantMixin):
    """Persistent record of a video generation job.

    Each job tracks the full lifecycle from submission through completion,
    including temporal coherence scoring and privacy enforcement status.

    Table: vid_generation_jobs
    """

    __tablename__ = "vid_generation_jobs"

    # Job classification
    job_type: Column[JobType] = Column(
        Enum(JobType, name="vid_job_type"),
        nullable=False,
        default=JobType.GENERATE,
        comment="Type of generation operation",
    )
    status: Column[JobStatus] = Column(
        Enum(JobStatus, name="vid_job_status"),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
        comment="Current job lifecycle status",
    )

    # Model configuration
    model_config_json: Column[dict[str, Any]] = Column(
        "model_config",
        JSONB,
        nullable=False,
        default=dict,
        comment="Model parameters (seed, guidance_scale, num_inference_steps, etc.)",
    )

    # Video properties
    num_frames: Column[int] = Column(
        Integer,
        nullable=False,
        default=25,
        comment="Number of frames to generate",
    )
    fps: Column[int] = Column(
        Integer,
        nullable=False,
        default=24,
        comment="Frames per second for output video",
    )
    resolution: Column[str] = Column(
        String(32),
        nullable=False,
        default="1280x720",
        comment="Output resolution as WIDTHxHEIGHT string",
    )
    duration_seconds: Column[Decimal | None] = Column(
        Numeric(precision=8, scale=3),
        nullable=True,
        comment="Computed duration (num_frames / fps)",
    )

    # Quality metrics
    temporal_coherence_score: Column[Decimal | None] = Column(
        Numeric(precision=5, scale=4),
        nullable=True,
        comment="Frame-to-frame coherence score 0.0–1.0 (higher is better)",
    )
    privacy_enforced: Column[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether per-frame privacy enforcement has been applied",
    )

    # Domain and output
    domain: Column[VideoDomain] = Column(
        Enum(VideoDomain, name="vid_domain"),
        nullable=False,
        default=VideoDomain.CUSTOM,
        index=True,
        comment="Target domain scenario",
    )
    prompt: Column[str | None] = Column(
        Text,
        nullable=True,
        comment="Text prompt used for generation",
    )
    scene_template_id: Column[str | None] = Column(
        String(36),
        ForeignKey("vid_scene_templates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Reference to scene template if scene_compose job type",
    )
    output_uri: Column[str | None] = Column(
        String(1024),
        nullable=True,
        comment="MinIO/S3 URI of the completed video artifact",
    )
    annotation_uri: Column[str | None] = Column(
        String(1024),
        nullable=True,
        comment="MinIO/S3 URI for COCO/YOLO annotation archive (GAP-90)",
    )
    annotation_format: Column[str | None] = Column(
        String(20),
        nullable=True,
        comment="Annotation format exported: 'coco', 'yolo', or None if not exported",
    )
    error_message: Column[str | None] = Column(
        Text,
        nullable=True,
        comment="Error details if status=failed",
    )

    # Relationships
    scene_template = relationship("SceneTemplate", back_populates="generation_jobs", lazy="select")

    def __repr__(self) -> str:
        return f"<VideoGenerationJob id={self.id} status={self.status} domain={self.domain}>"


class SceneTemplate(AumOSModel, TenantMixin):
    """Reusable 3D scene configuration for BlenderProc scene composition.

    Templates encode scene layout, object placement, lighting, and camera
    configurations for domain-specific synthetic video generation.

    Table: vid_scene_templates
    """

    __tablename__ = "vid_scene_templates"

    # Identity
    name: Column[str] = Column(
        String(255),
        nullable=False,
        index=True,
        comment="Human-readable template name",
    )
    domain: Column[VideoDomain] = Column(
        Enum(VideoDomain, name="vid_domain"),
        nullable=False,
        index=True,
        comment="Domain this template targets",
    )
    description: Column[str | None] = Column(
        Text,
        nullable=True,
        comment="Optional description of the scene scenario",
    )

    # Configuration
    scene_config: Column[dict[str, Any]] = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "BlenderProc scene configuration: lighting, camera_path, "
            "environment, render_settings"
        ),
    )
    objects: Column[list[dict[str, Any]]] = Column(
        JSONB,
        nullable=False,
        default=list,
        comment=(
            "List of 3D object descriptors: {id, model_path, position, rotation, scale, "
            "material_config}"
        ),
    )

    # Visibility
    is_public: Column[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="If True, available to all tenants; otherwise tenant-private",
    )
    version: Column[int] = Column(
        Integer,
        nullable=False,
        default=1,
        comment="Template version for change tracking",
    )

    # Relationships
    generation_jobs = relationship(
        "VideoGenerationJob",
        back_populates="scene_template",
        lazy="dynamic",
    )

    def __repr__(self) -> str:
        return f"<SceneTemplate id={self.id} name={self.name!r} domain={self.domain}>"


# ---------------------------------------------------------------------------
# Metadata value objects — shared across adapters and interfaces
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass, field as _field  # noqa: E402


@_dataclass
class VideoBoundingBox:
    """Axis-aligned bounding box with confidence and class label.

    Used for object detection, face detection, and tracking results.

    Attributes:
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
        label: Semantic class label (e.g., "face", "car", "object").
        confidence: Detection confidence in [0.0, 1.0].
        track_id: Optional temporal tracking identifier.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    track_id: str | None = None


@_dataclass
class VideoTemporalEvent:
    """A detected event spanning a range of frames.

    Attributes:
        event_type: Event category (e.g., "scene_transition", "motion_peak").
        start_frame: First frame index of the event.
        end_frame: Last frame index (inclusive).
        confidence: Detection confidence in [0.0, 1.0].
        attributes: Additional event-specific attributes.
    """

    event_type: str
    start_frame: int
    end_frame: int
    confidence: float
    attributes: dict[str, Any] = _field(default_factory=dict)


@_dataclass
class VideoMetadata:
    """Complete structured metadata for a video sequence.

    Produced by the metadata extractor adapter and consumed by the export
    service for metadata embedding and privacy flag propagation.

    Attributes:
        num_frames: Total number of frames analysed.
        resolution: (width, height) of frames in pixels.
        fps_estimated: Estimated or configured frame rate.
        dominant_action: Most likely action class label.
        action_scores: Dict mapping action labels to confidence scores.
        objects_per_frame: Per-frame list of detected bounding boxes.
        scene_class: Top scene classification label.
        scene_scores: Dict mapping scene labels to confidence scores.
        temporal_events: List of detected temporal events.
        face_detections: Per-frame list of face bounding boxes.
        motion_summary: Motion analysis summary statistics.
        privacy_flags: Privacy-relevant boolean flags.
    """

    num_frames: int
    resolution: tuple[int, int]
    fps_estimated: float
    dominant_action: str
    action_scores: dict[str, float]
    objects_per_frame: list[list[VideoBoundingBox]]
    scene_class: str
    scene_scores: dict[str, float]
    temporal_events: list[VideoTemporalEvent]
    face_detections: list[list[VideoBoundingBox]]
    motion_summary: dict[str, float]
    privacy_flags: dict[str, bool]
