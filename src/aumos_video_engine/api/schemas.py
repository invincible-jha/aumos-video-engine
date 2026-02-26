"""Pydantic request/response schemas for aumos-video-engine API."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator

from aumos_video_engine.core.models import JobStatus, JobType, VideoDomain


# ── Shared ────────────────────────────────────────────────────────


class ModelConfigRequest(BaseModel):
    """Model configuration parameters for video generation."""

    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    guidance_scale: float = Field(
        default=3.0, ge=0.0, le=20.0, description="Classifier-free guidance scale"
    )
    num_inference_steps: int = Field(
        default=25, ge=1, le=200, description="Denoising steps (higher = quality vs speed)"
    )
    motion_bucket_id: int = Field(
        default=127, ge=1, le=255, description="Motion intensity (SVD-specific, 1=slow, 255=fast)"
    )
    noise_aug_strength: float = Field(
        default=0.02, ge=0.0, le=1.0, description="Noise augmentation for conditioning image"
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific parameters"
    )


# ── Video Generation ──────────────────────────────────────────────


class VideoGenerateRequest(BaseModel):
    """Request body for POST /video/generate."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for video content")
    num_frames: int = Field(default=25, ge=1, le=300, description="Number of frames to generate")
    fps: int = Field(default=24, ge=1, le=120, description="Target frames per second")
    resolution: str = Field(default="1280x720", description="Output resolution as WIDTHxHEIGHT")
    domain: VideoDomain = Field(default=VideoDomain.CUSTOM, description="Generation domain")
    model_config_params: ModelConfigRequest = Field(
        default_factory=ModelConfigRequest,
        description="Model configuration parameters",
    )
    enforce_privacy: bool = Field(
        default=True, description="Apply per-frame privacy enforcement (face blur, plate redaction)"
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: str) -> str:
        """Validate WIDTHxHEIGHT format."""
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Resolution must be WIDTHxHEIGHT, got: {value!r}")
        try:
            width, height = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Resolution dimensions must be integers, got: {value!r}") from exc
        if width < 64 or height < 64:
            raise ValueError("Minimum resolution is 64x64")
        if width > 3840 or height > 2160:
            raise ValueError("Maximum resolution is 3840x2160 (4K)")
        return value


class VideoGenerateResponse(BaseModel):
    """Response for POST /video/generate — returns the created job for polling."""

    job_id: uuid.UUID
    status: JobStatus
    job_type: JobType
    domain: VideoDomain
    num_frames: int
    fps: int
    resolution: str
    duration_seconds: Decimal | None
    message: str = "Video generation job submitted. Poll GET /video/jobs/{job_id} for status."


# ── Scene Composition ─────────────────────────────────────────────


class ComposeSceneRequest(BaseModel):
    """Request body for POST /video/compose-scene."""

    template_id: uuid.UUID = Field(..., description="Scene template to use for composition")
    num_frames: int = Field(default=25, ge=1, le=300, description="Number of frames to render")
    fps: int = Field(default=24, ge=1, le=120, description="Target frames per second")
    resolution: str = Field(default="1280x720", description="Output resolution as WIDTHxHEIGHT")
    scene_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional overrides for scene_config parameters",
    )
    enforce_privacy: bool = Field(
        default=True, description="Apply per-frame privacy enforcement"
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: str) -> str:
        """Validate WIDTHxHEIGHT format."""
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Resolution must be WIDTHxHEIGHT, got: {value!r}")
        try:
            int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Resolution dimensions must be integers, got: {value!r}") from exc
        return value


class ComposeSceneResponse(BaseModel):
    """Response for POST /video/compose-scene."""

    job_id: uuid.UUID
    status: JobStatus
    template_id: uuid.UUID
    domain: VideoDomain
    num_frames: int
    fps: int
    resolution: str
    message: str = "Scene composition job submitted. Poll GET /video/jobs/{job_id} for status."


# ── Privacy Enforcement ───────────────────────────────────────────


class EnforcePrivacyRequest(BaseModel):
    """Request body for POST /video/enforce-privacy."""

    job_id: uuid.UUID = Field(..., description="Job ID of an existing completed video to re-enforce")
    blur_faces: bool = Field(default=True, description="Blur detected faces")
    redact_plates: bool = Field(default=True, description="Redact license plates")
    remove_pii: bool = Field(default=False, description="Remove other PII (text, badges, etc.)")


class EnforcePrivacyResponse(BaseModel):
    """Response for POST /video/enforce-privacy."""

    job_id: uuid.UUID
    new_job_id: uuid.UUID
    status: JobStatus
    detection_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Detection counts per entity type from enforcement run",
    )
    output_uri: str | None = None
    message: str = "Privacy enforcement job submitted."


# ── Job Status ────────────────────────────────────────────────────


class JobStatusResponse(BaseModel):
    """Response for GET /video/jobs/{id}."""

    job_id: uuid.UUID
    status: JobStatus
    job_type: JobType
    domain: VideoDomain
    num_frames: int
    fps: int
    resolution: str
    duration_seconds: Decimal | None
    temporal_coherence_score: Decimal | None
    privacy_enforced: bool
    output_uri: str | None
    error_message: str | None
    created_at: str
    updated_at: str


# ── Batch Generation ──────────────────────────────────────────────


class BatchJobConfig(BaseModel):
    """Configuration for a single video within a batch submission."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    num_frames: int = Field(default=25, ge=1, le=300)
    fps: int = Field(default=24, ge=1, le=120)
    resolution: str = Field(default="1280x720")
    domain: VideoDomain = Field(default=VideoDomain.CUSTOM)
    model_config_params: ModelConfigRequest = Field(default_factory=ModelConfigRequest)
    enforce_privacy: bool = Field(default=True)
    scene_template_id: uuid.UUID | None = Field(default=None)


class BatchGenerateRequest(BaseModel):
    """Request body for POST /video/batch."""

    jobs: list[BatchJobConfig] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of video generation configurations (max 50 per batch)",
    )


class BatchGenerateResponse(BaseModel):
    """Response for POST /video/batch."""

    batch_id: str
    num_jobs: int
    job_ids: list[uuid.UUID]
    message: str = "Batch submitted. Poll individual job IDs for status."


# ── Scene Templates ───────────────────────────────────────────────


class SceneTemplateResponse(BaseModel):
    """Single scene template in list/detail response."""

    template_id: uuid.UUID
    name: str
    domain: VideoDomain
    description: str | None
    is_public: bool
    version: int
    created_at: str


class SceneTemplateListResponse(BaseModel):
    """Response for GET /video/templates."""

    templates: list[SceneTemplateResponse]
    total: int
