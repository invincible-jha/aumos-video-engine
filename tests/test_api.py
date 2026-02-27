"""Tests for FastAPI API routes and Pydantic request/response schemas.

Validates schema validation, field constraints, and route response shapes
without requiring a running database or Kafka broker.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from aumos_video_engine.api.schemas import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    BatchJobConfig,
    ComposeSceneRequest,
    ComposeSceneResponse,
    EnforcePrivacyRequest,
    EnforcePrivacyResponse,
    JobStatusResponse,
    ModelConfigRequest,
    SceneTemplateListResponse,
    SceneTemplateResponse,
    VideoGenerateRequest,
    VideoGenerateResponse,
)
from aumos_video_engine.core.models import JobStatus, JobType, VideoDomain


# ── ModelConfigRequest ─────────────────────────────────────────────


class TestModelConfigRequest:
    """Tests for ModelConfigRequest Pydantic schema."""

    def test_default_values_are_valid(self) -> None:
        """ModelConfigRequest must instantiate with all defaults."""
        config = ModelConfigRequest()
        assert config.guidance_scale == 3.0
        assert config.num_inference_steps == 25
        assert config.motion_bucket_id == 127
        assert config.noise_aug_strength == 0.02
        assert config.extra_params == {}

    def test_accepts_optional_seed(self) -> None:
        """ModelConfigRequest must accept an explicit seed value."""
        config = ModelConfigRequest(seed=42)
        assert config.seed == 42

    def test_guidance_scale_upper_bound(self) -> None:
        """guidance_scale must reject values above 20.0."""
        with pytest.raises(ValidationError):
            ModelConfigRequest(guidance_scale=21.0)

    def test_guidance_scale_lower_bound(self) -> None:
        """guidance_scale must reject negative values."""
        with pytest.raises(ValidationError):
            ModelConfigRequest(guidance_scale=-1.0)

    def test_num_inference_steps_lower_bound(self) -> None:
        """num_inference_steps must be at least 1."""
        with pytest.raises(ValidationError):
            ModelConfigRequest(num_inference_steps=0)

    def test_motion_bucket_id_bounds(self) -> None:
        """motion_bucket_id must be between 1 and 255."""
        with pytest.raises(ValidationError):
            ModelConfigRequest(motion_bucket_id=0)
        with pytest.raises(ValidationError):
            ModelConfigRequest(motion_bucket_id=256)


# ── VideoGenerateRequest ───────────────────────────────────────────


class TestVideoGenerateRequest:
    """Tests for VideoGenerateRequest Pydantic schema."""

    def test_minimal_valid_request(self) -> None:
        """VideoGenerateRequest must accept a prompt as minimum required field."""
        req = VideoGenerateRequest(prompt="Manufacturing defect detection")
        assert req.prompt == "Manufacturing defect detection"
        assert req.num_frames == 25
        assert req.fps == 24
        assert req.resolution == "1280x720"
        assert req.enforce_privacy is True

    def test_prompt_cannot_be_empty(self) -> None:
        """VideoGenerateRequest must reject an empty prompt."""
        with pytest.raises(ValidationError):
            VideoGenerateRequest(prompt="")

    def test_prompt_max_length_enforced(self) -> None:
        """VideoGenerateRequest must reject prompts longer than 2000 characters."""
        with pytest.raises(ValidationError):
            VideoGenerateRequest(prompt="x" * 2001)

    def test_valid_custom_resolution(self) -> None:
        """VideoGenerateRequest must accept valid WxH resolution strings."""
        req = VideoGenerateRequest(prompt="Test", resolution="640x480")
        assert req.resolution == "640x480"

    def test_resolution_invalid_format_rejected(self) -> None:
        """VideoGenerateRequest must reject non-WxH resolution strings."""
        with pytest.raises(ValidationError):
            VideoGenerateRequest(prompt="Test", resolution="640-480")

    def test_resolution_below_minimum_rejected(self) -> None:
        """VideoGenerateRequest must reject resolutions below 64x64."""
        with pytest.raises(ValidationError):
            VideoGenerateRequest(prompt="Test", resolution="32x32")

    def test_resolution_above_4k_rejected(self) -> None:
        """VideoGenerateRequest must reject resolutions above 3840x2160."""
        with pytest.raises(ValidationError):
            VideoGenerateRequest(prompt="Test", resolution="4000x2500")

    def test_domain_defaults_to_custom(self) -> None:
        """VideoGenerateRequest domain must default to VideoDomain.CUSTOM."""
        req = VideoGenerateRequest(prompt="Test")
        assert req.domain == VideoDomain.CUSTOM

    def test_num_frames_upper_bound(self) -> None:
        """VideoGenerateRequest must reject num_frames above 300."""
        with pytest.raises(ValidationError):
            VideoGenerateRequest(prompt="Test", num_frames=301)

    def test_fps_upper_bound(self) -> None:
        """VideoGenerateRequest must reject fps above 120."""
        with pytest.raises(ValidationError):
            VideoGenerateRequest(prompt="Test", fps=121)


# ── VideoGenerateResponse ──────────────────────────────────────────


class TestVideoGenerateResponse:
    """Tests for VideoGenerateResponse Pydantic schema."""

    def test_constructs_with_required_fields(self) -> None:
        """VideoGenerateResponse must accept all required fields."""
        response = VideoGenerateResponse(
            job_id=uuid.uuid4(),
            status=JobStatus.PENDING,
            job_type=JobType.GENERATE,
            domain=VideoDomain.MANUFACTURING,
            num_frames=25,
            fps=24,
            resolution="1280x720",
            duration_seconds=Decimal("1.042"),
        )
        assert response.status == JobStatus.PENDING
        assert "Poll" in response.message

    def test_duration_seconds_can_be_none(self) -> None:
        """VideoGenerateResponse must accept None for duration_seconds."""
        response = VideoGenerateResponse(
            job_id=uuid.uuid4(),
            status=JobStatus.PENDING,
            job_type=JobType.GENERATE,
            domain=VideoDomain.CUSTOM,
            num_frames=25,
            fps=24,
            resolution="1280x720",
            duration_seconds=None,
        )
        assert response.duration_seconds is None


# ── ComposeSceneRequest ────────────────────────────────────────────


class TestComposeSceneRequest:
    """Tests for ComposeSceneRequest Pydantic schema."""

    def test_valid_minimal_request(self) -> None:
        """ComposeSceneRequest must accept a template_id as minimum."""
        template_id = uuid.uuid4()
        req = ComposeSceneRequest(template_id=template_id)
        assert req.template_id == template_id
        assert req.num_frames == 25
        assert req.enforce_privacy is True

    def test_resolution_validator_rejects_non_wh(self) -> None:
        """ComposeSceneRequest must validate resolution format."""
        with pytest.raises(ValidationError):
            ComposeSceneRequest(template_id=uuid.uuid4(), resolution="not-a-res")

    def test_scene_overrides_defaults_to_empty_dict(self) -> None:
        """scene_overrides must default to empty dict."""
        req = ComposeSceneRequest(template_id=uuid.uuid4())
        assert req.scene_overrides == {}


# ── EnforcePrivacyRequest / Response ──────────────────────────────


class TestEnforcePrivacySchemas:
    """Tests for EnforcePrivacyRequest and EnforcePrivacyResponse schemas."""

    def test_request_defaults_blur_and_redact_to_true(self) -> None:
        """EnforcePrivacyRequest must default blur_faces and redact_plates to True."""
        req = EnforcePrivacyRequest(job_id=uuid.uuid4())
        assert req.blur_faces is True
        assert req.redact_plates is True
        assert req.remove_pii is False

    def test_response_contains_job_ids_and_status(self) -> None:
        """EnforcePrivacyResponse must expose job_id, new_job_id, and status."""
        job_id = uuid.uuid4()
        new_job_id = uuid.uuid4()
        resp = EnforcePrivacyResponse(
            job_id=job_id,
            new_job_id=new_job_id,
            status=JobStatus.PENDING,
        )
        assert resp.job_id == job_id
        assert resp.new_job_id == new_job_id
        assert resp.detection_summary == {}


# ── JobStatusResponse ──────────────────────────────────────────────


class TestJobStatusResponse:
    """Tests for JobStatusResponse schema completeness."""

    def test_all_quality_fields_present(self) -> None:
        """JobStatusResponse must include coherence score and privacy_enforced."""
        resp = JobStatusResponse(
            job_id=uuid.uuid4(),
            status=JobStatus.COMPLETED,
            job_type=JobType.GENERATE,
            domain=VideoDomain.MANUFACTURING,
            num_frames=25,
            fps=24,
            resolution="1280x720",
            duration_seconds=Decimal("1.042"),
            temporal_coherence_score=Decimal("0.8500"),
            privacy_enforced=True,
            output_uri="s3://bucket/video.mp4",
            error_message=None,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:01:00",
        )
        assert resp.temporal_coherence_score == Decimal("0.8500")
        assert resp.privacy_enforced is True
        assert resp.output_uri == "s3://bucket/video.mp4"

    def test_error_message_can_be_none(self) -> None:
        """JobStatusResponse must accept None for error_message."""
        resp = JobStatusResponse(
            job_id=uuid.uuid4(),
            status=JobStatus.PENDING,
            job_type=JobType.GENERATE,
            domain=VideoDomain.CUSTOM,
            num_frames=10,
            fps=24,
            resolution="640x480",
            duration_seconds=None,
            temporal_coherence_score=None,
            privacy_enforced=False,
            output_uri=None,
            error_message=None,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        assert resp.error_message is None


# ── BatchGenerateRequest ───────────────────────────────────────────


class TestBatchGenerateRequest:
    """Tests for BatchGenerateRequest Pydantic schema."""

    def test_single_job_batch_is_valid(self) -> None:
        """BatchGenerateRequest must accept a list with a single BatchJobConfig."""
        req = BatchGenerateRequest(
            jobs=[BatchJobConfig(prompt="Crowd monitoring")]
        )
        assert len(req.jobs) == 1

    def test_empty_jobs_list_rejected(self) -> None:
        """BatchGenerateRequest must reject an empty jobs list."""
        with pytest.raises(ValidationError):
            BatchGenerateRequest(jobs=[])

    def test_jobs_list_over_50_rejected(self) -> None:
        """BatchGenerateRequest must reject more than 50 job configs."""
        with pytest.raises(ValidationError):
            BatchGenerateRequest(
                jobs=[BatchJobConfig(prompt=f"Video {i}") for i in range(51)]
            )

    def test_batch_job_config_inherits_defaults(self) -> None:
        """BatchJobConfig must use sensible defaults for optional fields."""
        config = BatchJobConfig(prompt="Test video")
        assert config.num_frames == 25
        assert config.fps == 24
        assert config.resolution == "1280x720"
        assert config.domain == VideoDomain.CUSTOM
        assert config.enforce_privacy is True
        assert config.scene_template_id is None


# ── SceneTemplate schemas ──────────────────────────────────────────


class TestSceneTemplateSchemas:
    """Tests for SceneTemplateResponse and SceneTemplateListResponse."""

    def test_scene_template_response_fields(self) -> None:
        """SceneTemplateResponse must expose all required fields."""
        resp = SceneTemplateResponse(
            template_id=uuid.uuid4(),
            name="Factory Floor",
            domain=VideoDomain.MANUFACTURING,
            description="Manufacturing simulation",
            is_public=False,
            version=1,
            created_at="2026-01-01T00:00:00",
        )
        assert resp.name == "Factory Floor"
        assert resp.domain == VideoDomain.MANUFACTURING

    def test_scene_template_list_response_has_total(self) -> None:
        """SceneTemplateListResponse must include a total count."""
        resp = SceneTemplateListResponse(
            templates=[
                SceneTemplateResponse(
                    template_id=uuid.uuid4(),
                    name="Template A",
                    domain=VideoDomain.TRAFFIC,
                    description=None,
                    is_public=True,
                    version=2,
                    created_at="2026-01-01T00:00:00",
                )
            ],
            total=1,
        )
        assert resp.total == 1
        assert len(resp.templates) == 1
