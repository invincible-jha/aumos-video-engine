"""Tests for ORM model definitions and enum types.

Validates model attributes, enum values, relationships,
and table naming conventions as defined in core/models.py.
"""

from __future__ import annotations

import pytest

from aumos_video_engine.core.models import (
    JobStatus,
    JobType,
    SceneTemplate,
    VideoDomain,
    VideoGenerationJob,
)


class TestJobTypeEnum:
    """Tests for the JobType string enum."""

    def test_job_type_values_are_strings(self) -> None:
        """All JobType members must be string-valued for JSON serialization."""
        for member in JobType:
            assert isinstance(member.value, str), f"Expected str, got {type(member.value)}"

    def test_job_type_generate_value(self) -> None:
        """GENERATE must serialize to 'generate'."""
        assert JobType.GENERATE.value == "generate"

    def test_job_type_scene_compose_value(self) -> None:
        """SCENE_COMPOSE must serialize to 'scene_compose'."""
        assert JobType.SCENE_COMPOSE.value == "scene_compose"

    def test_job_type_privacy_enforce_value(self) -> None:
        """PRIVACY_ENFORCE must serialize to 'privacy_enforce'."""
        assert JobType.PRIVACY_ENFORCE.value == "privacy_enforce"

    def test_job_type_roundtrip_from_string(self) -> None:
        """JobType must be constructible from its string value."""
        assert JobType("generate") is JobType.GENERATE
        assert JobType("scene_compose") is JobType.SCENE_COMPOSE


class TestJobStatusEnum:
    """Tests for the JobStatus lifecycle enum."""

    def test_all_lifecycle_states_present(self) -> None:
        """All five lifecycle states must exist."""
        expected = {"pending", "running", "completed", "failed", "cancelled"}
        actual = {member.value for member in JobStatus}
        assert actual == expected

    def test_terminal_states(self) -> None:
        """COMPLETED, FAILED, CANCELLED are terminal states and must exist."""
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_pending_is_initial_state(self) -> None:
        """PENDING is the initial state for new jobs."""
        assert JobStatus.PENDING.value == "pending"


class TestVideoDomainEnum:
    """Tests for the VideoDomain scenario enum."""

    def test_all_domains_present(self) -> None:
        """All four domains must be defined."""
        expected = {"manufacturing", "surveillance", "traffic", "custom"}
        actual = {member.value for member in VideoDomain}
        assert actual == expected

    def test_custom_domain_value(self) -> None:
        """CUSTOM domain must serialize to 'custom'."""
        assert VideoDomain.CUSTOM.value == "custom"

    def test_domain_roundtrip_from_string(self) -> None:
        """VideoDomain must be constructible from string values."""
        assert VideoDomain("manufacturing") is VideoDomain.MANUFACTURING
        assert VideoDomain("surveillance") is VideoDomain.SURVEILLANCE
        assert VideoDomain("traffic") is VideoDomain.TRAFFIC


class TestVideoGenerationJobModel:
    """Tests for VideoGenerationJob ORM model structure."""

    def test_table_name_has_vid_prefix(self) -> None:
        """VideoGenerationJob must use the vid_ table prefix."""
        assert VideoGenerationJob.__tablename__ == "vid_generation_jobs"

    def test_model_has_required_columns(self) -> None:
        """VideoGenerationJob must define all required column attributes."""
        required_attrs = [
            "job_type",
            "status",
            "model_config_json",
            "num_frames",
            "fps",
            "resolution",
            "duration_seconds",
            "temporal_coherence_score",
            "privacy_enforced",
            "domain",
            "prompt",
            "output_uri",
            "error_message",
        ]
        for attr in required_attrs:
            assert hasattr(VideoGenerationJob, attr), f"Missing column: {attr}"

    def test_scene_template_relationship_defined(self) -> None:
        """VideoGenerationJob must have a scene_template relationship."""
        assert hasattr(VideoGenerationJob, "scene_template")

    def test_repr_contains_status_and_domain(self) -> None:
        """__repr__ must include id, status and domain for debugging."""
        from unittest.mock import MagicMock
        import uuid

        job = MagicMock(spec=VideoGenerationJob)
        job.id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        job.status = JobStatus.PENDING
        job.domain = VideoDomain.MANUFACTURING
        # Call the actual __repr__ via the class method with the mock
        repr_str = VideoGenerationJob.__repr__(job)
        assert "PENDING" in repr_str or "pending" in repr_str
        assert "MANUFACTURING" in repr_str or "manufacturing" in repr_str


class TestSceneTemplateModel:
    """Tests for SceneTemplate ORM model structure."""

    def test_table_name_has_vid_prefix(self) -> None:
        """SceneTemplate must use the vid_ table prefix."""
        assert SceneTemplate.__tablename__ == "vid_scene_templates"

    def test_model_has_required_columns(self) -> None:
        """SceneTemplate must define all required column attributes."""
        required_attrs = [
            "name",
            "domain",
            "description",
            "scene_config",
            "objects",
            "is_public",
            "version",
        ]
        for attr in required_attrs:
            assert hasattr(SceneTemplate, attr), f"Missing column: {attr}"

    def test_generation_jobs_relationship_defined(self) -> None:
        """SceneTemplate must have a generation_jobs relationship."""
        assert hasattr(SceneTemplate, "generation_jobs")

    def test_repr_contains_name_and_domain(self) -> None:
        """__repr__ must include id, name, and domain."""
        from unittest.mock import MagicMock
        import uuid

        template = MagicMock(spec=SceneTemplate)
        template.id = uuid.UUID("22222222-2222-2222-2222-222222222222")
        template.name = "Test Template"
        template.domain = VideoDomain.TRAFFIC
        repr_str = SceneTemplate.__repr__(template)
        assert "Test Template" in repr_str
        assert "TRAFFIC" in repr_str or "traffic" in repr_str
