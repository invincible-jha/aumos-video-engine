"""FastAPI router for aumos-video-engine.

All endpoints are thin delegates — no business logic here.
Routes validate input (via Pydantic), call services, and return structured responses.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.pagination import PageRequest, PageResponse

from aumos_video_engine.api.dependencies import (
    get_batch_service,
    get_generation_service,
    get_privacy_enforcement_service,
    get_scene_composition_service,
)
from aumos_video_engine.api.schemas import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    ComposeSceneRequest,
    ComposeSceneResponse,
    EnforcePrivacyRequest,
    EnforcePrivacyResponse,
    JobStatusResponse,
    SceneTemplateListResponse,
    SceneTemplateResponse,
    VideoGenerateRequest,
    VideoGenerateResponse,
)
from aumos_video_engine.core.services import (
    BatchService,
    GenerationService,
    PrivacyEnforcementService,
    SceneCompositionService,
)

router = APIRouter(prefix="/video", tags=["Video Generation"])


@router.post(
    "/generate",
    response_model=VideoGenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate synthetic video",
    description=(
        "Submit an asynchronous video generation job using Stable Video Diffusion. "
        "Returns immediately with a job_id for polling. "
        "Privacy enforcement (face blur, plate redaction) is applied by default."
    ),
)
async def generate_video(
    request: VideoGenerateRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[GenerationService, Depends(get_generation_service)],
) -> VideoGenerateResponse:
    """Submit a video generation job."""
    job = await service.create_job(
        tenant=tenant,
        prompt=request.prompt,
        num_frames=request.num_frames,
        fps=request.fps,
        resolution=request.resolution,
        domain=request.domain,
        model_config=request.model_config_params.model_dump(),
        enforce_privacy=request.enforce_privacy,
    )
    return VideoGenerateResponse(
        job_id=job.id,
        status=job.status,
        job_type=job.job_type,
        domain=job.domain,
        num_frames=job.num_frames,
        fps=job.fps,
        resolution=job.resolution,
        duration_seconds=job.duration_seconds,
    )


@router.post(
    "/compose-scene",
    response_model=ComposeSceneResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Compose 3D scene and generate video",
    description=(
        "Render a domain-specific 3D scene using BlenderProc and submit it as a generation job. "
        "Uses a stored SceneTemplate for scene configuration. "
        "Returns a job_id for polling."
    ),
)
async def compose_scene(
    request: ComposeSceneRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    generation_service: Annotated[GenerationService, Depends(get_generation_service)],
    scene_service: Annotated[SceneCompositionService, Depends(get_scene_composition_service)],
) -> ComposeSceneResponse:
    """Submit a scene composition job."""
    template = await scene_service.get_template(
        template_id=request.template_id,
        tenant=tenant,
    )
    job = await generation_service.create_job(
        tenant=tenant,
        prompt=f"Scene composition from template: {template.name}",
        num_frames=request.num_frames,
        fps=request.fps,
        resolution=request.resolution,
        domain=template.domain,
        model_config=request.scene_overrides,
        enforce_privacy=request.enforce_privacy,
        scene_template_id=str(request.template_id),
    )
    return ComposeSceneResponse(
        job_id=job.id,
        status=job.status,
        template_id=request.template_id,
        domain=job.domain,
        num_frames=job.num_frames,
        fps=job.fps,
        resolution=job.resolution,
    )


@router.post(
    "/enforce-privacy",
    response_model=EnforcePrivacyResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Apply per-frame privacy enforcement to existing video",
    description=(
        "Re-process a completed video job with privacy enforcement. "
        "Creates a new job that applies face blur, plate redaction, and/or PII removal. "
        "The original job is not modified."
    ),
)
async def enforce_privacy(
    request: EnforcePrivacyRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    generation_service: Annotated[GenerationService, Depends(get_generation_service)],
    privacy_service: Annotated[PrivacyEnforcementService, Depends(get_privacy_enforcement_service)],
) -> EnforcePrivacyResponse:
    """Submit a privacy enforcement job for an existing video."""
    # Verify source job exists and belongs to this tenant
    source_job = await generation_service.get_job(request.job_id, tenant)

    # Create a new privacy-enforce job
    new_job = await generation_service.create_job(
        tenant=tenant,
        prompt=f"Privacy enforcement of job {source_job.id}",
        num_frames=source_job.num_frames,
        fps=source_job.fps,
        resolution=source_job.resolution,
        domain=source_job.domain,
        model_config={
            "source_job_id": str(source_job.id),
            "blur_faces": request.blur_faces,
            "redact_plates": request.redact_plates,
            "remove_pii": request.remove_pii,
        },
        enforce_privacy=True,
    )
    return EnforcePrivacyResponse(
        job_id=request.job_id,
        new_job_id=new_job.id,
        status=new_job.status,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get video generation job status",
    description="Poll the status of a video generation job. Returns output_uri when completed.",
)
async def get_job_status(
    job_id: uuid.UUID,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[GenerationService, Depends(get_generation_service)],
) -> JobStatusResponse:
    """Retrieve video generation job status."""
    job = await service.get_job(job_id, tenant)
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        job_type=job.job_type,
        domain=job.domain,
        num_frames=job.num_frames,
        fps=job.fps,
        resolution=job.resolution,
        duration_seconds=job.duration_seconds,
        temporal_coherence_score=job.temporal_coherence_score,
        privacy_enforced=job.privacy_enforced,
        output_uri=job.output_uri,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )


@router.post(
    "/batch",
    response_model=BatchGenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch video generation",
    description=(
        "Submit up to 50 video generation jobs in a single request. "
        "All jobs are created immediately with PENDING status. "
        "Poll individual job IDs for status."
    ),
)
async def submit_batch(
    request: BatchGenerateRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[BatchService, Depends(get_batch_service)],
) -> BatchGenerateResponse:
    """Submit a batch of video generation jobs."""
    job_configs = [
        {
            "prompt": config.prompt,
            "num_frames": config.num_frames,
            "fps": config.fps,
            "resolution": config.resolution,
            "domain": config.domain.value,
            "model_config": config.model_config_params.model_dump(),
            "enforce_privacy": config.enforce_privacy,
            "scene_template_id": str(config.scene_template_id) if config.scene_template_id else None,
        }
        for config in request.jobs
    ]
    jobs = await service.submit_batch(tenant=tenant, job_configs=job_configs)
    batch_id = str(uuid.uuid4())
    return BatchGenerateResponse(
        batch_id=batch_id,
        num_jobs=len(jobs),
        job_ids=[job.id for job in jobs],
    )


@router.get(
    "/templates",
    response_model=SceneTemplateListResponse,
    summary="List available scene templates",
    description=(
        "List all scene templates accessible to the current tenant. "
        "Includes tenant-private templates and public (shared) templates. "
        "Use domain query parameter to filter."
    ),
)
async def list_templates(
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[SceneCompositionService, Depends(get_scene_composition_service)],
    domain: str | None = None,
) -> SceneTemplateListResponse:
    """List scene templates available to the current tenant."""
    from aumos_video_engine.core.models import VideoDomain

    domain_filter = VideoDomain(domain) if domain else None
    templates = await service.list_templates(tenant=tenant, domain=domain_filter)
    return SceneTemplateListResponse(
        templates=[
            SceneTemplateResponse(
                template_id=template.id,
                name=template.name,
                domain=template.domain,
                description=template.description,
                is_public=template.is_public,
                version=template.version,
                created_at=template.created_at.isoformat(),
            )
            for template in templates
        ],
        total=len(templates),
    )
