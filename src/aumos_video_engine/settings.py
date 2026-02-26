"""Service-specific settings extending AumOS base config."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """AumOS Video Engine settings.

    All standard AumOS env vars are inherited from AumOSSettings.
    Video-engine-specific vars use the AUMOS_VIDEO_ prefix.
    """

    service_name: str = "aumos-video-engine"

    # GPU configuration
    gpu_enabled: bool = Field(default=False, description="Enable CUDA GPU acceleration")
    cuda_device: int = Field(default=0, description="CUDA device index")

    # Upstream service URLs
    privacy_engine_url: str = Field(
        default="http://localhost:8010",
        description="URL of the aumos-privacy-engine service",
    )
    image_engine_url: str = Field(
        default="http://localhost:8009",
        description="URL of the aumos-image-engine service",
    )

    # Video generation defaults
    default_fps: int = Field(default=24, description="Default frames per second")
    default_resolution: str = Field(default="1280x720", description="Default resolution WxH")
    max_frames: int = Field(default=300, description="Maximum frames per video job")
    max_batch_size: int = Field(default=50, description="Maximum videos in a single batch job")
    storage_prefix: str = Field(default="video-jobs", description="MinIO object key prefix")

    # Stable Video Diffusion
    svd_model_id: str = Field(
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        description="HuggingFace model ID for Stable Video Diffusion",
    )
    svd_cache_dir: str = Field(
        default="/tmp/model-cache",
        description="Local directory for model weight caching",
    )

    # Temporal coherence
    temporal_coherence_min_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable temporal coherence score (0.0–1.0)",
    )
    temporal_coherence_window_frames: int = Field(
        default=8,
        description="Number of frames used in sliding window coherence evaluation",
    )

    model_config = SettingsConfigDict(env_prefix="AUMOS_VIDEO_")
