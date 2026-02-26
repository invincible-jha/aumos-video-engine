"""Stable Video Diffusion (SVD) adapter for video frame generation.

Integrates HuggingFace diffusers StableVideoDiffusionPipeline to generate
temporally coherent frame sequences from conditioning images.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
from aumos_common.observability import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_PIPELINE_AVAILABLE = False
try:
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image

    _PIPELINE_AVAILABLE = True
except ImportError:
    logger.warning("diffusers/torch not installed — SVD generator unavailable")


class StableVideoDiffusionGenerator:
    """Generates video frame sequences using Stable Video Diffusion img2vid-xt.

    Uses HuggingFace diffusers pipeline with optional GPU acceleration.
    Falls back gracefully to CPU if CUDA is unavailable.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        cache_dir: str = "/tmp/model-cache",
        gpu_enabled: bool = False,
        cuda_device: int = 0,
    ) -> None:
        """Initialize the SVD generator.

        Args:
            model_id: HuggingFace model identifier for the SVD pipeline.
            cache_dir: Local directory for caching model weights.
            gpu_enabled: Whether to use CUDA GPU acceleration.
            cuda_device: CUDA device index (ignored if gpu_enabled=False).
        """
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._gpu_enabled = gpu_enabled
        self._cuda_device = cuda_device
        self._pipeline: Any | None = None

    def _get_device(self) -> str:
        """Determine the compute device string.

        Returns:
            "cuda:N" if GPU enabled and CUDA available, else "cpu".
        """
        if not self._gpu_enabled:
            return "cpu"
        if not _PIPELINE_AVAILABLE:
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{self._cuda_device}"
        except Exception:
            pass
        return "cpu"

    def _load_pipeline(self) -> None:
        """Lazy-load the SVD pipeline on first use."""
        if self._pipeline is not None:
            return
        if not _PIPELINE_AVAILABLE:
            raise RuntimeError(
                "diffusers and torch must be installed to use StableVideoDiffusionGenerator"
            )
        device = self._get_device()
        logger.info("Loading SVD pipeline", model_id=self._model_id, device=device)
        import torch

        self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            cache_dir=self._cache_dir,
        )
        self._pipeline = self._pipeline.to(device)
        logger.info("SVD pipeline loaded", device=device)

    async def generate_frames(
        self,
        prompt: str,
        num_frames: int,
        fps: int,
        resolution: tuple[int, int],
        model_config: dict[str, Any],
        reference_image: bytes | None,
    ) -> list[np.ndarray]:
        """Generate video frames using Stable Video Diffusion.

        SVD is an image-to-video model — it requires a conditioning image.
        If no reference_image is provided, a noise image is generated instead.

        Args:
            prompt: Text prompt (used for logging/metadata — SVD is img2vid, not txt2vid).
            num_frames: Number of frames to generate (SVD typically produces 25).
            fps: Target FPS for motion calibration.
            resolution: Output frame resolution (width, height).
            model_config: SVD parameters: seed, guidance_scale, num_inference_steps,
                motion_bucket_id, noise_aug_strength.
            reference_image: PNG/JPEG bytes for conditioning image.

        Returns:
            List of RGB uint8 numpy arrays with shape (H, W, 3).
        """
        self._load_pipeline()

        width, height = resolution
        seed = model_config.get("seed")
        guidance_scale = float(model_config.get("guidance_scale", 3.0))
        num_inference_steps = int(model_config.get("num_inference_steps", 25))
        motion_bucket_id = int(model_config.get("motion_bucket_id", 127))
        noise_aug_strength = float(model_config.get("noise_aug_strength", 0.02))

        # Prepare conditioning image
        if reference_image is not None:
            from PIL import Image

            pil_image = Image.open(io.BytesIO(reference_image)).convert("RGB")
        else:
            # Generate a neutral grey conditioning image
            from PIL import Image

            pil_image = Image.fromarray(
                np.full((height, width, 3), 128, dtype=np.uint8), mode="RGB"
            )

        pil_image = pil_image.resize((width, height))

        logger.info(
            "Running SVD inference",
            num_frames=num_frames,
            resolution=resolution,
            motion_bucket_id=motion_bucket_id,
        )

        import torch

        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        pipeline_output = self._pipeline(
            pil_image,
            num_frames=min(num_frames, 25),  # SVD-XT supports up to 25 frames natively
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            generator=generator,
        )

        # Convert PIL frames to numpy arrays
        frames: list[np.ndarray] = []
        for pil_frame in pipeline_output.frames[0]:
            frame_array = np.array(pil_frame.resize((width, height)).convert("RGB"))
            frames.append(frame_array)

        logger.info("SVD inference complete", num_frames_generated=len(frames))
        return frames

    async def is_available(self) -> bool:
        """Check whether the SVD pipeline can accept requests.

        Returns:
            True if diffusers/torch are installed; False otherwise.
        """
        return _PIPELINE_AVAILABLE
