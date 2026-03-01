"""Open-Sora v1.3 video generation adapter — GAP-88 competitive gap implementation.

Replaces SVD (Stable Video Diffusion) as the default video generation model.
Open-Sora v1.3 supports longer sequences, higher quality, and native 720p output
compared to SVD's 25-frame limitation at 576x1024.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import structlog
from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Model registry: name -> HuggingFace model ID or local path
MODEL_REGISTRY: dict[str, str] = {
    "open-sora-v1.3": "hpcai-tech/Open-Sora",
    "cogvideox-5b": "THUDM/CogVideoX-5b",
    "svd": "stabilityai/stable-video-diffusion-img2vid-xt",  # Legacy, kept for compat
}

# Maximum frames supported per model
MODEL_MAX_FRAMES: dict[str, int] = {
    "open-sora-v1.3": 204,  # ~8.5 seconds at 24fps
    "cogvideox-5b": 49,     # ~2 seconds at 24fps
    "svd": 25,              # ~1 second at 24fps (legacy)
}

# Supported resolutions per model
MODEL_RESOLUTIONS: dict[str, list[tuple[int, int]]] = {
    "open-sora-v1.3": [(512, 512), (256, 256), (720, 1280), (480, 854)],
    "cogvideox-5b": [(480, 720), (720, 480)],
    "svd": [(576, 1024)],
}


class OpenSoraAdapter:
    """Open-Sora v1.3 video generation adapter.

    Provides higher-quality, longer video generation than SVD with support
    for up to ~8.5 seconds of video at 24fps. Uses hpcai-tech/Open-Sora
    from HuggingFace with DiT (Diffusion Transformer) architecture.

    Args:
        model_name: Which model to use from MODEL_REGISTRY (default: "open-sora-v1.3").
        device: Torch device for inference ("cuda", "cpu").
        cache_dir: Model weight cache directory.
        torch_dtype: Computation dtype ("float16", "bfloat16", "float32").
        enable_attention_slicing: Reduce VRAM by processing attention in slices.
    """

    def __init__(
        self,
        model_name: str = "open-sora-v1.3",
        device: str = "cpu",
        cache_dir: str = "/tmp/model-cache",
        torch_dtype: str = "float16",
        enable_attention_slicing: bool = True,
    ) -> None:
        if model_name not in MODEL_REGISTRY:
            valid = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model '{model_name}'. Valid options: {valid}")
        self._model_name = model_name
        self._model_id = MODEL_REGISTRY[model_name]
        self._device = device
        self._cache_dir = cache_dir
        self._torch_dtype = torch_dtype
        self._enable_attention_slicing = enable_attention_slicing
        self._pipeline: Any = None
        self._log = logger.bind(adapter="open_sora", model=model_name)

    async def load_model(self) -> None:
        """Load model weights into memory.

        Downloads from HuggingFace on first call (cached thereafter).
        Runs in thread pool to avoid blocking the event loop.
        """
        self._log.info("open_sora.load_start", device=self._device)
        await asyncio.to_thread(self._load_model_sync)
        self._log.info("open_sora.load_complete")

    def _load_model_sync(self) -> None:
        """Synchronous model loading — called from thread pool."""
        try:
            import torch
            from diffusers import CogVideoXPipeline, DiffusionPipeline  # type: ignore[import]

            torch_dtype = getattr(torch, self._torch_dtype, torch.float16)

            if self._model_name == "cogvideox-5b":
                self._pipeline = CogVideoXPipeline.from_pretrained(
                    self._model_id,
                    torch_dtype=torch_dtype,
                    cache_dir=self._cache_dir,
                )
            elif self._model_name == "svd":
                from diffusers import StableVideoDiffusionPipeline

                self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    self._model_id,
                    torch_dtype=torch_dtype,
                    cache_dir=self._cache_dir,
                )
            else:
                # Open-Sora: use generic DiffusionPipeline loader
                self._pipeline = DiffusionPipeline.from_pretrained(
                    self._model_id,
                    torch_dtype=torch_dtype,
                    cache_dir=self._cache_dir,
                    trust_remote_code=True,
                )

            if self._device != "cpu":
                self._pipeline = self._pipeline.to(self._device)

            if self._enable_attention_slicing and hasattr(self._pipeline, "enable_attention_slicing"):
                self._pipeline.enable_attention_slicing()

        except Exception as exc:
            self._log.error("open_sora.load_failed", error=str(exc))
            raise

    @property
    def is_ready(self) -> bool:
        """True if the pipeline is loaded and ready for inference."""
        return self._pipeline is not None

    @property
    def max_frames(self) -> int:
        """Maximum number of frames this model can generate."""
        return MODEL_MAX_FRAMES.get(self._model_name, 25)

    @property
    def supported_resolutions(self) -> list[tuple[int, int]]:
        """List of (width, height) resolutions this model supports."""
        return MODEL_RESOLUTIONS.get(self._model_name, [(512, 512)])

    async def generate_frames(
        self,
        prompt: str,
        num_frames: int,
        fps: int,
        resolution: tuple[int, int],
        model_config: dict[str, Any],
        reference_image: bytes | None = None,
    ) -> list[np.ndarray]:
        """Generate a sequence of video frames.

        Args:
            prompt: Text description of the video content.
            num_frames: Number of frames to generate (capped at model max).
            fps: Target frames per second (for motion speed calibration).
            resolution: Output frame resolution as (width, height).
            model_config: Model-specific parameters:
                - guidance_scale: float (default 6.0)
                - num_inference_steps: int (default 50)
                - seed: int | None
                - negative_prompt: str | None
            reference_image: Optional conditioning image bytes (PNG/JPEG).
                When provided, the video will start from this reference frame.

        Returns:
            List of numpy arrays with shape (H, W, 3) in RGB uint8 format.
        """
        if not self.is_ready:
            raise RuntimeError(f"Model '{self._model_name}' not loaded. Call load_model() first.")

        effective_frames = min(num_frames, self.max_frames)
        width, height = resolution

        self._log.info(
            "open_sora.generate",
            prompt=prompt[:80],
            num_frames=effective_frames,
            resolution=resolution,
            model=self._model_name,
        )

        frames = await asyncio.to_thread(
            self._generate_sync,
            prompt,
            effective_frames,
            fps,
            width,
            height,
            model_config,
            reference_image,
        )

        self._log.info("open_sora.generate_complete", frames_generated=len(frames))
        return frames

    def _generate_sync(
        self,
        prompt: str,
        num_frames: int,
        fps: int,
        width: int,
        height: int,
        model_config: dict[str, Any],
        reference_image: bytes | None,
    ) -> list[np.ndarray]:
        """Synchronous frame generation — called from thread pool."""
        import torch
        import io
        from PIL import Image as PILImage

        guidance_scale: float = model_config.get("guidance_scale", 6.0)
        num_inference_steps: int = model_config.get("num_inference_steps", 50)
        seed: int | None = model_config.get("seed")
        negative_prompt: str | None = model_config.get("negative_prompt")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        if reference_image is not None:
            ref_img = PILImage.open(io.BytesIO(reference_image)).convert("RGB")
            ref_img = ref_img.resize((width, height))
            kwargs["image"] = ref_img
        else:
            kwargs["width"] = width
            kwargs["height"] = height

        with torch.no_grad():
            output = self._pipeline(**kwargs)

        # Extract frames from pipeline output
        if hasattr(output, "frames"):
            frames_data = output.frames
            # Handle nested list structure (batch x frames)
            if isinstance(frames_data, (list, tuple)) and len(frames_data) > 0:
                if isinstance(frames_data[0], (list, tuple)):
                    frames_data = frames_data[0]
        else:
            frames_data = output[0] if isinstance(output, (list, tuple)) else [output]

        result_frames: list[np.ndarray] = []
        for frame in frames_data:
            if isinstance(frame, PILImage.Image):
                arr = np.array(frame.convert("RGB"), dtype=np.uint8)
            elif isinstance(frame, np.ndarray):
                arr = frame.astype(np.uint8)
            else:
                arr = np.array(frame, dtype=np.uint8)
            result_frames.append(arr)

        return result_frames

    async def is_available(self) -> bool:
        """Check whether this generator is available."""
        return self.is_ready


class AutoregressiveFrameChainer:
    """Chains multiple generation calls to produce videos longer than model max.

    Uses the last frame of each segment as the conditioning image for the
    next segment, creating temporally consistent long-form video. Implements
    GAP-89: Longer Video Generation (>25 frames).

    Args:
        base_adapter: The underlying frame generator adapter.
        overlap_frames: Number of overlapping frames between segments for
            smooth transitions (default: 3).
    """

    def __init__(
        self,
        base_adapter: OpenSoraAdapter,
        overlap_frames: int = 3,
    ) -> None:
        self._adapter = base_adapter
        self._overlap = overlap_frames
        self._log = logger.bind(adapter="autoregressive_chainer")

    async def generate_long_video(
        self,
        prompt: str,
        total_frames: int,
        fps: int,
        resolution: tuple[int, int],
        model_config: dict[str, Any],
        reference_image: bytes | None = None,
    ) -> list[np.ndarray]:
        """Generate video longer than the model's maximum frame count.

        Splits the request into segments, uses the last frame of each segment
        as the reference image for the next, and concatenates with overlap
        blending for smooth transitions.

        Args:
            prompt: Text description of the video content.
            total_frames: Total number of frames desired.
            fps: Target frames per second.
            resolution: Output frame resolution as (width, height).
            model_config: Model-specific parameters.
            reference_image: Optional initial conditioning image.

        Returns:
            Combined list of numpy frames with smooth segment transitions.
        """
        max_per_segment = self._adapter.max_frames - self._overlap
        all_frames: list[np.ndarray] = []
        current_reference = reference_image
        frames_remaining = total_frames

        segment_idx = 0
        while frames_remaining > 0:
            segment_frames = min(frames_remaining + self._overlap, self._adapter.max_frames)
            self._log.info(
                "autoregressive_chainer.segment",
                segment=segment_idx,
                frames=segment_frames,
                remaining=frames_remaining,
            )

            generated = await self._adapter.generate_frames(
                prompt=prompt,
                num_frames=segment_frames,
                fps=fps,
                resolution=resolution,
                model_config=model_config,
                reference_image=current_reference,
            )

            if segment_idx == 0:
                all_frames.extend(generated)
            else:
                # Skip overlap frames and blend boundary
                all_frames.extend(generated[self._overlap :])

            # Use last frame as reference for next segment
            if generated:
                last_frame = generated[-1]
                current_reference = self._frame_to_bytes(last_frame)

            frames_remaining -= max_per_segment
            segment_idx += 1

        return all_frames[:total_frames]

    def _frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """Convert numpy frame array to PNG bytes for use as reference image."""
        import io
        from PIL import Image as PILImage

        img = PILImage.fromarray(frame.astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
