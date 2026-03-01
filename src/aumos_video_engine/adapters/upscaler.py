"""Real-ESRGAN video upscaler adapter — GAP-91: Higher Resolution Support (1080p+).

Applies Real-ESRGAN 4x upscaling to post-process video frames from 720p (1280x720)
to 1080p (1920x1080) or beyond. Enables 1080p output from Open-Sora v1.3
without retraining at higher resolution.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class VideoUpscaler:
    """Real-ESRGAN 4x video upscaler for post-generation super-resolution.

    Upscales individual frames using Real-ESRGAN to achieve resolutions
    beyond the native generation model capability. Supports:
    - 720p → 1080p (1.5x)
    - 720p → 2160p/4K (4x via two 2x passes or direct 4x)
    - Any resolution upscale via Real-ESRGAN tile processing

    Args:
        scale: Upscaling factor. Must be a multiple of 2 that the model supports.
            Common values: 2 (2x), 4 (4x).
        model_name: Real-ESRGAN model variant. Options:
            - "RealESRGAN_x4plus" — general-purpose 4x upscaler (default)
            - "RealESRGAN_x2plus" — 2x upscaler, less artifact-prone
            - "RealESRGAN_x4plus_anime_6B" — anime/illustration optimized
        tile_size: Tile size for processing large frames. Reduce for lower VRAM.
            0 = no tiling (process whole image at once).
        device: Torch device string ("cuda", "cpu", "cuda:0").
        cache_dir: Directory for model weight caching.
    """

    SUPPORTED_MODELS: list[str] = [
        "RealESRGAN_x4plus",
        "RealESRGAN_x2plus",
        "RealESRGAN_x4plus_anime_6B",
    ]

    def __init__(
        self,
        scale: int = 4,
        model_name: str = "RealESRGAN_x4plus",
        tile_size: int = 256,
        device: str = "cpu",
        cache_dir: str = "/tmp/realesrgan-cache",
    ) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Valid: {self.SUPPORTED_MODELS}")
        self._scale = scale
        self._model_name = model_name
        self._tile_size = tile_size
        self._device = device
        self._cache_dir = cache_dir
        self._model: Any = None
        self._log = logger.bind(component="video_upscaler", model=model_name, scale=scale)

    async def warm_up(self) -> None:
        """Load Real-ESRGAN model weights into memory.

        Downloads weights from HuggingFace on first call, then caches.
        Offloads to thread pool to avoid blocking event loop.
        """
        self._log.info("upscaler.warm_up_start", device=self._device)
        await asyncio.to_thread(self._load_model_sync)
        self._log.info("upscaler.warm_up_complete")

    def _load_model_sync(self) -> None:
        """Load model weights synchronously — called from thread pool."""
        try:
            from realesrgan import RealESRGANer  # type: ignore[import]
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore[import]
            import torch

            num_in_ch = 3
            num_out_ch = 3

            if self._model_name == "RealESRGAN_x4plus_anime_6B":
                model_arch = RRDBNet(
                    num_in_ch=num_in_ch,
                    num_out_ch=num_out_ch,
                    num_feat=64,
                    num_block=6,
                    num_grow_ch=32,
                    scale=4,
                )
                netscale = 4
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            elif self._model_name == "RealESRGAN_x2plus":
                model_arch = RRDBNet(
                    num_in_ch=num_in_ch,
                    num_out_ch=num_out_ch,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
                netscale = 2
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            else:
                # RealESRGAN_x4plus (default)
                model_arch = RRDBNet(
                    num_in_ch=num_in_ch,
                    num_out_ch=num_out_ch,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
                netscale = 4
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

            self._model = RealESRGANer(
                scale=netscale,
                model_path=model_url,
                model=model_arch,
                tile=self._tile_size,
                tile_pad=10,
                pre_pad=0,
                half=self._device != "cpu",
            )

            if self._device != "cpu" and torch.cuda.is_available():
                self._model.device = torch.device(self._device)
                self._model.model = self._model.model.to(self._device)

        except ImportError as exc:
            self._log.warning(
                "upscaler.realesrgan_not_installed",
                error=str(exc),
                fallback="lanczos_interpolation",
            )
            self._model = None

    @property
    def is_ready(self) -> bool:
        """True if upscaler model is loaded or fallback is active."""
        return True  # fallback always available

    async def upscale_frame(
        self,
        frame: np.ndarray,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> np.ndarray:
        """Upscale a single video frame.

        Args:
            frame: Input frame as (H, W, 3) uint8 RGB array.
            target_width: Optional target width in pixels. If provided,
                the output is cropped/padded to this exact width after upscaling.
            target_height: Optional target height in pixels.

        Returns:
            Upscaled frame as (H*scale, W*scale, 3) uint8 RGB array.
        """
        return await asyncio.to_thread(
            self._upscale_frame_sync, frame, target_width, target_height
        )

    def _upscale_frame_sync(
        self,
        frame: np.ndarray,
        target_width: int | None,
        target_height: int | None,
    ) -> np.ndarray:
        """Upscale frame synchronously — called from thread pool.

        Args:
            frame: Input (H, W, 3) uint8 RGB frame.
            target_width: Optional target width for final crop/resize.
            target_height: Optional target height for final crop/resize.

        Returns:
            Upscaled (H', W', 3) uint8 RGB frame.
        """
        import cv2

        if self._model is not None:
            try:
                # Real-ESRGAN expects BGR
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                output_bgr, _ = self._model.enhance(bgr_frame, outscale=self._scale)
                upscaled = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                self._log.warning("upscaler.realesrgan_failed", error=str(exc), fallback="lanczos")
                upscaled = self._lanczos_upscale(frame, cv2)
        else:
            upscaled = self._lanczos_upscale(frame, cv2)

        if target_width is not None and target_height is not None:
            current_h, current_w = upscaled.shape[:2]
            if current_w != target_width or current_h != target_height:
                upscaled = cv2.resize(
                    upscaled,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )

        return upscaled.astype(np.uint8)

    def _lanczos_upscale(self, frame: np.ndarray, cv2: Any) -> np.ndarray:
        """Fallback Lanczos upscaling when Real-ESRGAN is unavailable.

        Args:
            frame: Input (H, W, 3) uint8 RGB frame.
            cv2: OpenCV module (already imported in caller).

        Returns:
            Lanczos-upscaled frame.
        """
        h, w = frame.shape[:2]
        new_w, new_h = w * self._scale, h * self._scale
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    async def upscale_batch(
        self,
        frames: list[np.ndarray],
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> list[np.ndarray]:
        """Upscale a batch of video frames.

        Processes frames sequentially in a thread pool to avoid OOM.
        For GPU inference, batching would be more efficient but requires
        more VRAM than tile-based processing.

        Args:
            frames: List of (H, W, 3) uint8 RGB frames.
            target_width: Optional output width.
            target_height: Optional output height.

        Returns:
            List of upscaled frames in the same order as input.
        """
        self._log.info("upscaler.batch_start", num_frames=len(frames))

        upscaled_frames: list[np.ndarray] = []
        for idx, frame in enumerate(frames):
            upscaled = await self.upscale_frame(frame, target_width, target_height)
            upscaled_frames.append(upscaled)
            if idx % 10 == 0:
                self._log.debug("upscaler.batch_progress", frame=idx, total=len(frames))

        self._log.info("upscaler.batch_complete", num_frames=len(upscaled_frames))
        return upscaled_frames
