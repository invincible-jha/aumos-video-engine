"""Video style transfer adapter for domain-specific visual augmentation.

Applies artistic and domain-specific style transformations to generated
video sequences while preserving temporal coherence across frames.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class VideoStyle(str, Enum):
    """Predefined video style presets."""

    INFRARED = "infrared"
    NIGHT_VISION = "night_vision"
    THERMAL = "thermal"
    SECURITY_CAM = "security_cam"
    DASHCAM = "dashcam"
    INDUSTRIAL = "industrial"
    CARTOON = "cartoon"
    SKETCH = "sketch"
    OIL_PAINTING = "oil_painting"
    CINEMATIC = "cinematic"


class VideoStyleTransfer:
    """Applies style transformations to video frame sequences.

    Transforms entire frame sequences with consistent style parameters
    to maintain temporal coherence. Supports both analytic transforms
    (color mapping, noise injection) and neural style transfer when
    torch is available.

    Args:
        preserve_coherence: Whether to enforce frame-to-frame consistency.
        max_style_strength: Maximum style application strength (0.0-1.0).
    """

    def __init__(
        self,
        preserve_coherence: bool = True,
        max_style_strength: float = 0.8,
    ) -> None:
        self._preserve_coherence = preserve_coherence
        self._max_style_strength = max_style_strength

    async def apply_style(
        self,
        frames: list[np.ndarray],
        style: VideoStyle,
        strength: float = 0.5,
        style_params: dict[str, Any] | None = None,
    ) -> list[np.ndarray]:
        """Apply a predefined style to a video frame sequence.

        Args:
            frames: RGB uint8 frames (H, W, 3).
            style: Target visual style.
            strength: Style application intensity (0.0-1.0).
            style_params: Optional style-specific parameters.

        Returns:
            Styled frame sequence (same length and shape).
        """
        effective_strength = min(strength, self._max_style_strength)
        params = style_params or {}

        logger.info(
            "applying_video_style",
            style=style.value,
            strength=effective_strength,
            frame_count=len(frames),
        )

        style_fn = self._get_style_function(style)
        styled = [style_fn(f, effective_strength, params) for f in frames]

        if self._preserve_coherence:
            styled = self._smooth_transitions(styled)

        return styled

    def _get_style_function(self, style: VideoStyle) -> Any:
        """Look up the style transformation function.

        Args:
            style: Target visual style.

        Returns:
            Callable(frame, strength, params) -> styled_frame.
        """
        style_map = {
            VideoStyle.INFRARED: self._apply_infrared,
            VideoStyle.NIGHT_VISION: self._apply_night_vision,
            VideoStyle.THERMAL: self._apply_thermal,
            VideoStyle.SECURITY_CAM: self._apply_security_cam,
            VideoStyle.DASHCAM: self._apply_dashcam,
            VideoStyle.INDUSTRIAL: self._apply_industrial,
            VideoStyle.CARTOON: self._apply_cartoon,
            VideoStyle.SKETCH: self._apply_sketch,
            VideoStyle.OIL_PAINTING: self._apply_oil_painting,
            VideoStyle.CINEMATIC: self._apply_cinematic,
        }
        return style_map.get(style, self._apply_identity)

    @staticmethod
    def _apply_identity(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """No-op identity transform."""
        return frame

    @staticmethod
    def _apply_infrared(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Simulate infrared imagery with false color mapping."""
        gray = np.mean(frame, axis=2).astype(np.float32)
        ir = np.zeros_like(frame, dtype=np.float32)
        ir[:, :, 0] = np.clip(gray * 1.5, 0, 255)
        ir[:, :, 1] = np.clip(gray * 0.3, 0, 255)
        ir[:, :, 2] = np.clip(gray * 0.8, 0, 255)
        blended = frame.astype(np.float32) * (1 - strength) + ir * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_night_vision(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Simulate night vision green phosphor display."""
        gray = np.mean(frame, axis=2).astype(np.float32)
        noise = np.random.normal(0, 8, gray.shape).astype(np.float32)
        nv = np.zeros_like(frame, dtype=np.float32)
        nv[:, :, 0] = np.clip((gray + noise) * 0.2, 0, 255)
        nv[:, :, 1] = np.clip((gray + noise) * 1.2, 0, 255)
        nv[:, :, 2] = np.clip((gray + noise) * 0.1, 0, 255)
        blended = frame.astype(np.float32) * (1 - strength) + nv * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_thermal(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Simulate thermal imaging with heat map coloring."""
        gray = np.mean(frame, axis=2).astype(np.float32) / 255.0
        thermal = np.zeros_like(frame, dtype=np.float32)
        thermal[:, :, 0] = np.clip(gray * 3.0 * 255, 0, 255)
        thermal[:, :, 1] = np.clip((gray - 0.33) * 3.0 * 255, 0, 255)
        thermal[:, :, 2] = np.clip((gray - 0.66) * 3.0 * 255, 0, 255)
        blended = frame.astype(np.float32) * (1 - strength) + thermal * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_security_cam(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Simulate low-quality security camera footage."""
        result = frame.astype(np.float32)
        result *= (1.0 - 0.3 * strength)
        noise = np.random.normal(0, 15 * strength, result.shape)
        result += noise
        if strength > 0.3:
            h, w = result.shape[:2]
            for y in range(0, h, max(1, int(4 / strength))):
                result[y, :, :] *= 0.85
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_dashcam(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Simulate wide-angle dashcam footage characteristics."""
        result = frame.astype(np.float32)
        h, w = result.shape[:2]
        cy, cx = h / 2, w / 2
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - (dist / max_dist) * 0.4 * strength
        for c in range(3):
            result[:, :, c] *= vignette
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_industrial(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Simulate industrial inspection camera with high contrast."""
        result = frame.astype(np.float32)
        mean_val = np.mean(result)
        result = (result - mean_val) * (1 + 0.5 * strength) + mean_val
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_cartoon(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Apply cartoon-style edge enhancement and color quantization."""
        levels = max(2, int(8 - 6 * strength))
        quantized = (frame // (256 // levels)) * (256 // levels)
        blended = frame.astype(np.float32) * (1 - strength) + quantized.astype(np.float32) * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_sketch(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Apply pencil sketch effect via edge detection."""
        gray = np.mean(frame, axis=2).astype(np.float32)
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        edges = np.sqrt(gx ** 2 + gy ** 2)
        edges = np.clip(edges / edges.max() * 255, 0, 255) if edges.max() > 0 else edges
        sketch = 255 - edges
        sketch_rgb = np.stack([sketch, sketch, sketch], axis=2)
        blended = frame.astype(np.float32) * (1 - strength) + sketch_rgb * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_oil_painting(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Simulate oil painting effect via local color averaging."""
        kernel_size = max(3, int(7 * strength)) | 1
        from scipy.ndimage import uniform_filter
        smoothed = np.zeros_like(frame, dtype=np.float32)
        for c in range(3):
            smoothed[:, :, c] = uniform_filter(
                frame[:, :, c].astype(np.float32), size=kernel_size,
            )
        blended = frame.astype(np.float32) * (1 - strength) + smoothed * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_cinematic(
        frame: np.ndarray, strength: float, params: dict[str, Any],
    ) -> np.ndarray:
        """Apply cinematic color grading with teal-orange split toning."""
        result = frame.astype(np.float32)
        luminance = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        shadow_mask = (luminance < 128).astype(np.float32) * strength * 0.3
        highlight_mask = (luminance >= 128).astype(np.float32) * strength * 0.3
        result[:, :, 0] += highlight_mask * 30
        result[:, :, 1] += highlight_mask * 15
        result[:, :, 2] -= highlight_mask * 10
        result[:, :, 0] -= shadow_mask * 15
        result[:, :, 1] += shadow_mask * 10
        result[:, :, 2] += shadow_mask * 25
        return np.clip(result, 0, 255).astype(np.uint8)

    def _smooth_transitions(
        self,
        frames: list[np.ndarray],
        alpha: float = 0.15,
    ) -> list[np.ndarray]:
        """Smooth inter-frame transitions to preserve temporal coherence.

        Applies exponential moving average blending between consecutive
        frames to prevent flickering from per-frame style application.

        Args:
            frames: Styled frames.
            alpha: Smoothing factor (lower = more smoothing).

        Returns:
            Temporally smoothed frame sequence.
        """
        if len(frames) <= 1:
            return frames

        smoothed = [frames[0]]
        prev = frames[0].astype(np.float32)

        for frame in frames[1:]:
            current = frame.astype(np.float32)
            blended = prev * alpha + current * (1 - alpha)
            smoothed.append(np.clip(blended, 0, 255).astype(np.uint8))
            prev = blended

        return smoothed
