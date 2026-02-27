"""Motion generation and frame interpolation adapter.

Implements RIFE-style frame interpolation, motion vector synthesis, temporal
upsampling (e.g. 24 fps to 60 fps), motion blur simulation, camera motion
(pan/tilt/zoom), and physics-aware motion constraints for synthetic video.
"""

from __future__ import annotations

import asyncio
import math
from enum import Enum
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_OPENCV_AVAILABLE = False
try:
    import cv2  # type: ignore[import-untyped]

    _OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python-headless not installed — motion generator using numpy fallbacks")


class CameraMotionType(str, Enum):
    """Supported camera motion types for video generation."""

    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    STATIC = "static"
    COMBINED = "combined"


class MotionGenerator:
    """Generates smooth inter-frame motion via interpolation and camera simulation.

    Provides frame interpolation for temporal upsampling, optical-flow-guided
    motion vector synthesis, motion blur, camera motion transforms, and
    physics-aware velocity clamping.
    """

    def __init__(
        self,
        max_motion_magnitude: float = 50.0,
        motion_blur_kernel_size: int = 7,
        interpolation_iterations: int = 1,
    ) -> None:
        """Initialize MotionGenerator.

        Args:
            max_motion_magnitude: Physics constraint — maximum pixel displacement
                per frame for any generated motion vector. Higher values allow
                faster motion.
            motion_blur_kernel_size: Kernel size for motion blur simulation.
                Must be odd and positive.
            interpolation_iterations: Number of recursive interpolation passes.
                Each pass doubles temporal resolution. Value of 1 means one
                midpoint per pair; value of 2 means three midpoints, etc.
        """
        if motion_blur_kernel_size % 2 == 0 or motion_blur_kernel_size < 1:
            raise ValueError(f"motion_blur_kernel_size must be a positive odd integer, got {motion_blur_kernel_size}")
        self._max_magnitude = max_motion_magnitude
        self._blur_kernel_size = motion_blur_kernel_size
        self._interp_iterations = interpolation_iterations

    async def interpolate_frames(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """Generate intermediate frames between two keyframes (RIFE-style).

        Uses optical flow-guided warping when OpenCV is available,
        falling back to weighted linear blending otherwise.

        Args:
            frame_a: Starting keyframe (H, W, 3) RGB uint8.
            frame_b: Ending keyframe (H, W, 3) RGB uint8.
            num_intermediate: Number of frames to insert between frame_a and frame_b.

        Returns:
            List of interpolated frames (excludes frame_a and frame_b).
        """
        if num_intermediate <= 0:
            return []

        loop = asyncio.get_running_loop()
        interpolated = await loop.run_in_executor(
            None,
            self._interpolate_cpu,
            frame_a,
            frame_b,
            num_intermediate,
        )
        logger.debug(
            "Frame interpolation complete",
            num_intermediate=num_intermediate,
            method="optical_flow" if _OPENCV_AVAILABLE else "linear_blend",
        )
        return interpolated

    def _interpolate_cpu(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """CPU-bound frame interpolation.

        Args:
            frame_a: Start frame.
            frame_b: End frame.
            num_intermediate: Number of frames.

        Returns:
            Interpolated frame list.
        """
        if _OPENCV_AVAILABLE:
            return self._flow_warp_interpolate(frame_a, frame_b, num_intermediate)
        return self._linear_blend_interpolate(frame_a, frame_b, num_intermediate)

    def _flow_warp_interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """Optical flow-guided frame interpolation via bidirectional warping.

        Computes forward (A->B) and backward (B->A) optical flow, then blends
        warped versions of each frame at fractional time steps to produce
        perceptually smooth intermediate frames.

        Args:
            frame_a: Start frame.
            frame_b: End frame.
            num_intermediate: Number of intermediate frames.

        Returns:
            Interpolated frame list.
        """
        prev_gray = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

        flow_forward = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=4, winsize=15,
            iterations=5, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_backward = cv2.calcOpticalFlowFarneback(
            next_gray, prev_gray, None,
            pyr_scale=0.5, levels=4, winsize=15,
            iterations=5, poly_n=5, poly_sigma=1.2, flags=0,
        )

        height, width = frame_a.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

        intermediate: list[np.ndarray] = []
        for step in range(1, num_intermediate + 1):
            t = step / (num_intermediate + 1)

            # Forward-warped frame_a towards frame_b
            map_x_fwd = (grid_x + t * flow_forward[:, :, 0]).astype(np.float32)
            map_y_fwd = (grid_y + t * flow_forward[:, :, 1]).astype(np.float32)
            warped_a = cv2.remap(frame_a, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # Backward-warped frame_b towards frame_a
            map_x_bwd = (grid_x + (1.0 - t) * flow_backward[:, :, 0]).astype(np.float32)
            map_y_bwd = (grid_y + (1.0 - t) * flow_backward[:, :, 1]).astype(np.float32)
            warped_b = cv2.remap(frame_b, map_x_bwd, map_y_bwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # Blend: more from A at t=0, more from B at t=1
            blended = ((1.0 - t) * warped_a.astype(np.float32) + t * warped_b.astype(np.float32))
            intermediate.append(np.clip(blended, 0, 255).astype(np.uint8))

        return intermediate

    def _linear_blend_interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """Linear pixel-space interpolation fallback.

        Args:
            frame_a: Start frame.
            frame_b: End frame.
            num_intermediate: Number of intermediate frames.

        Returns:
            Interpolated frame list.
        """
        intermediate: list[np.ndarray] = []
        for step in range(1, num_intermediate + 1):
            t = step / (num_intermediate + 1)
            blended = (
                (1.0 - t) * frame_a.astype(np.float32)
                + t * frame_b.astype(np.float32)
            )
            intermediate.append(np.clip(blended, 0, 255).astype(np.uint8))
        return intermediate

    async def synthesize_motion_vectors(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Compute dense optical flow motion vectors for each consecutive frame pair.

        Returns a visualisation of motion vectors as RGB images where hue encodes
        direction and value encodes magnitude (HSV-style optical flow visualisation).

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).

        Returns:
            List of motion vector visualisation frames (H, W, 3) RGB uint8.
            Length is len(frames) - 1.
        """
        if len(frames) < 2:
            return []

        loop = asyncio.get_running_loop()
        motion_viz = await loop.run_in_executor(None, self._compute_motion_vectors, frames)
        logger.debug("Motion vectors synthesised", num_pairs=len(motion_viz))
        return motion_viz

    def _compute_motion_vectors(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """CPU-bound motion vector computation and visualisation.

        Args:
            frames: Frame sequence.

        Returns:
            Flow visualisation frames.
        """
        visualisations: list[np.ndarray] = []
        for i in range(len(frames) - 1):
            if _OPENCV_AVAILABLE:
                viz = self._flow_to_hsv_image(frames[i], frames[i + 1])
            else:
                # Fallback: difference image as motion proxy
                diff = np.abs(frames[i].astype(np.float32) - frames[i + 1].astype(np.float32))
                viz = np.clip(diff, 0, 255).astype(np.uint8)
            visualisations.append(viz)
        return visualisations

    def _flow_to_hsv_image(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> np.ndarray:
        """Convert optical flow to HSV colour-coded visualisation.

        Hue encodes flow direction, saturation is fixed at maximum,
        value encodes flow magnitude (brighter = faster motion).

        Args:
            frame_a: Previous frame.
            frame_b: Next frame.

        Returns:
            RGB uint8 visualisation of optical flow.
        """
        prev_gray = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv = np.zeros((frame_a.shape[0], frame_a.shape[1], 3), dtype=np.uint8)
        # Hue: direction
        hsv[:, :, 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
        # Saturation: full
        hsv[:, :, 1] = 255
        # Value: normalised magnitude
        max_mag = float(magnitude.max()) if magnitude.max() > 0 else 1.0
        hsv[:, :, 2] = np.clip((magnitude / max_mag) * 255, 0, 255).astype(np.uint8)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    async def temporal_upsample(
        self,
        frames: list[np.ndarray],
        source_fps: int,
        target_fps: int,
    ) -> list[np.ndarray]:
        """Upsample a video from source_fps to target_fps via frame interpolation.

        Computes the integer multiplier and inserts interpolated frames between
        each original frame pair. Only integer upsampling ratios are supported;
        the target_fps must be an integer multiple of source_fps.

        Args:
            frames: RGB uint8 source frames at source_fps.
            source_fps: Original frame rate (e.g., 24).
            target_fps: Target frame rate (e.g., 48 or 72).

        Returns:
            Upsampled frame list at target_fps.

        Raises:
            ValueError: If target_fps is not an integer multiple of source_fps,
                or if source_fps is zero.
        """
        if source_fps <= 0:
            raise ValueError(f"source_fps must be positive, got {source_fps}")
        if target_fps % source_fps != 0:
            raise ValueError(
                f"target_fps ({target_fps}) must be an integer multiple of source_fps ({source_fps})"
            )

        multiplier = target_fps // source_fps
        if multiplier == 1:
            return frames

        num_intermediate = multiplier - 1
        upsampled: list[np.ndarray] = []

        logger.info(
            "Temporal upsampling",
            source_fps=source_fps,
            target_fps=target_fps,
            multiplier=multiplier,
            source_frames=len(frames),
        )

        for i, frame in enumerate(frames):
            upsampled.append(frame)
            if i < len(frames) - 1:
                interp_frames = await self.interpolate_frames(frame, frames[i + 1], num_intermediate)
                upsampled.extend(interp_frames)

        logger.info(
            "Temporal upsampling complete",
            output_frames=len(upsampled),
            expected_frames=len(frames) * multiplier - (multiplier - 1),
        )
        return upsampled

    async def apply_motion_blur(
        self,
        frames: list[np.ndarray],
        blur_strength: float = 1.0,
        direction_degrees: float = 0.0,
    ) -> list[np.ndarray]:
        """Simulate directional motion blur on each frame.

        Creates a linear motion blur kernel in the specified direction and
        convolves each frame. Strength scales the kernel size.

        Args:
            frames: RGB uint8 frames to blur.
            blur_strength: Multiplier applied to the kernel size in [0.0, 5.0].
                0.0 means no blur; 1.0 uses the configured kernel size.
            direction_degrees: Direction of motion blur in degrees (0 = horizontal).

        Returns:
            Motion-blurred RGB uint8 frames.
        """
        if blur_strength <= 0.0 or not frames:
            return frames

        loop = asyncio.get_running_loop()
        blurred = await loop.run_in_executor(
            None,
            self._apply_motion_blur_cpu,
            frames,
            blur_strength,
            direction_degrees,
        )
        logger.debug(
            "Motion blur applied",
            num_frames=len(blurred),
            strength=blur_strength,
            direction=direction_degrees,
        )
        return blurred

    def _apply_motion_blur_cpu(
        self,
        frames: list[np.ndarray],
        blur_strength: float,
        direction_degrees: float,
    ) -> list[np.ndarray]:
        """CPU-bound motion blur application.

        Args:
            frames: Input frames.
            blur_strength: Kernel strength multiplier.
            direction_degrees: Blur direction.

        Returns:
            Blurred frames.
        """
        kernel_size = max(3, int(self._blur_kernel_size * blur_strength))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        centre = kernel_size // 2
        angle_rad = math.radians(direction_degrees)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        for i in range(kernel_size):
            offset = i - centre
            px = int(round(centre + offset * dx))
            py = int(round(centre + offset * dy))
            if 0 <= px < kernel_size and 0 <= py < kernel_size:
                kernel[py, px] = 1.0
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum

        blurred_frames: list[np.ndarray] = []
        for frame in frames:
            if _OPENCV_AVAILABLE:
                blurred = cv2.filter2D(frame, -1, kernel)
            else:
                # Manual channel convolution via numpy
                blurred = np.zeros_like(frame)
                for c in range(3):
                    channel = frame[:, :, c].astype(np.float32)
                    pad = kernel_size // 2
                    padded = np.pad(channel, pad, mode="edge")
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            if kernel[ki, kj] > 0:
                                blurred[:, :, c] += (
                                    kernel[ki, kj] * padded[ki:ki + channel.shape[0], kj:kj + channel.shape[1]]
                                )
                    blurred[:, :, c] = np.clip(blurred[:, :, c], 0, 255)
            blurred_frames.append(blurred.astype(np.uint8))

        return blurred_frames

    async def apply_camera_motion(
        self,
        frames: list[np.ndarray],
        motion_type: CameraMotionType,
        intensity: float = 0.05,
        motion_params: dict[str, Any] | None = None,
    ) -> list[np.ndarray]:
        """Simulate camera motion by applying geometric transforms across frames.

        Each frame is transformed progressively to simulate continuous camera
        movement. The intensity parameter controls the per-frame displacement
        or zoom factor.

        Args:
            frames: RGB uint8 source frames (H, W, 3).
            motion_type: Type of camera motion to simulate.
            intensity: Per-frame motion step. For pan/tilt: fraction of frame
                dimension. For zoom: scale factor increment per frame.
            motion_params: Optional overrides for specific motion configurations.
                For COMBINED, expects keys: pan_x, pan_y, zoom from motion_params.

        Returns:
            List of motion-applied RGB uint8 frames (same resolution as input).
        """
        if not frames or motion_type == CameraMotionType.STATIC:
            return frames

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._apply_camera_motion_cpu,
            frames,
            motion_type,
            intensity,
            motion_params or {},
        )
        logger.debug(
            "Camera motion applied",
            motion_type=motion_type.value,
            intensity=intensity,
            num_frames=len(result),
        )
        return result

    def _apply_camera_motion_cpu(
        self,
        frames: list[np.ndarray],
        motion_type: CameraMotionType,
        intensity: float,
        motion_params: dict[str, Any],
    ) -> list[np.ndarray]:
        """CPU-bound camera motion transform application.

        Args:
            frames: Input frames.
            motion_type: Motion type.
            intensity: Per-frame step magnitude.
            motion_params: Additional parameters.

        Returns:
            Transformed frames.
        """
        height, width = frames[0].shape[:2]
        result: list[np.ndarray] = []

        for frame_idx, frame in enumerate(frames):
            progress = frame_idx / max(len(frames) - 1, 1)

            if motion_type == CameraMotionType.PAN_LEFT:
                tx = -int(width * intensity * progress)
                ty = 0
                frame = self._translate_frame(frame, tx, ty)

            elif motion_type == CameraMotionType.PAN_RIGHT:
                tx = int(width * intensity * progress)
                ty = 0
                frame = self._translate_frame(frame, tx, ty)

            elif motion_type == CameraMotionType.TILT_UP:
                tx = 0
                ty = -int(height * intensity * progress)
                frame = self._translate_frame(frame, tx, ty)

            elif motion_type == CameraMotionType.TILT_DOWN:
                tx = 0
                ty = int(height * intensity * progress)
                frame = self._translate_frame(frame, tx, ty)

            elif motion_type == CameraMotionType.ZOOM_IN:
                scale = 1.0 + intensity * progress
                frame = self._zoom_frame(frame, scale)

            elif motion_type == CameraMotionType.ZOOM_OUT:
                scale = max(0.5, 1.0 - intensity * progress)
                frame = self._zoom_frame(frame, scale)

            elif motion_type == CameraMotionType.COMBINED:
                pan_x_factor = float(motion_params.get("pan_x", 0.0))
                pan_y_factor = float(motion_params.get("pan_y", 0.0))
                zoom_factor = float(motion_params.get("zoom", 0.0))

                tx = int(width * pan_x_factor * intensity * progress)
                ty = int(height * pan_y_factor * intensity * progress)
                frame = self._translate_frame(frame, tx, ty)

                if zoom_factor != 0.0:
                    scale = 1.0 + zoom_factor * intensity * progress
                    frame = self._zoom_frame(frame, max(0.5, scale))

            result.append(frame)

        return result

    def _translate_frame(
        self,
        frame: np.ndarray,
        tx: int,
        ty: int,
    ) -> np.ndarray:
        """Translate a frame by (tx, ty) pixels, filling edges by replication.

        Args:
            frame: RGB uint8 frame.
            tx: Horizontal translation (positive = shift right).
            ty: Vertical translation (positive = shift down).

        Returns:
            Translated RGB uint8 frame (same size).
        """
        if _OPENCV_AVAILABLE:
            height, width = frame.shape[:2]
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

        # Numpy fallback via roll + edge padding
        result = np.roll(frame, ty, axis=0)
        result = np.roll(result, tx, axis=1)
        return result

    def _zoom_frame(
        self,
        frame: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Zoom a frame by the given scale factor, cropping/padding to original size.

        Values greater than 1.0 zoom in (crop centre outwards);
        values less than 1.0 zoom out (letterbox).

        Args:
            frame: RGB uint8 frame (H, W, 3).
            scale: Zoom scale factor (> 0.0).

        Returns:
            Zoomed RGB uint8 frame (same size as input).
        """
        height, width = frame.shape[:2]

        if _OPENCV_AVAILABLE:
            centre_x = width / 2.0
            centre_y = height / 2.0
            zoom_matrix = cv2.getRotationMatrix2D((centre_x, centre_y), angle=0.0, scale=scale)
            return cv2.warpAffine(frame, zoom_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

        # Numpy fallback: crop and resize
        if scale >= 1.0:
            crop_w = int(width / scale)
            crop_h = int(height / scale)
            x_start = (width - crop_w) // 2
            y_start = (height - crop_h) // 2
            cropped = frame[y_start:y_start + crop_h, x_start:x_start + crop_w]
            # Upsample back to original size via nearest-neighbour
            row_idx = (np.arange(height) * crop_h / height).astype(int)
            col_idx = (np.arange(width) * crop_w / width).astype(int)
            return cropped[np.ix_(row_idx, col_idx)]
        else:
            # Zoom out — pad with border replication
            new_w = int(width * scale)
            new_h = int(height * scale)
            col_idx = (np.arange(new_w) * width / new_w).astype(int)
            row_idx = (np.arange(new_h) * height / new_h).astype(int)
            downsampled = frame[np.ix_(row_idx, col_idx)]
            padded = np.zeros_like(frame)
            x_off = (width - new_w) // 2
            y_off = (height - new_h) // 2
            padded[y_off:y_off + new_h, x_off:x_off + new_w] = downsampled
            # Replicate borders
            if y_off > 0:
                padded[:y_off] = padded[y_off:y_off + 1]
                padded[y_off + new_h:] = padded[y_off + new_h - 1:y_off + new_h]
            if x_off > 0:
                padded[:, :x_off] = padded[:, x_off:x_off + 1]
                padded[:, x_off + new_w:] = padded[:, x_off + new_w - 1:x_off + new_w]
            return padded

    def apply_physics_constraints(
        self,
        motion_vectors: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Clamp motion vectors to physically plausible magnitudes.

        Constrains each motion vector field so that no pixel displacement
        exceeds max_motion_magnitude per frame, preventing unrealistic
        instantaneous teleportation artefacts.

        Args:
            motion_vectors: List of float32 flow arrays (H, W, 2) with
                per-pixel (dx, dy) displacement.

        Returns:
            Magnitude-clamped motion vector list (same shapes).
        """
        clamped: list[np.ndarray] = []
        for flow in motion_vectors:
            magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
            scale_mask = np.where(
                magnitude > self._max_magnitude,
                self._max_magnitude / (magnitude + 1e-8),
                1.0,
            )
            clamped_flow = flow * scale_mask[:, :, np.newaxis]
            clamped.append(clamped_flow.astype(flow.dtype))

        logger.debug(
            "Physics constraints applied",
            num_vectors=len(clamped),
            max_magnitude=self._max_magnitude,
        )
        return clamped
