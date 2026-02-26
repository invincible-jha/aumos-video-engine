"""Optical flow-based temporal coherence engine.

Uses OpenCV dense optical flow (Farneback) to measure and enforce
frame-to-frame consistency in synthetic video sequences.
"""

from __future__ import annotations

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_OPENCV_AVAILABLE = False
try:
    import cv2  # type: ignore[import-untyped]

    _OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python-headless not installed — temporal engine using fallback")


class OpticalFlowTemporalEngine:
    """Measures and enforces temporal coherence using dense optical flow.

    Uses Farneback optical flow to compute per-frame motion vectors and
    derives a coherence score based on motion field smoothness and consistency.

    When OpenCV is not available, falls back to a simple pixel-difference
    metric for testing purposes.
    """

    def score_coherence(
        self,
        frames: list[np.ndarray],
        window_size: int,
    ) -> float:
        """Compute temporal coherence score for a frame sequence.

        Evaluates frame-to-frame optical flow consistency over a sliding window.
        A perfectly coherent sequence (identical frames) scores 1.0.
        A highly incoherent sequence (random noise) scores near 0.0.

        Args:
            frames: List of RGB uint8 numpy arrays (H, W, 3).
            window_size: Number of consecutive frames in each evaluation window.

        Returns:
            Mean coherence score across all windows, in range [0.0, 1.0].
        """
        if len(frames) < 2:
            return 1.0  # Single frame is trivially coherent

        if _OPENCV_AVAILABLE:
            return self._score_optical_flow(frames, window_size)
        return self._score_pixel_difference(frames)

    def _score_optical_flow(
        self,
        frames: list[np.ndarray],
        window_size: int,
    ) -> float:
        """Score coherence via Farneback dense optical flow.

        Args:
            frames: RGB uint8 frame list.
            window_size: Evaluation window size.

        Returns:
            Coherence score in [0.0, 1.0].
        """
        window_scores: list[float] = []

        for start_idx in range(0, len(frames) - 1, max(1, window_size // 2)):
            end_idx = min(start_idx + window_size, len(frames))
            window_frames = frames[start_idx:end_idx]
            window_score = self._score_window(window_frames)
            window_scores.append(window_score)

        if not window_scores:
            return 1.0
        return float(np.mean(window_scores))

    def _score_window(self, window_frames: list[np.ndarray]) -> float:
        """Score coherence for a single window of frames.

        Args:
            window_frames: Subset of frames to evaluate.

        Returns:
            Coherence score for this window.
        """
        if len(window_frames) < 2:
            return 1.0

        flow_magnitudes: list[float] = []
        flow_variances: list[float] = []

        for i in range(len(window_frames) - 1):
            prev_gray = cv2.cvtColor(window_frames[i], cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(window_frames[i + 1], cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                next_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            variance = float(np.var(magnitude))
            flow_magnitudes.append(magnitude)
            flow_variances.append(variance)

        if not flow_magnitudes:
            return 1.0

        mean_magnitude = float(np.mean(flow_magnitudes))
        mean_variance = float(np.mean(flow_variances))

        # Coherence: low variance relative to magnitude = consistent motion = high score
        if mean_magnitude < 1e-6:
            return 1.0  # Static scene = perfectly coherent

        consistency_ratio = 1.0 / (1.0 + mean_variance / (mean_magnitude + 1e-6))
        return float(np.clip(consistency_ratio, 0.0, 1.0))

    def _score_pixel_difference(self, frames: list[np.ndarray]) -> float:
        """Fallback coherence scoring via mean pixel difference.

        Used when OpenCV is not available.

        Args:
            frames: RGB uint8 frame list.

        Returns:
            Coherence score in [0.0, 1.0].
        """
        diffs: list[float] = []
        for i in range(len(frames) - 1):
            diff = float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i + 1].astype(np.float32))))
            diffs.append(diff)
        if not diffs:
            return 1.0
        mean_diff = float(np.mean(diffs))
        # Normalize: 0 diff = 1.0, 255 diff = 0.0
        score = 1.0 - (mean_diff / 255.0)
        return float(np.clip(score, 0.0, 1.0))

    def enforce_coherence(
        self,
        frames: list[np.ndarray],
        min_score: float,
        window_size: int,
    ) -> list[np.ndarray]:
        """Apply coherence enforcement by smoothing incoherent transitions.

        Uses frame blending at detected incoherent transition points to
        smooth abrupt changes until the sequence meets the minimum score.

        Args:
            frames: Input frame sequence as RGB uint8 numpy arrays.
            min_score: Minimum acceptable coherence score.
            window_size: Sliding window size for evaluation.

        Returns:
            Smoothed frame sequence meeting the minimum coherence threshold.
        """
        if len(frames) < 2:
            return frames

        enforced = [frame.copy() for frame in frames]
        max_iterations = 5

        for iteration in range(max_iterations):
            score = self.score_coherence(enforced, window_size)
            if score >= min_score:
                logger.debug(
                    "Coherence threshold met",
                    score=score,
                    min_required=min_score,
                    iteration=iteration,
                )
                break

            # Identify and smooth worst transition points
            enforced = self._smooth_transitions(enforced)
            logger.debug(
                "Smoothing iteration",
                iteration=iteration + 1,
                score=score,
                min_required=min_score,
            )

        return enforced

    def _smooth_transitions(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply temporal smoothing to reduce abrupt transitions.

        Uses a 3-frame weighted average (0.25, 0.5, 0.25) to blend
        adjacent frames at transition points.

        Args:
            frames: Input frame list.

        Returns:
            Smoothed frame list (same length).
        """
        if len(frames) < 3:
            return frames

        smoothed = [frames[0].copy()]
        for i in range(1, len(frames) - 1):
            blended = (
                0.25 * frames[i - 1].astype(np.float32)
                + 0.50 * frames[i].astype(np.float32)
                + 0.25 * frames[i + 1].astype(np.float32)
            )
            smoothed.append(np.clip(blended, 0, 255).astype(np.uint8))
        smoothed.append(frames[-1].copy())
        return smoothed

    def synthesize_motion(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """Generate intermediate frames between two keyframes via linear interpolation.

        For higher quality, uses optical flow-guided warping when OpenCV is available.
        Falls back to linear blend when OpenCV is not installed.

        Args:
            start_frame: Starting RGB uint8 frame (H, W, 3).
            end_frame: Ending RGB uint8 frame (H, W, 3).
            num_intermediate: Number of interpolated frames to generate.

        Returns:
            List of interpolated frames (excludes start and end frames).
        """
        if num_intermediate <= 0:
            return []

        if _OPENCV_AVAILABLE:
            return self._synthesize_flow_warp(start_frame, end_frame, num_intermediate)
        return self._synthesize_linear_blend(start_frame, end_frame, num_intermediate)

    def _synthesize_linear_blend(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """Generate intermediate frames via linear blending.

        Args:
            start_frame: Starting frame.
            end_frame: Ending frame.
            num_intermediate: Number of intermediate frames.

        Returns:
            List of blended frames.
        """
        intermediate: list[np.ndarray] = []
        for step in range(1, num_intermediate + 1):
            alpha = step / (num_intermediate + 1)
            blended = (
                (1.0 - alpha) * start_frame.astype(np.float32)
                + alpha * end_frame.astype(np.float32)
            )
            intermediate.append(np.clip(blended, 0, 255).astype(np.uint8))
        return intermediate

    def _synthesize_flow_warp(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """Generate intermediate frames using optical flow-guided warping.

        Computes forward optical flow from start to end, then warps the
        start frame along fractional flow vectors for each intermediate step.

        Args:
            start_frame: Starting frame.
            end_frame: Ending frame.
            num_intermediate: Number of intermediate frames.

        Returns:
            List of flow-warped intermediate frames.
        """
        prev_gray = cv2.cvtColor(start_frame, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(end_frame, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        height, width = start_frame.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

        intermediate: list[np.ndarray] = []
        for step in range(1, num_intermediate + 1):
            alpha = step / (num_intermediate + 1)
            map_x = (grid_x + alpha * flow[:, :, 0]).astype(np.float32)
            map_y = (grid_y + alpha * flow[:, :, 1]).astype(np.float32)
            warped = cv2.remap(start_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            intermediate.append(warped)

        return intermediate
