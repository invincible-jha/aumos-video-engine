"""Video quality evaluation adapter.

Computes multi-dimensional quality scores for synthetic video sequences:
LPIPS perceptual similarity, optical flow consistency, temporal coherence via
SSIM, scene transition accuracy, motion smoothness, and aggregated fidelity.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_OPENCV_AVAILABLE = False
try:
    import cv2  # type: ignore[import-untyped]

    _OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python-headless not installed — quality evaluator using numpy fallbacks")

_SKIMAGE_AVAILABLE = False
try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore[import-untyped]

    _SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning("scikit-image not installed — SSIM using simplified fallback")


class VideoQualityEvaluator:
    """Evaluates synthetic video fidelity across multiple perceptual dimensions.

    Provides per-frame LPIPS approximation, optical flow consistency scoring,
    temporal SSIM coherence, scene transition detection, motion smoothness,
    and an aggregate fidelity score with a structured comparison report.
    """

    def __init__(
        self,
        lpips_weight: float = 0.30,
        flow_weight: float = 0.25,
        ssim_weight: float = 0.25,
        motion_weight: float = 0.20,
    ) -> None:
        """Initialize VideoQualityEvaluator with configurable metric weights.

        Args:
            lpips_weight: Weight of LPIPS score in final fidelity aggregate.
            flow_weight: Weight of optical flow consistency in final aggregate.
            ssim_weight: Weight of temporal SSIM in final aggregate.
            motion_weight: Weight of motion smoothness in final aggregate.

        Raises:
            ValueError: If weights do not sum to approximately 1.0.
        """
        total = lpips_weight + flow_weight + ssim_weight + motion_weight
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Metric weights must sum to 1.0, got {total:.4f}. "
                "Adjust lpips_weight, flow_weight, ssim_weight, or motion_weight."
            )
        self._lpips_weight = lpips_weight
        self._flow_weight = flow_weight
        self._ssim_weight = ssim_weight
        self._motion_weight = motion_weight

    async def score_lpips_per_frame(
        self,
        reference_frames: list[np.ndarray],
        generated_frames: list[np.ndarray],
    ) -> list[float]:
        """Compute per-frame LPIPS-approximated perceptual similarity scores.

        Uses a multi-scale gradient-magnitude difference as a differentiable
        perceptual proxy when the full lpips library is not available. Scores
        are normalised to [0.0, 1.0] where 1.0 means perceptually identical.

        Args:
            reference_frames: Ground-truth or conditioning RGB uint8 frames (H, W, 3).
            generated_frames: Synthesised RGB uint8 frames of matching length.

        Returns:
            Per-frame perceptual similarity scores in [0.0, 1.0].

        Raises:
            ValueError: If frame lists have different lengths or are empty.
        """
        if len(reference_frames) != len(generated_frames):
            raise ValueError(
                f"Frame list length mismatch: reference={len(reference_frames)}, "
                f"generated={len(generated_frames)}"
            )
        if not reference_frames:
            raise ValueError("Frame lists must not be empty")

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None,
            self._compute_lpips_cpu,
            reference_frames,
            generated_frames,
        )
        logger.debug(
            "LPIPS per-frame scoring complete",
            num_frames=len(scores),
            mean_score=float(np.mean(scores)),
        )
        return scores

    def _compute_lpips_cpu(
        self,
        reference_frames: list[np.ndarray],
        generated_frames: list[np.ndarray],
    ) -> list[float]:
        """CPU-bound LPIPS approximation using multi-scale gradient features.

        Args:
            reference_frames: Reference RGB uint8 frames.
            generated_frames: Generated RGB uint8 frames.

        Returns:
            Per-frame similarity scores.
        """
        scores: list[float] = []
        for ref, gen in zip(reference_frames, generated_frames, strict=True):
            score = self._lpips_approximate(ref, gen)
            scores.append(score)
        return scores

    def _lpips_approximate(
        self,
        reference: np.ndarray,
        generated: np.ndarray,
    ) -> float:
        """Approximate LPIPS using multi-scale gradient magnitude difference.

        Computes Sobel gradients at two scales and measures cosine similarity
        of flattened gradient feature vectors as a perceptual proxy.

        Args:
            reference: Reference frame (H, W, 3) RGB uint8.
            generated: Generated frame (H, W, 3) RGB uint8.

        Returns:
            Perceptual similarity in [0.0, 1.0].
        """
        ref_f = reference.astype(np.float32) / 255.0
        gen_f = generated.astype(np.float32) / 255.0

        if _OPENCV_AVAILABLE:
            ref_gray = cv2.cvtColor((ref_f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gen_gray = cv2.cvtColor((gen_f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

            # Full-scale Sobel gradients
            ref_gx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
            ref_gy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
            gen_gx = cv2.Sobel(gen_gray, cv2.CV_32F, 1, 0, ksize=3)
            gen_gy = cv2.Sobel(gen_gray, cv2.CV_32F, 0, 1, ksize=3)

            ref_mag = np.sqrt(ref_gx**2 + ref_gy**2).flatten()
            gen_mag = np.sqrt(gen_gx**2 + gen_gy**2).flatten()

            # Half-scale
            ref_small = cv2.resize(ref_gray, (ref_gray.shape[1] // 2, ref_gray.shape[0] // 2))
            gen_small = cv2.resize(gen_gray, (gen_gray.shape[1] // 2, gen_gray.shape[0] // 2))
            ref_gx2 = cv2.Sobel(ref_small, cv2.CV_32F, 1, 0, ksize=3)
            ref_gy2 = cv2.Sobel(ref_small, cv2.CV_32F, 0, 1, ksize=3)
            gen_gx2 = cv2.Sobel(gen_small, cv2.CV_32F, 1, 0, ksize=3)
            gen_gy2 = cv2.Sobel(gen_small, cv2.CV_32F, 0, 1, ksize=3)
            ref_mag2 = np.sqrt(ref_gx2**2 + ref_gy2**2).flatten()
            gen_mag2 = np.sqrt(gen_gx2**2 + gen_gy2**2).flatten()

            ref_feat = np.concatenate([ref_mag, ref_mag2])
            gen_feat = np.concatenate([gen_mag, gen_mag2])
        else:
            # Pure numpy fallback: pixel-level L1 distance
            ref_feat = ref_f.flatten()
            gen_feat = gen_f.flatten()

        norm_ref = np.linalg.norm(ref_feat)
        norm_gen = np.linalg.norm(gen_feat)
        if norm_ref < 1e-8 or norm_gen < 1e-8:
            # Both frames are near-uniform — treat as identical
            pixel_diff = float(np.mean(np.abs(ref_f - gen_f)))
            return float(np.clip(1.0 - pixel_diff, 0.0, 1.0))

        cosine_sim = float(np.dot(ref_feat, gen_feat) / (norm_ref * norm_gen))
        # Map cosine similarity from [-1,1] to [0,1]
        return float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

    async def score_optical_flow_consistency(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Score optical flow consistency across consecutive frame pairs.

        Computes Farneback dense optical flow for each adjacent pair and
        measures how stable the flow field magnitude is over time. Stable,
        slowly-varying flow indicates consistent motion.

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).

        Returns:
            Flow consistency score in [0.0, 1.0] (1.0 = perfectly consistent).
        """
        if len(frames) < 2:
            return 1.0

        loop = asyncio.get_running_loop()
        score = await loop.run_in_executor(None, self._compute_flow_consistency, frames)
        logger.debug("Optical flow consistency score", score=score, num_frames=len(frames))
        return score

    def _compute_flow_consistency(self, frames: list[np.ndarray]) -> float:
        """CPU-bound optical flow consistency computation.

        Args:
            frames: Frame sequence.

        Returns:
            Consistency score in [0.0, 1.0].
        """
        if not _OPENCV_AVAILABLE:
            return self._flow_consistency_fallback(frames)

        magnitudes: list[float] = []
        for i in range(len(frames) - 1):
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            magnitudes.append(mag)

        if not magnitudes:
            return 1.0

        magnitude_array = np.array(magnitudes)
        mean_mag = float(np.mean(magnitude_array))
        std_mag = float(np.std(magnitude_array))

        if mean_mag < 1e-6:
            return 1.0  # Static scene

        # Coefficient of variation — lower = more consistent
        cv = std_mag / (mean_mag + 1e-6)
        consistency = 1.0 / (1.0 + cv)
        return float(np.clip(consistency, 0.0, 1.0))

    def _flow_consistency_fallback(self, frames: list[np.ndarray]) -> float:
        """Numpy fallback for flow consistency when OpenCV is unavailable.

        Args:
            frames: Frame sequence.

        Returns:
            Consistency score proxy.
        """
        diffs: list[float] = []
        for i in range(len(frames) - 1):
            diff = float(np.mean(np.abs(
                frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)
            )))
            diffs.append(diff)

        if not diffs:
            return 1.0
        diff_arr = np.array(diffs)
        cv = float(np.std(diff_arr)) / (float(np.mean(diff_arr)) + 1e-6)
        return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

    async def score_temporal_coherence_ssim(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Measure temporal coherence by computing inter-frame SSIM stability.

        SSIM values between consecutive frames are computed and their variance
        is used to assess how stable the visual quality is over time. A sequence
        with consistently high inter-frame SSIM indicates good temporal coherence.

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).

        Returns:
            SSIM stability score in [0.0, 1.0].
        """
        if len(frames) < 2:
            return 1.0

        loop = asyncio.get_running_loop()
        score = await loop.run_in_executor(None, self._compute_ssim_stability, frames)
        logger.debug("Temporal SSIM coherence score", score=score)
        return score

    def _compute_ssim_stability(self, frames: list[np.ndarray]) -> float:
        """CPU-bound SSIM stability computation.

        Args:
            frames: Frame sequence.

        Returns:
            SSIM stability score.
        """
        ssim_values: list[float] = []
        for i in range(len(frames) - 1):
            frame_a = frames[i]
            frame_b = frames[i + 1]
            ssim_val = self._frame_ssim(frame_a, frame_b)
            ssim_values.append(ssim_val)

        if not ssim_values:
            return 1.0

        mean_ssim = float(np.mean(ssim_values))
        std_ssim = float(np.std(ssim_values))
        # Penalise large variance — stable sequences have consistent SSIM
        stability = mean_ssim * (1.0 - min(std_ssim, 0.5))
        return float(np.clip(stability, 0.0, 1.0))

    def _frame_ssim(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Compute SSIM between two frames.

        Uses scikit-image if available, otherwise a simplified luminance-contrast proxy.

        Args:
            frame_a: First frame (H, W, 3) RGB uint8.
            frame_b: Second frame (H, W, 3) RGB uint8.

        Returns:
            SSIM value in [-1.0, 1.0], clipped to [0.0, 1.0].
        """
        if _SKIMAGE_AVAILABLE:
            result = ssim(frame_a, frame_b, channel_axis=2, data_range=255)
            return float(np.clip(result, 0.0, 1.0))

        # Simplified proxy: normalised cross-correlation of luminance
        if _OPENCV_AVAILABLE:
            gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray_a = frame_a.mean(axis=2).astype(np.float32)
            gray_b = frame_b.mean(axis=2).astype(np.float32)

        mean_a = float(np.mean(gray_a))
        mean_b = float(np.mean(gray_b))
        std_a = float(np.std(gray_a))
        std_b = float(np.std(gray_b))

        if std_a < 1e-6 or std_b < 1e-6:
            # Uniform frames — score on brightness similarity
            return float(np.clip(1.0 - abs(mean_a - mean_b) / 255.0, 0.0, 1.0))

        covariance = float(np.mean((gray_a - mean_a) * (gray_b - mean_b)))
        correlation = covariance / (std_a * std_b)
        return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))

    async def detect_scene_transitions(
        self,
        frames: list[np.ndarray],
        threshold: float = 0.4,
    ) -> list[int]:
        """Detect scene transition frame indices using histogram difference.

        Computes colour histogram differences between consecutive frames and
        identifies frames where the difference exceeds the threshold as scene
        transition points.

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).
            threshold: Histogram distance threshold in [0.0, 1.0].
                Higher values detect only hard cuts; lower values also detect
                gradual transitions.

        Returns:
            List of frame indices (1-based) where scene transitions were detected.
        """
        if len(frames) < 2:
            return []

        loop = asyncio.get_running_loop()
        transition_indices = await loop.run_in_executor(
            None,
            self._find_transitions,
            frames,
            threshold,
        )
        logger.debug(
            "Scene transitions detected",
            num_transitions=len(transition_indices),
            threshold=threshold,
        )
        return transition_indices

    def _find_transitions(
        self,
        frames: list[np.ndarray],
        threshold: float,
    ) -> list[int]:
        """CPU-bound scene transition detection.

        Args:
            frames: Frame sequence.
            threshold: Detection threshold.

        Returns:
            List of transition frame indices.
        """
        transitions: list[int] = []
        for i in range(len(frames) - 1):
            distance = self._histogram_distance(frames[i], frames[i + 1])
            if distance > threshold:
                transitions.append(i + 1)
        return transitions

    def _histogram_distance(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> float:
        """Compute normalised histogram intersection distance between two frames.

        Args:
            frame_a: First frame RGB uint8.
            frame_b: Second frame RGB uint8.

        Returns:
            Distance in [0.0, 1.0] where 1.0 means completely different.
        """
        if _OPENCV_AVAILABLE:
            hist_a = cv2.calcHist([frame_a], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
            hist_b = cv2.calcHist([frame_b], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_a, hist_a, alpha=1.0, norm_type=cv2.NORM_L1)
            cv2.normalize(hist_b, hist_b, alpha=1.0, norm_type=cv2.NORM_L1)
            intersection = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_INTERSECT))
            return float(np.clip(1.0 - intersection, 0.0, 1.0))

        # Numpy fallback: normalised L1 on 8-bin per-channel histograms
        distance_per_channel: list[float] = []
        for channel in range(3):
            hist_a, _ = np.histogram(frame_a[..., channel], bins=8, range=(0, 256))
            hist_b, _ = np.histogram(frame_b[..., channel], bins=8, range=(0, 256))
            hist_a = hist_a.astype(np.float32) / (hist_a.sum() + 1e-8)
            hist_b = hist_b.astype(np.float32) / (hist_b.sum() + 1e-8)
            intersection = float(np.minimum(hist_a, hist_b).sum())
            distance_per_channel.append(1.0 - intersection)
        return float(np.mean(distance_per_channel))

    async def score_motion_smoothness(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Score motion smoothness by measuring acceleration in optical flow fields.

        Computes the second-order difference of per-frame flow magnitudes.
        Low acceleration variance indicates smooth, physically plausible motion.

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).

        Returns:
            Motion smoothness score in [0.0, 1.0] (1.0 = perfectly smooth).
        """
        if len(frames) < 3:
            return 1.0

        loop = asyncio.get_running_loop()
        score = await loop.run_in_executor(None, self._compute_motion_smoothness, frames)
        logger.debug("Motion smoothness score", score=score)
        return score

    def _compute_motion_smoothness(self, frames: list[np.ndarray]) -> float:
        """CPU-bound motion smoothness via flow acceleration.

        Args:
            frames: Frame sequence.

        Returns:
            Smoothness score in [0.0, 1.0].
        """
        magnitudes: list[float] = []

        if _OPENCV_AVAILABLE:
            for i in range(len(frames) - 1):
                prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, next_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
                magnitudes.append(mag)
        else:
            for i in range(len(frames) - 1):
                diff = float(np.mean(np.abs(
                    frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)
                )))
                magnitudes.append(diff)

        if len(magnitudes) < 2:
            return 1.0

        # Acceleration = second difference of magnitudes
        mag_array = np.array(magnitudes)
        acceleration = np.diff(mag_array)
        accel_std = float(np.std(acceleration))
        mean_mag = float(np.mean(mag_array))

        if mean_mag < 1e-6:
            return 1.0

        # Normalised acceleration variance
        smoothness = 1.0 / (1.0 + accel_std / (mean_mag + 1e-6))
        return float(np.clip(smoothness, 0.0, 1.0))

    async def aggregate_fidelity_score(
        self,
        frames: list[np.ndarray],
        reference_frames: list[np.ndarray] | None = None,
    ) -> float:
        """Compute overall video fidelity as a weighted aggregate of all metrics.

        When reference frames are provided, LPIPS is computed against them.
        Otherwise, LPIPS is substituted by a self-consistency proxy (same weight).

        Args:
            frames: Generated RGB uint8 frame sequence (H, W, 3).
            reference_frames: Optional ground-truth frames for LPIPS comparison.

        Returns:
            Aggregate fidelity score in [0.0, 1.0].
        """
        flow_task = asyncio.create_task(self.score_optical_flow_consistency(frames))
        ssim_task = asyncio.create_task(self.score_temporal_coherence_ssim(frames))
        motion_task = asyncio.create_task(self.score_motion_smoothness(frames))

        if reference_frames is not None:
            lpips_scores = await self.score_lpips_per_frame(reference_frames, frames)
            lpips_score = float(np.mean(lpips_scores))
        else:
            # Self-consistency proxy: LPIPS between odd and even frames
            if len(frames) >= 2:
                pairs_ref = frames[:-1]
                pairs_gen = frames[1:]
                lpips_scores = await self.score_lpips_per_frame(pairs_ref, pairs_gen)
                lpips_score = float(np.mean(lpips_scores))
            else:
                lpips_score = 1.0

        flow_score = await flow_task
        ssim_score = await ssim_task
        motion_score = await motion_task

        fidelity = (
            self._lpips_weight * lpips_score
            + self._flow_weight * flow_score
            + self._ssim_weight * ssim_score
            + self._motion_weight * motion_score
        )

        logger.info(
            "Video fidelity aggregated",
            lpips=lpips_score,
            flow_consistency=flow_score,
            ssim_stability=ssim_score,
            motion_smoothness=motion_score,
            fidelity=fidelity,
        )
        return float(np.clip(fidelity, 0.0, 1.0))

    async def generate_comparison_report(
        self,
        frames: list[np.ndarray],
        reference_frames: list[np.ndarray] | None = None,
        scene_transition_threshold: float = 0.4,
    ) -> dict[str, Any]:
        """Generate a structured quality comparison report for a video sequence.

        Runs all metrics in parallel and assembles results into a JSON-serialisable
        dict suitable for storage in JSONB columns or event payloads.

        Args:
            frames: Generated RGB uint8 frame sequence.
            reference_frames: Optional reference frames for LPIPS comparison.
            scene_transition_threshold: Threshold for scene transition detection.

        Returns:
            Dict with keys: num_frames, metrics (sub-dict), scene_transitions,
            aggregate_fidelity, evaluation_notes.
        """
        logger.info("Generating quality comparison report", num_frames=len(frames))

        flow_task = asyncio.create_task(self.score_optical_flow_consistency(frames))
        ssim_task = asyncio.create_task(self.score_temporal_coherence_ssim(frames))
        motion_task = asyncio.create_task(self.score_motion_smoothness(frames))
        transition_task = asyncio.create_task(
            self.detect_scene_transitions(frames, threshold=scene_transition_threshold)
        )

        if reference_frames is not None:
            per_frame_lpips = await self.score_lpips_per_frame(reference_frames, frames)
            lpips_mean = float(np.mean(per_frame_lpips))
            lpips_min = float(np.min(per_frame_lpips))
        else:
            per_frame_lpips = []
            lpips_mean = 0.0
            lpips_min = 0.0

        flow_score = await flow_task
        ssim_score = await ssim_task
        motion_score = await motion_task
        transitions = await transition_task

        aggregate = (
            self._lpips_weight * (lpips_mean if reference_frames else 1.0)
            + self._flow_weight * flow_score
            + self._ssim_weight * ssim_score
            + self._motion_weight * motion_score
        )
        aggregate = float(np.clip(aggregate, 0.0, 1.0))

        notes: list[str] = []
        if aggregate < 0.5:
            notes.append("Low fidelity — consider re-generation or increased inference steps.")
        if len(transitions) > max(1, len(frames) // 10):
            notes.append(f"High scene transition count ({len(transitions)}) — may indicate flickering.")
        if ssim_score < 0.6:
            notes.append("Low SSIM stability — temporal coherence enforcement recommended.")

        return {
            "num_frames": len(frames),
            "metrics": {
                "lpips_mean": lpips_mean,
                "lpips_min": lpips_min,
                "per_frame_lpips": per_frame_lpips,
                "optical_flow_consistency": flow_score,
                "temporal_ssim_stability": ssim_score,
                "motion_smoothness": motion_score,
            },
            "weights": {
                "lpips": self._lpips_weight,
                "flow": self._flow_weight,
                "ssim": self._ssim_weight,
                "motion": self._motion_weight,
            },
            "scene_transitions": transitions,
            "aggregate_fidelity": aggregate,
            "evaluation_notes": notes,
            "reference_provided": reference_frames is not None,
        }
