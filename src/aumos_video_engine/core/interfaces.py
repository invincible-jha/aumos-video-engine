"""Protocol definitions for aumos-video-engine adapters.

These interfaces decouple the core business logic from specific infrastructure
implementations (SVD, BlenderProc, OpenCV, etc.), enabling testing and swapping.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameGeneratorProtocol(Protocol):
    """Protocol for synthetic video frame generators.

    Implementations include StableVideoDiffusionGenerator and
    BlenderProcSceneGenerator.
    """

    async def generate_frames(
        self,
        prompt: str,
        num_frames: int,
        fps: int,
        resolution: tuple[int, int],
        model_config: dict[str, Any],
        reference_image: bytes | None,
    ) -> list[np.ndarray]:
        """Generate a sequence of video frames.

        Args:
            prompt: Text description of the video content.
            num_frames: Number of frames to generate.
            fps: Target frames per second (used for motion speed calibration).
            resolution: Output frame resolution as (width, height).
            model_config: Model-specific parameters (seed, guidance_scale, etc.).
            reference_image: Optional conditioning image bytes (PNG/JPEG).

        Returns:
            List of numpy arrays with shape (H, W, 3) in RGB uint8 format.
        """
        ...

    async def is_available(self) -> bool:
        """Check whether this generator is available (model loaded, GPU ready).

        Returns:
            True if the generator can accept requests.
        """
        ...


@runtime_checkable
class SceneComposerProtocol(Protocol):
    """Protocol for 3D scene composition adapters.

    Implementations render structured 3D scenes into video frame sequences
    using BlenderProc or similar renderers.
    """

    async def compose_scene(
        self,
        scene_config: dict[str, Any],
        objects: list[dict[str, Any]],
        num_frames: int,
        fps: int,
        resolution: tuple[int, int],
    ) -> list[np.ndarray]:
        """Render a 3D scene into a sequence of video frames.

        Args:
            scene_config: Scene parameters (lighting, camera_path, environment).
            objects: List of 3D object descriptors with position/rotation/scale.
            num_frames: Number of frames to render.
            fps: Target frames per second.
            resolution: Output resolution as (width, height).

        Returns:
            List of numpy arrays with shape (H, W, 3) in RGB uint8 format.
        """
        ...

    async def is_available(self) -> bool:
        """Check whether the scene renderer is available.

        Returns:
            True if the renderer can accept requests.
        """
        ...


@runtime_checkable
class TemporalEngineProtocol(Protocol):
    """Protocol for temporal coherence enforcement engines.

    Measures and enforces frame-to-frame consistency in synthetic video,
    detecting motion artifacts and incoherent transitions.
    """

    def score_coherence(
        self,
        frames: list[np.ndarray],
        window_size: int,
    ) -> float:
        """Compute temporal coherence score for a frame sequence.

        Uses optical flow analysis to measure frame-to-frame consistency.

        Args:
            frames: List of RGB uint8 numpy arrays (H, W, 3).
            window_size: Number of frames in sliding evaluation window.

        Returns:
            Coherence score in range [0.0, 1.0] (1.0 = perfectly coherent).
        """
        ...

    def enforce_coherence(
        self,
        frames: list[np.ndarray],
        min_score: float,
        window_size: int,
    ) -> list[np.ndarray]:
        """Apply coherence enforcement to a frame sequence.

        Smooths incoherent transitions to meet the minimum score threshold.

        Args:
            frames: Input frame sequence as RGB uint8 numpy arrays.
            min_score: Minimum acceptable coherence score.
            window_size: Sliding window size for evaluation.

        Returns:
            Coherence-enforced frame sequence (same length as input).
        """
        ...

    def synthesize_motion(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """Generate intermediate frames between two keyframes via optical flow.

        Args:
            start_frame: Starting keyframe (H, W, 3) RGB uint8.
            end_frame: Ending keyframe (H, W, 3) RGB uint8.
            num_intermediate: Number of interpolated frames to generate.

        Returns:
            List of interpolated frames (does not include start/end frames).
        """
        ...


@runtime_checkable
class PrivacyEnforcerProtocol(Protocol):
    """Protocol for per-frame privacy enforcement.

    Detects and redacts PII from video frames: faces, license plates,
    and other identifiable information.
    """

    async def enforce_frame(
        self,
        frame: np.ndarray,
        blur_faces: bool,
        redact_plates: bool,
        remove_pii: bool,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Apply privacy enforcement to a single video frame.

        Args:
            frame: RGB uint8 numpy array (H, W, 3).
            blur_faces: Whether to blur detected faces.
            redact_plates: Whether to redact license plates.
            remove_pii: Whether to remove other PII (text, badges, etc.).

        Returns:
            Tuple of (processed_frame, detection_counts) where detection_counts
            maps entity type to number of detections (e.g., {"faces": 2, "plates": 1}).
        """
        ...

    async def enforce_batch(
        self,
        frames: list[np.ndarray],
        blur_faces: bool,
        redact_plates: bool,
        remove_pii: bool,
    ) -> tuple[list[np.ndarray], dict[str, int]]:
        """Apply privacy enforcement to a batch of frames.

        Batch processing is more efficient than per-frame calls for large sequences.

        Args:
            frames: List of RGB uint8 numpy arrays (H, W, 3).
            blur_faces: Whether to blur detected faces.
            redact_plates: Whether to redact license plates.
            remove_pii: Whether to remove other PII.

        Returns:
            Tuple of (processed_frames, aggregate_detection_counts).
        """
        ...
