"""Protocol definitions for aumos-video-engine adapters.

These interfaces decouple the core business logic from specific infrastructure
implementations (SVD, BlenderProc, OpenCV, etc.), enabling testing and swapping.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from aumos_video_engine.core.models import VideoMetadata


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


@runtime_checkable
class VideoQualityEvaluatorProtocol(Protocol):
    """Protocol for multi-dimensional video quality evaluation.

    Implementations compute LPIPS perceptual similarity, optical flow
    consistency, temporal SSIM coherence, motion smoothness, and aggregate
    fidelity scores for synthetic video sequences.
    """

    async def score_lpips_per_frame(
        self,
        reference_frames: list[np.ndarray],
        generated_frames: list[np.ndarray],
    ) -> list[float]:
        """Compute per-frame LPIPS-approximated perceptual similarity scores.

        Args:
            reference_frames: Ground-truth or conditioning RGB uint8 frames (H, W, 3).
            generated_frames: Synthesised RGB uint8 frames of matching length.

        Returns:
            Per-frame perceptual similarity scores in [0.0, 1.0].
        """
        ...

    async def score_optical_flow_consistency(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Score optical flow consistency across consecutive frame pairs.

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).

        Returns:
            Flow consistency score in [0.0, 1.0].
        """
        ...

    async def score_temporal_coherence_ssim(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Measure temporal coherence via inter-frame SSIM stability.

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).

        Returns:
            SSIM stability score in [0.0, 1.0].
        """
        ...

    async def score_motion_smoothness(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Score motion smoothness via optical flow acceleration analysis.

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).

        Returns:
            Motion smoothness score in [0.0, 1.0].
        """
        ...

    async def aggregate_fidelity_score(
        self,
        frames: list[np.ndarray],
        reference_frames: list[np.ndarray] | None,
    ) -> float:
        """Compute overall video fidelity as a weighted metric aggregate.

        Args:
            frames: Generated RGB uint8 frame sequence.
            reference_frames: Optional reference frames for LPIPS comparison.

        Returns:
            Aggregate fidelity score in [0.0, 1.0].
        """
        ...

    async def generate_comparison_report(
        self,
        frames: list[np.ndarray],
        reference_frames: list[np.ndarray] | None,
        scene_transition_threshold: float,
    ) -> dict[str, Any]:
        """Generate a structured quality comparison report.

        Args:
            frames: Generated RGB uint8 frame sequence.
            reference_frames: Optional reference frames for LPIPS comparison.
            scene_transition_threshold: Threshold for scene transition detection.

        Returns:
            JSON-serialisable report dict.
        """
        ...


@runtime_checkable
class SceneCompositorProtocol(Protocol):
    """Protocol for multi-object scene composition with spatial reasoning.

    Implementations handle object placement, depth ordering, occlusion,
    lighting consistency, background blending, and temporal object tracking.
    """

    async def compose_sequence(
        self,
        objects: list[dict[str, Any]],
        scene_config: dict[str, Any],
        num_frames: int,
        resolution: tuple[int, int],
        background_frames: list[np.ndarray] | None,
    ) -> list[np.ndarray]:
        """Compose a full multi-frame video sequence from object descriptors.

        Args:
            objects: List of object descriptor dicts (object_id, label, position, etc.).
            scene_config: Scene-level configuration (lighting, background, parent_child).
            num_frames: Number of frames to generate.
            resolution: Output frame size as (width, height).
            background_frames: Optional per-frame background images.

        Returns:
            List of composed RGB uint8 frames (H, W, 3).
        """
        ...

    async def validate_composition_quality(
        self,
        frames: list[np.ndarray],
        scene_graph: Any,
        resolution: tuple[int, int],
    ) -> dict[str, Any]:
        """Validate composition quality (object visibility, coverage, occlusion).

        Args:
            frames: Composed frame sequence.
            scene_graph: Scene graph used for composition.
            resolution: Canvas resolution.

        Returns:
            Dict with quality assessment results.
        """
        ...


@runtime_checkable
class MotionGeneratorProtocol(Protocol):
    """Protocol for frame interpolation and motion synthesis.

    Implementations provide temporal upsampling, camera motion simulation,
    motion blur, and physics-constrained motion vector generation.
    """

    async def interpolate_frames(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        num_intermediate: int,
    ) -> list[np.ndarray]:
        """Generate intermediate frames between two keyframes.

        Args:
            frame_a: Starting keyframe (H, W, 3) RGB uint8.
            frame_b: Ending keyframe (H, W, 3) RGB uint8.
            num_intermediate: Number of frames to insert.

        Returns:
            Interpolated frames (excludes frame_a and frame_b).
        """
        ...

    async def temporal_upsample(
        self,
        frames: list[np.ndarray],
        source_fps: int,
        target_fps: int,
    ) -> list[np.ndarray]:
        """Upsample a video from source_fps to target_fps via interpolation.

        Args:
            frames: Source frames at source_fps.
            source_fps: Original frame rate.
            target_fps: Target frame rate (must be integer multiple of source_fps).

        Returns:
            Upsampled frame list.
        """
        ...

    async def apply_camera_motion(
        self,
        frames: list[np.ndarray],
        motion_type: Any,
        intensity: float,
        motion_params: dict[str, Any] | None,
    ) -> list[np.ndarray]:
        """Simulate camera motion via progressive geometric transforms.

        Args:
            frames: Source RGB uint8 frames.
            motion_type: Camera motion type enum value.
            intensity: Per-frame motion step magnitude.
            motion_params: Optional additional parameters.

        Returns:
            Motion-transformed RGB uint8 frames.
        """
        ...


@runtime_checkable
class VideoExportHandlerProtocol(Protocol):
    """Protocol for video file export and storage upload.

    Implementations encode frame sequences to standard video containers
    (MP4, WebM, AVI) and upload the results to MinIO/S3 storage.
    """

    async def export_mp4(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None,
        codec: Any,
        crf: int,
        preset: str,
        audio_bytes: bytes | None,
        metadata: dict[str, str] | None,
    ) -> bytes:
        """Encode frames to MP4 container with H.264 or H.265 codec.

        Args:
            frames: RGB uint8 frame list.
            fps: Target frames per second.
            output_resolution: Optional resize target.
            codec: VideoCodec enum value.
            crf: Constant Rate Factor quality setting.
            preset: Encoding speed/quality preset.
            audio_bytes: Optional PCM audio bytes.
            metadata: Optional metadata tags.

        Returns:
            MP4-encoded video as raw bytes.
        """
        ...

    async def export_webm(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None,
        crf: int,
        cpu_used: int,
        audio_bytes: bytes | None,
        metadata: dict[str, str] | None,
    ) -> bytes:
        """Encode frames to WebM container with VP9 codec.

        Args:
            frames: RGB uint8 frame list.
            fps: Target frames per second.
            output_resolution: Optional resize target.
            crf: Constant Rate Factor for VP9.
            cpu_used: VP9 encoding speed.
            audio_bytes: Optional PCM audio bytes.
            metadata: Optional metadata tags.

        Returns:
            WebM-encoded video as raw bytes.
        """
        ...

    async def upload_to_storage(
        self,
        video_bytes: bytes,
        job_id: str,
        tenant_id: str,
        container_format: Any,
    ) -> str:
        """Upload encoded video bytes to MinIO/S3 and return the URI.

        Args:
            video_bytes: Encoded video bytes.
            job_id: Job UUID string.
            tenant_id: Tenant UUID string.
            container_format: VideoContainer enum value.

        Returns:
            Full storage URI.
        """
        ...

    async def extract_thumbnail(
        self,
        frames: list[np.ndarray],
        frame_index: int | None,
    ) -> bytes:
        """Extract and JPEG-encode a thumbnail from the frame sequence.

        Args:
            frames: RGB uint8 frame sequence.
            frame_index: Frame to use; None selects the middle frame.

        Returns:
            JPEG-encoded thumbnail bytes.
        """
        ...


@runtime_checkable
class VideoMetadataExtractorProtocol(Protocol):
    """Protocol for semantic video metadata extraction.

    Implementations analyse synthetic video sequences to extract action
    recognition, object detection, scene classification, temporal events,
    face detection, motion analysis, and structured JSON metadata.
    """

    async def extract_metadata(
        self,
        frames: list[np.ndarray],
        fps: int,
        run_face_detection: bool,
        run_object_detection: bool,
        object_sample_rate: int,
    ) -> VideoMetadata:
        """Run the full metadata extraction pipeline.

        Args:
            frames: RGB uint8 video frame sequence.
            fps: Source video frame rate.
            run_face_detection: Whether to run face detection.
            run_object_detection: Whether to run object detection.
            object_sample_rate: Frame sampling rate for object/face detection.

        Returns:
            Fully populated VideoMetadata instance.
        """
        ...

    def export_metadata_json(self, metadata: VideoMetadata) -> dict[str, Any]:
        """Serialise VideoMetadata to a JSON-serialisable dict.

        Args:
            metadata: Fully populated VideoMetadata instance.

        Returns:
            JSON-serialisable dict.
        """
        ...
