"""Video metadata extraction adapter.

Analyses synthetic video frame sequences to extract structured metadata:
action recognition, object detection and tracking, scene classification,
temporal event detection, face detection, motion analysis, and JSON export.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

# Import canonical domain value objects from core — adapters must not define them
from aumos_video_engine.core.models import (
    VideoBoundingBox as BoundingBox,
    VideoMetadata,
    VideoTemporalEvent as TemporalEvent,
)

logger = get_logger(__name__)

_OPENCV_AVAILABLE = False
try:
    import cv2  # type: ignore[import-untyped]

    _OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python-headless not installed — metadata extractor using numpy fallbacks")


class VideoMetadataExtractor:
    """Extracts structured semantic metadata from synthetic video frame sequences.

    Uses OpenCV for local detection (face detection via Haar cascades, optical
    flow for motion analysis). Scene and action classification use colour/texture
    heuristics when deep learning models are not available.
    """

    _HAAR_CASCADE_FACE = "haarcascade_frontalface_default.xml"

    # Colour-based scene heuristics (dominant colour signature -> scene label)
    _SCENE_COLOUR_SIGNATURES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
        "outdoor_daylight": ((80, 120, 150), (200, 230, 255)),
        "indoor_artificial": ((100, 80, 70), (200, 180, 160)),
        "manufacturing_floor": ((50, 60, 50), (150, 160, 140)),
        "road_intersection": ((80, 80, 80), (180, 180, 180)),
        "unknown": ((0, 0, 0), (255, 255, 255)),
    }

    # Motion-based action heuristics
    _ACTION_MOTION_THRESHOLDS: dict[str, tuple[float, float]] = {
        "static_idle": (0.0, 2.0),
        "slow_walking": (2.0, 8.0),
        "moderate_activity": (8.0, 20.0),
        "fast_movement": (20.0, 50.0),
        "rapid_action": (50.0, float("inf")),
    }

    def __init__(
        self,
        face_detection_scale_factor: float = 1.1,
        face_detection_min_neighbours: int = 5,
        object_detection_confidence_threshold: float = 0.5,
        motion_analysis_window_size: int = 8,
    ) -> None:
        """Initialize VideoMetadataExtractor.

        Args:
            face_detection_scale_factor: Scale factor for Haar cascade face detection.
            face_detection_min_neighbours: Minimum neighbours for face detection
                (higher = fewer false positives).
            object_detection_confidence_threshold: Minimum confidence for object
                detections to be included in output.
            motion_analysis_window_size: Frame window for motion aggregation.
        """
        self._face_scale_factor = face_detection_scale_factor
        self._face_min_neighbours = face_detection_min_neighbours
        self._obj_confidence_threshold = object_detection_confidence_threshold
        self._motion_window = motion_analysis_window_size

        self._face_cascade: Any = None
        if _OPENCV_AVAILABLE:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + self._HAAR_CASCADE_FACE
            )

    async def recognise_actions(
        self,
        frames: list[np.ndarray],
    ) -> dict[str, float]:
        """Classify the dominant action in the video based on motion statistics.

        Uses optical flow magnitude distribution across all frame pairs to
        categorise overall activity level into action classes.

        Args:
            frames: RGB uint8 frame sequence (H, W, 3).

        Returns:
            Dict mapping action label to confidence score (all sum to 1.0).
        """
        if len(frames) < 2:
            return {"static_idle": 1.0}

        loop = asyncio.get_running_loop()
        action_scores = await loop.run_in_executor(None, self._classify_actions_cpu, frames)
        logger.debug("Action recognition complete", top_action=max(action_scores, key=action_scores.get))  # type: ignore[arg-type]
        return action_scores

    def _classify_actions_cpu(self, frames: list[np.ndarray]) -> dict[str, float]:
        """CPU-bound action classification via motion magnitude histogramming.

        Args:
            frames: Input frames.

        Returns:
            Action score dict.
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

        mean_mag = float(np.mean(magnitudes)) if magnitudes else 0.0

        # Soft-assign to action categories using Gaussian membership
        raw_scores: dict[str, float] = {}
        for label, (low, high) in self._ACTION_MOTION_THRESHOLDS.items():
            midpoint = (low + (high if high != float("inf") else low + 50)) / 2.0
            sigma = max((high - low) / 4.0 if high != float("inf") else 20.0, 1.0)
            raw_scores[label] = float(np.exp(-0.5 * ((mean_mag - midpoint) / sigma) ** 2))

        total = sum(raw_scores.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in raw_scores.items()}
        return {"static_idle": 1.0}

    async def detect_objects(
        self,
        frames: list[np.ndarray],
        sample_every_n_frames: int = 5,
    ) -> list[list[BoundingBox]]:
        """Detect objects in video frames using lightweight colour/edge heuristics.

        Applies a simplified blob-detection approach using colour segmentation
        and contour analysis when deep learning detectors are not available.
        Detections are propagated to unseen frames via simple spatial interpolation.

        Args:
            frames: RGB uint8 frame sequence.
            sample_every_n_frames: Frequency of full detection runs.
                Intermediate frames use the previous detection result.

        Returns:
            Per-frame list of BoundingBox detections (length == len(frames)).
        """
        if not frames:
            return []

        loop = asyncio.get_running_loop()
        detections = await loop.run_in_executor(
            None,
            self._detect_objects_cpu,
            frames,
            sample_every_n_frames,
        )
        logger.debug(
            "Object detection complete",
            num_frames=len(detections),
            sample_rate=sample_every_n_frames,
        )
        return detections

    def _detect_objects_cpu(
        self,
        frames: list[np.ndarray],
        sample_every_n_frames: int,
    ) -> list[list[BoundingBox]]:
        """CPU-bound object detection via contour analysis.

        Args:
            frames: Input frames.
            sample_every_n_frames: Detection interval.

        Returns:
            Per-frame detection lists.
        """
        all_detections: list[list[BoundingBox]] = [[] for _ in frames]
        last_detections: list[BoundingBox] = []

        for frame_idx, frame in enumerate(frames):
            if frame_idx % sample_every_n_frames == 0:
                last_detections = self._detect_blobs(frame, frame_idx)
            all_detections[frame_idx] = last_detections

        return all_detections

    def _detect_blobs(self, frame: np.ndarray, frame_idx: int) -> list[BoundingBox]:
        """Detect salient blobs in a single frame via contour analysis.

        Args:
            frame: RGB uint8 frame.
            frame_idx: Frame index (unused, reserved for tracking ID).

        Returns:
            List of BoundingBox detections.
        """
        if not _OPENCV_AVAILABLE:
            return self._detect_blobs_numpy(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[BoundingBox] = []
        height, width = frame.shape[:2]
        min_area = (width * height) * 0.005  # Minimum 0.5% of frame area

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(1.0, float(area) / (width * height * 0.1))
            detections.append(BoundingBox(
                x1=x, y1=y, x2=x + w, y2=y + h,
                label="object",
                confidence=round(confidence, 3),
            ))

        return detections[:10]  # Cap at 10 detections per frame

    def _detect_blobs_numpy(self, frame: np.ndarray) -> list[BoundingBox]:
        """Numpy-only blob detection via local variance.

        Args:
            frame: RGB uint8 frame.

        Returns:
            Simplified detection list.
        """
        gray = frame.mean(axis=2).astype(np.float32)
        height, width = gray.shape
        block_size = max(1, min(height, width) // 8)
        detections: list[BoundingBox] = []

        for row in range(0, height - block_size, block_size):
            for col in range(0, width - block_size, block_size):
                block = gray[row:row + block_size, col:col + block_size]
                variance = float(np.var(block))
                if variance > 500.0:
                    confidence = min(1.0, variance / 5000.0)
                    detections.append(BoundingBox(
                        x1=col, y1=row,
                        x2=col + block_size, y2=row + block_size,
                        label="object",
                        confidence=round(confidence, 3),
                    ))

        return detections[:10]

    async def classify_scenes(
        self,
        frames: list[np.ndarray],
    ) -> dict[str, float]:
        """Classify the dominant scene type across all frames.

        Uses dominant colour analysis to match against known scene colour
        signatures. Returns a probability distribution over scene classes.

        Args:
            frames: RGB uint8 frame sequence.

        Returns:
            Dict mapping scene class label to confidence score.
        """
        if not frames:
            return {"unknown": 1.0}

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._classify_scene_cpu, frames)
        logger.debug("Scene classification complete", top_scene=max(scores, key=scores.get))  # type: ignore[arg-type]
        return scores

    def _classify_scene_cpu(self, frames: list[np.ndarray]) -> dict[str, float]:
        """CPU-bound scene classification via colour matching.

        Args:
            frames: Input frames.

        Returns:
            Scene score dict.
        """
        # Use middle frame and uniform sample for efficiency
        sample_indices = list(range(0, len(frames), max(1, len(frames) // 5)))[:5]
        sample_frames = [frames[i] for i in sample_indices]

        mean_colour = np.mean([frame.mean(axis=(0, 1)) for frame in sample_frames], axis=0)

        raw_scores: dict[str, float] = {}
        for scene_label, (low_rgb, high_rgb) in self._SCENE_COLOUR_SIGNATURES.items():
            low = np.array(low_rgb, dtype=np.float32)
            high = np.array(high_rgb, dtype=np.float32)
            mid = (low + high) / 2.0
            sigma = (high - low) / 3.0 + 1e-6
            score = float(np.prod(np.exp(-0.5 * ((mean_colour - mid) / sigma) ** 2)))
            raw_scores[scene_label] = score

        total = sum(raw_scores.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in raw_scores.items()}
        return {"unknown": 1.0}

    async def detect_temporal_events(
        self,
        frames: list[np.ndarray],
        motion_peak_threshold: float = 25.0,
        transition_threshold: float = 40.0,
    ) -> list[TemporalEvent]:
        """Detect temporal events including scene transitions and motion peaks.

        Args:
            frames: RGB uint8 frame sequence.
            motion_peak_threshold: Mean optical flow magnitude above which a
                frame window is classified as a motion peak event.
            transition_threshold: Mean pixel difference above which a frame
                boundary is classified as a scene transition.

        Returns:
            List of detected TemporalEvent objects.
        """
        if len(frames) < 2:
            return []

        loop = asyncio.get_running_loop()
        events = await loop.run_in_executor(
            None,
            self._detect_events_cpu,
            frames,
            motion_peak_threshold,
            transition_threshold,
        )
        logger.debug("Temporal event detection complete", num_events=len(events))
        return events

    def _detect_events_cpu(
        self,
        frames: list[np.ndarray],
        motion_peak_threshold: float,
        transition_threshold: float,
    ) -> list[TemporalEvent]:
        """CPU-bound temporal event detection.

        Args:
            frames: Input frames.
            motion_peak_threshold: Motion magnitude threshold.
            transition_threshold: Pixel difference threshold for transitions.

        Returns:
            Detected events.
        """
        events: list[TemporalEvent] = []
        magnitudes: list[float] = []
        diffs: list[float] = []

        for i in range(len(frames) - 1):
            diff = float(np.mean(np.abs(
                frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)
            )))
            diffs.append(diff)

            if _OPENCV_AVAILABLE:
                prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, next_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            else:
                mag = diff
            magnitudes.append(mag)

        # Scene transitions — single-frame events
        for i, diff in enumerate(diffs):
            if diff > transition_threshold:
                confidence = min(1.0, diff / (transition_threshold * 2.0))
                events.append(TemporalEvent(
                    event_type="scene_transition",
                    start_frame=i,
                    end_frame=i + 1,
                    confidence=round(confidence, 3),
                    attributes={"pixel_difference": round(diff, 2)},
                ))

        # Motion peaks — windowed events
        window = self._motion_window
        for start in range(0, len(magnitudes) - window + 1, window // 2):
            window_mags = magnitudes[start:start + window]
            window_mean = float(np.mean(window_mags))
            if window_mean > motion_peak_threshold:
                confidence = min(1.0, window_mean / (motion_peak_threshold * 2.0))
                events.append(TemporalEvent(
                    event_type="motion_peak",
                    start_frame=start,
                    end_frame=min(start + window - 1, len(frames) - 1),
                    confidence=round(confidence, 3),
                    attributes={"mean_flow_magnitude": round(window_mean, 3)},
                ))

        return events

    async def detect_faces(
        self,
        frames: list[np.ndarray],
        sample_every_n_frames: int = 5,
    ) -> list[list[BoundingBox]]:
        """Detect faces in video frames using Haar cascades for privacy processing.

        Results are used by downstream privacy enforcement to determine which
        frames require face blurring. Detection is run on a sampled subset and
        propagated to intermediate frames.

        Args:
            frames: RGB uint8 frame sequence.
            sample_every_n_frames: Detection run frequency.

        Returns:
            Per-frame list of face BoundingBox detections.
        """
        if not frames:
            return []

        loop = asyncio.get_running_loop()
        detections = await loop.run_in_executor(
            None,
            self._detect_faces_cpu,
            frames,
            sample_every_n_frames,
        )
        total_faces = sum(len(d) for d in detections)
        logger.debug(
            "Face detection complete",
            num_frames=len(detections),
            total_face_detections=total_faces,
        )
        return detections

    def _detect_faces_cpu(
        self,
        frames: list[np.ndarray],
        sample_every_n_frames: int,
    ) -> list[list[BoundingBox]]:
        """CPU-bound face detection using Haar cascades.

        Args:
            frames: Input frames.
            sample_every_n_frames: Detection interval.

        Returns:
            Per-frame face detections.
        """
        all_detections: list[list[BoundingBox]] = [[] for _ in frames]
        last_detections: list[BoundingBox] = []

        for frame_idx, frame in enumerate(frames):
            if frame_idx % sample_every_n_frames == 0:
                last_detections = self._haar_face_detect(frame)
            all_detections[frame_idx] = last_detections

        return all_detections

    def _haar_face_detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """Run Haar cascade face detection on a single frame.

        Args:
            frame: RGB uint8 frame.

        Returns:
            List of face bounding boxes.
        """
        if not _OPENCV_AVAILABLE or self._face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=self._face_scale_factor,
            minNeighbors=self._face_min_neighbours,
            minSize=(30, 30),
        )

        detections: list[BoundingBox] = []
        if len(faces) > 0:
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                detections.append(BoundingBox(
                    x1=x, y1=y, x2=x + w, y2=y + h,
                    label="face",
                    confidence=0.9,
                ))
        return detections

    async def analyse_motion(
        self,
        frames: list[np.ndarray],
    ) -> dict[str, float]:
        """Compute aggregate motion statistics for the video sequence.

        Returns statistics suitable for activity summarisation and metadata
        embedding: mean magnitude, peak magnitude, standard deviation,
        percentage of high-motion frames, and directional bias.

        Args:
            frames: RGB uint8 frame sequence.

        Returns:
            Dict with keys: mean_magnitude, peak_magnitude, std_magnitude,
            high_motion_frame_ratio, horizontal_bias, vertical_bias.
        """
        if len(frames) < 2:
            return {
                "mean_magnitude": 0.0,
                "peak_magnitude": 0.0,
                "std_magnitude": 0.0,
                "high_motion_frame_ratio": 0.0,
                "horizontal_bias": 0.5,
                "vertical_bias": 0.5,
            }

        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(None, self._analyse_motion_cpu, frames)
        logger.debug("Motion analysis complete", **summary)
        return summary

    def _analyse_motion_cpu(self, frames: list[np.ndarray]) -> dict[str, float]:
        """CPU-bound motion statistics computation.

        Args:
            frames: Input frames.

        Returns:
            Motion statistics dict.
        """
        magnitudes: list[float] = []
        h_biases: list[float] = []
        v_biases: list[float] = []
        high_motion_threshold = 15.0

        for i in range(len(frames) - 1):
            if _OPENCV_AVAILABLE:
                prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, next_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
                h_energy = float(np.mean(np.abs(flow[..., 0])))
                v_energy = float(np.mean(np.abs(flow[..., 1])))
                total_energy = h_energy + v_energy + 1e-8
                h_biases.append(h_energy / total_energy)
                v_biases.append(v_energy / total_energy)
            else:
                diff = frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)
                mag = float(np.mean(np.abs(diff)))
                h_biases.append(0.5)
                v_biases.append(0.5)

            magnitudes.append(mag)

        mag_array = np.array(magnitudes)
        high_motion_count = int(np.sum(mag_array > high_motion_threshold))

        return {
            "mean_magnitude": round(float(np.mean(mag_array)), 4),
            "peak_magnitude": round(float(np.max(mag_array)), 4),
            "std_magnitude": round(float(np.std(mag_array)), 4),
            "high_motion_frame_ratio": round(high_motion_count / len(magnitudes), 4),
            "horizontal_bias": round(float(np.mean(h_biases)), 4),
            "vertical_bias": round(float(np.mean(v_biases)), 4),
        }

    async def extract_metadata(
        self,
        frames: list[np.ndarray],
        fps: int = 24,
        run_face_detection: bool = True,
        run_object_detection: bool = True,
        object_sample_rate: int = 5,
    ) -> VideoMetadata:
        """Run full metadata extraction pipeline and return structured VideoMetadata.

        Executes all analysis passes (action, object, scene, temporal events,
        faces, motion) and assembles the results into a VideoMetadata dataclass.

        Args:
            frames: RGB uint8 video frame sequence.
            fps: Source video frame rate (used to annotate metadata).
            run_face_detection: Whether to run face detection.
            run_object_detection: Whether to run object detection.
            object_sample_rate: Frame sampling rate for object detection.

        Returns:
            Fully populated VideoMetadata instance.
        """
        if not frames:
            height, width = 0, 0
            return VideoMetadata(
                num_frames=0,
                resolution=(width, height),
                fps_estimated=float(fps),
                dominant_action="unknown",
                action_scores={},
                objects_per_frame=[],
                scene_class="unknown",
                scene_scores={},
                temporal_events=[],
                face_detections=[],
                motion_summary={},
                privacy_flags={"faces_detected": False, "high_motion": False},
            )

        height, width = frames[0].shape[:2]

        logger.info(
            "Starting full metadata extraction",
            num_frames=len(frames),
            resolution=(width, height),
            run_face_detection=run_face_detection,
            run_object_detection=run_object_detection,
        )

        action_task = asyncio.create_task(self.recognise_actions(frames))
        scene_task = asyncio.create_task(self.classify_scenes(frames))
        event_task = asyncio.create_task(self.detect_temporal_events(frames))
        motion_task = asyncio.create_task(self.analyse_motion(frames))

        if run_object_detection:
            object_task = asyncio.create_task(
                self.detect_objects(frames, sample_every_n_frames=object_sample_rate)
            )
        else:
            object_task = None

        if run_face_detection:
            face_task = asyncio.create_task(
                self.detect_faces(frames, sample_every_n_frames=object_sample_rate)
            )
        else:
            face_task = None

        action_scores = await action_task
        scene_scores = await scene_task
        temporal_events = await event_task
        motion_summary = await motion_task

        objects_per_frame = await object_task if object_task else [[] for _ in frames]
        face_detections = await face_task if face_task else [[] for _ in frames]

        dominant_action = max(action_scores, key=action_scores.get) if action_scores else "unknown"  # type: ignore[arg-type]
        scene_class = max(scene_scores, key=scene_scores.get) if scene_scores else "unknown"  # type: ignore[arg-type]

        total_faces = sum(len(fd) for fd in face_detections)
        high_motion = motion_summary.get("high_motion_frame_ratio", 0.0) > 0.3

        logger.info(
            "Metadata extraction complete",
            dominant_action=dominant_action,
            scene_class=scene_class,
            temporal_events=len(temporal_events),
            total_faces=total_faces,
        )

        return VideoMetadata(
            num_frames=len(frames),
            resolution=(width, height),
            fps_estimated=float(fps),
            dominant_action=dominant_action,
            action_scores=action_scores,
            objects_per_frame=objects_per_frame,
            scene_class=scene_class,
            scene_scores=scene_scores,
            temporal_events=temporal_events,
            face_detections=face_detections,
            motion_summary=motion_summary,
            privacy_flags={
                "faces_detected": total_faces > 0,
                "high_motion": high_motion,
            },
        )

    def export_metadata_json(self, metadata: VideoMetadata) -> dict[str, Any]:
        """Serialise VideoMetadata to a JSON-serialisable dict.

        Converts all dataclass instances to plain dicts/lists suitable for
        storage as JSONB, Kafka event payloads, or HTTP response bodies.

        Args:
            metadata: Fully populated VideoMetadata instance.

        Returns:
            JSON-serialisable dict representation of the metadata.
        """
        def bbox_to_dict(bbox: BoundingBox) -> dict[str, Any]:
            return {
                "x1": bbox.x1,
                "y1": bbox.y1,
                "x2": bbox.x2,
                "y2": bbox.y2,
                "label": bbox.label,
                "confidence": bbox.confidence,
                "track_id": bbox.track_id,
            }

        def event_to_dict(event: TemporalEvent) -> dict[str, Any]:
            return {
                "event_type": event.event_type,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "confidence": event.confidence,
                "attributes": event.attributes,
            }

        return {
            "num_frames": metadata.num_frames,
            "resolution": {"width": metadata.resolution[0], "height": metadata.resolution[1]},
            "fps_estimated": metadata.fps_estimated,
            "action": {
                "dominant": metadata.dominant_action,
                "scores": metadata.action_scores,
            },
            "scene": {
                "class": metadata.scene_class,
                "scores": metadata.scene_scores,
            },
            "objects_per_frame": [
                [bbox_to_dict(b) for b in frame_bboxes]
                for frame_bboxes in metadata.objects_per_frame
            ],
            "temporal_events": [event_to_dict(e) for e in metadata.temporal_events],
            "face_detections": [
                [bbox_to_dict(b) for b in frame_faces]
                for frame_faces in metadata.face_detections
            ],
            "motion_summary": metadata.motion_summary,
            "privacy_flags": metadata.privacy_flags,
        }
