"""Local per-frame privacy enforcer using OpenCV.

Provides a local fallback implementation of PrivacyEnforcerProtocol
when the remote privacy-engine service is unavailable. Uses OpenCV
Haar cascades for face detection and plate detection.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_OPENCV_AVAILABLE = False
try:
    import cv2  # type: ignore[import-untyped]

    _OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python-headless not installed — local privacy enforcer unavailable")


class LocalPrivacyEnforcer:
    """Local per-frame privacy enforcer using OpenCV Haar cascades.

    This is a fallback implementation for when the privacy-engine service
    is unreachable. For production workloads, prefer the PrivacyEngineClient
    which uses a more accurate model-based detection pipeline.
    """

    def __init__(
        self,
        blur_kernel_size: tuple[int, int] = (51, 51),
        face_scale_factor: float = 1.1,
        face_min_neighbors: int = 5,
    ) -> None:
        """Initialize LocalPrivacyEnforcer.

        Args:
            blur_kernel_size: Gaussian blur kernel size for face/plate redaction.
            face_scale_factor: Scale factor for Haar cascade face detection.
            face_min_neighbors: Minimum neighbor rectangles for face detection.
        """
        self._blur_kernel_size = blur_kernel_size
        self._face_scale_factor = face_scale_factor
        self._face_min_neighbors = face_min_neighbors
        self._face_cascade: Any | None = None
        self._plate_cascade: Any | None = None

    def _load_cascades(self) -> None:
        """Lazy-load Haar cascade classifiers."""
        if not _OPENCV_AVAILABLE:
            return
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
            if os.path.exists(cascade_path):
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                logger.warning("Face cascade XML not found", path=cascade_path)

        if self._plate_cascade is None:
            plate_cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"  # type: ignore[attr-defined]
            if os.path.exists(plate_cascade_path):
                self._plate_cascade = cv2.CascadeClassifier(plate_cascade_path)

    def _blur_regions(
        self,
        frame: np.ndarray,
        regions: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Apply Gaussian blur to a list of bounding box regions.

        Args:
            frame: RGB uint8 frame.
            regions: List of (x, y, w, h) bounding boxes to blur.

        Returns:
            Frame with specified regions blurred.
        """
        result = frame.copy()
        for x, y, w, h in regions:
            roi = result[y : y + h, x : x + w]
            blurred_roi = cv2.GaussianBlur(roi, self._blur_kernel_size, 0)  # type: ignore[call-overload]
            result[y : y + h, x : x + w] = blurred_roi
        return result

    def _detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect face bounding boxes in a frame.

        Args:
            frame: RGB uint8 numpy array.

        Returns:
            List of (x, y, w, h) bounding boxes for detected faces.
        """
        if not _OPENCV_AVAILABLE or self._face_cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=self._face_scale_factor,
            minNeighbors=self._face_min_neighbors,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _detect_plates(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect license plate bounding boxes in a frame.

        Args:
            frame: RGB uint8 numpy array.

        Returns:
            List of (x, y, w, h) bounding boxes for detected plates.
        """
        if not _OPENCV_AVAILABLE or self._plate_cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        plates = self._plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 15),
        )
        if len(plates) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in plates]

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
            remove_pii: Whether to remove other PII (not implemented locally — requires
                remote privacy-engine for text/badge detection).

        Returns:
            Tuple of (processed_frame, detection_counts).
        """
        self._load_cascades()
        result = frame.copy()
        detection_counts: dict[str, int] = {}

        if blur_faces:
            faces = self._detect_faces(result)
            if faces:
                result = self._blur_regions(result, faces)
            detection_counts["faces"] = len(faces)

        if redact_plates:
            plates = self._detect_plates(result)
            if plates:
                result = self._blur_regions(result, plates)
            detection_counts["plates"] = len(plates)

        if remove_pii:
            # Local PII removal not implemented — requires remote service
            logger.debug("PII removal requested but not available in local enforcer")
            detection_counts["pii"] = 0

        return result, detection_counts

    async def enforce_batch(
        self,
        frames: list[np.ndarray],
        blur_faces: bool,
        redact_plates: bool,
        remove_pii: bool,
    ) -> tuple[list[np.ndarray], dict[str, int]]:
        """Apply privacy enforcement to a batch of frames.

        Args:
            frames: List of RGB uint8 numpy arrays (H, W, 3).
            blur_faces: Whether to blur faces.
            redact_plates: Whether to redact plates.
            remove_pii: Whether to remove other PII.

        Returns:
            Tuple of (processed_frames, aggregate_detection_counts).
        """
        processed_frames: list[np.ndarray] = []
        total_counts: dict[str, int] = {}

        for frame in frames:
            processed_frame, counts = await self.enforce_frame(
                frame=frame,
                blur_faces=blur_faces,
                redact_plates=redact_plates,
                remove_pii=remove_pii,
            )
            processed_frames.append(processed_frame)
            for entity_type, count in counts.items():
                total_counts[entity_type] = total_counts.get(entity_type, 0) + count

        return processed_frames, total_counts
