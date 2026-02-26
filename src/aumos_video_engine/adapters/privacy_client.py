"""HTTP client for the aumos-privacy-engine service.

Implements PrivacyEnforcerProtocol by delegating per-frame privacy enforcement
to the remote aumos-privacy-engine microservice, with automatic fallback to the
local OpenCV enforcer when the remote service is unreachable.
"""

from __future__ import annotations

import base64

import httpx
import numpy as np
from aumos_common.errors import ExternalServiceError
from aumos_common.observability import get_logger
from PIL import Image
import io

from aumos_video_engine.adapters.privacy_enforcer import LocalPrivacyEnforcer

logger = get_logger(__name__)


class PrivacyEngineClient:
    """HTTP client for the aumos-privacy-engine API.

    Primary adapter for per-frame PII detection and redaction.
    Delegates to LocalPrivacyEnforcer on connection failure to maintain
    service availability even when privacy-engine is down.
    """

    def __init__(
        self,
        base_url: str,
        fallback: LocalPrivacyEnforcer,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize PrivacyEngineClient.

        Args:
            base_url: Base URL of the aumos-privacy-engine service.
            fallback: Local enforcer to use when remote service is unavailable.
            timeout_seconds: HTTP request timeout.
            max_retries: Number of retry attempts on transient failures.
        """
        self._base_url = base_url.rstrip("/")
        self._fallback = fallback
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode a numpy frame as a base64 PNG string.

        Args:
            frame: RGB uint8 numpy array (H, W, 3).

        Returns:
            Base64-encoded PNG image string.
        """
        pil_image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _base64_to_frame(self, encoded: str, original_shape: tuple[int, ...]) -> np.ndarray:
        """Decode a base64 PNG string back to a numpy frame.

        Args:
            encoded: Base64-encoded PNG string.
            original_shape: Expected output shape (H, W, 3).

        Returns:
            Decoded RGB uint8 numpy array.
        """
        image_bytes = base64.b64decode(encoded)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_image, dtype=np.uint8)

    async def enforce_frame(
        self,
        frame: np.ndarray,
        blur_faces: bool,
        redact_plates: bool,
        remove_pii: bool,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Apply privacy enforcement to a single frame via privacy-engine API.

        Falls back to local enforcement on connection errors.

        Args:
            frame: RGB uint8 numpy array (H, W, 3).
            blur_faces: Whether to blur faces.
            redact_plates: Whether to redact plates.
            remove_pii: Whether to remove other PII.

        Returns:
            Tuple of (processed_frame, detection_counts).
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                payload = {
                    "image_b64": self._frame_to_base64(frame),
                    "blur_faces": blur_faces,
                    "redact_plates": redact_plates,
                    "remove_pii": remove_pii,
                }
                response = await client.post(
                    f"{self._base_url}/api/v1/privacy/enforce-frame",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                processed_frame = self._base64_to_frame(data["image_b64"], frame.shape)
                detection_counts: dict[str, int] = data.get("detection_counts", {})
                return processed_frame, detection_counts

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                "Privacy engine unreachable — using local fallback",
                error=str(exc),
                base_url=self._base_url,
            )
            return await self._fallback.enforce_frame(
                frame=frame,
                blur_faces=blur_faces,
                redact_plates=redact_plates,
                remove_pii=remove_pii,
            )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Privacy engine returned error",
                status_code=exc.response.status_code,
                error=str(exc),
            )
            raise ExternalServiceError(
                f"Privacy engine returned {exc.response.status_code}: {exc.response.text}"
            ) from exc

    async def enforce_batch(
        self,
        frames: list[np.ndarray],
        blur_faces: bool,
        redact_plates: bool,
        remove_pii: bool,
    ) -> tuple[list[np.ndarray], dict[str, int]]:
        """Apply privacy enforcement to a batch of frames via privacy-engine API.

        Uses batch endpoint when available; falls back to per-frame calls.

        Args:
            frames: List of RGB uint8 numpy arrays (H, W, 3).
            blur_faces: Whether to blur faces.
            redact_plates: Whether to redact plates.
            remove_pii: Whether to remove other PII.

        Returns:
            Tuple of (processed_frames, aggregate_detection_counts).
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout * 2) as client:
                payload = {
                    "images_b64": [self._frame_to_base64(f) for f in frames],
                    "blur_faces": blur_faces,
                    "redact_plates": redact_plates,
                    "remove_pii": remove_pii,
                }
                response = await client.post(
                    f"{self._base_url}/api/v1/privacy/enforce-batch",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                processed_frames = [
                    self._base64_to_frame(enc, frames[i].shape)
                    for i, enc in enumerate(data["images_b64"])
                ]
                total_counts: dict[str, int] = data.get("aggregate_detection_counts", {})
                return processed_frames, total_counts

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                "Privacy engine batch endpoint unreachable — using local fallback",
                error=str(exc),
            )
            return await self._fallback.enforce_batch(
                frames=frames,
                blur_faces=blur_faces,
                redact_plates=redact_plates,
                remove_pii=remove_pii,
            )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Privacy engine batch returned error",
                status_code=exc.response.status_code,
            )
            raise ExternalServiceError(
                f"Privacy engine batch returned {exc.response.status_code}"
            ) from exc
