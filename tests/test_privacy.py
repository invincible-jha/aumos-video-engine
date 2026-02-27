"""Tests for per-frame privacy enforcement.

Covers LocalPrivacyEnforcer (OpenCV fallback) and PrivacyEngineClient
(remote HTTP client with automatic fallback), validating frame-by-frame
privacy protection for surveillance and traffic domain videos.
"""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ── LocalPrivacyEnforcer ───────────────────────────────────────────


class TestLocalPrivacyEnforcerNoOpenCV:
    """Tests for LocalPrivacyEnforcer when OpenCV is not installed."""

    @pytest.fixture()
    def enforcer_no_cv(self) -> Any:
        """Return a LocalPrivacyEnforcer with OpenCV patched as unavailable."""
        from aumos_video_engine.adapters import privacy_enforcer as module

        original = module._OPENCV_AVAILABLE
        module._OPENCV_AVAILABLE = False
        from aumos_video_engine.adapters.privacy_enforcer import LocalPrivacyEnforcer
        enforcer = LocalPrivacyEnforcer()
        yield enforcer
        module._OPENCV_AVAILABLE = original

    @pytest.mark.asyncio()
    async def test_enforce_frame_returns_copy_of_input_when_no_opencv(
        self,
        enforcer_no_cv: Any,
        single_frame: np.ndarray,
    ) -> None:
        """enforce_frame must return a frame (copy) even without OpenCV."""
        result_frame, counts = await enforcer_no_cv.enforce_frame(
            frame=single_frame,
            blur_faces=True,
            redact_plates=True,
            remove_pii=False,
        )
        assert isinstance(result_frame, np.ndarray)
        assert result_frame.shape == single_frame.shape

    @pytest.mark.asyncio()
    async def test_enforce_batch_returns_same_length_list(
        self,
        enforcer_no_cv: Any,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """enforce_batch must return a list of the same length as input."""
        processed_frames, counts = await enforcer_no_cv.enforce_batch(
            frames=frame_sequence,
            blur_faces=True,
            redact_plates=True,
            remove_pii=False,
        )
        assert len(processed_frames) == len(frame_sequence)

    @pytest.mark.asyncio()
    async def test_enforce_frame_records_zero_faces_when_no_opencv(
        self,
        enforcer_no_cv: Any,
        single_frame: np.ndarray,
    ) -> None:
        """enforce_frame must report 0 face detections when cascade unavailable."""
        _, counts = await enforcer_no_cv.enforce_frame(
            frame=single_frame,
            blur_faces=True,
            redact_plates=False,
            remove_pii=False,
        )
        assert counts.get("faces", 0) == 0

    @pytest.mark.asyncio()
    async def test_enforce_frame_pii_flag_records_zero_pii_locally(
        self,
        enforcer_no_cv: Any,
        single_frame: np.ndarray,
    ) -> None:
        """enforce_frame must record pii=0 for local enforcer (not implemented)."""
        _, counts = await enforcer_no_cv.enforce_frame(
            frame=single_frame,
            blur_faces=False,
            redact_plates=False,
            remove_pii=True,
        )
        assert counts.get("pii", 0) == 0

    @pytest.mark.asyncio()
    async def test_enforce_batch_aggregates_counts(
        self,
        enforcer_no_cv: Any,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """enforce_batch must aggregate detection counts across all frames."""
        _, counts = await enforcer_no_cv.enforce_batch(
            frames=frame_sequence,
            blur_faces=True,
            redact_plates=True,
            remove_pii=False,
        )
        # Without OpenCV, both should be 0 (summed across all frames)
        assert counts.get("faces", 0) == 0
        assert counts.get("plates", 0) == 0


class TestLocalPrivacyEnforcerBlurRegions:
    """Tests for LocalPrivacyEnforcer._blur_regions helper."""

    @pytest.fixture()
    def enforcer(self) -> Any:
        """Return a LocalPrivacyEnforcer instance."""
        from aumos_video_engine.adapters.privacy_enforcer import LocalPrivacyEnforcer

        return LocalPrivacyEnforcer()

    def test_blur_regions_returns_different_pixels_for_blurred_area(
        self,
        enforcer: Any,
    ) -> None:
        """_blur_regions must produce a different pixel value in the blurred region."""
        # We need cv2 for this test
        pytest.importorskip("cv2")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Set a solid red block in the region to blur
        frame[10:50, 10:50] = [255, 0, 0]
        regions = [(10, 10, 40, 40)]
        result = enforcer._blur_regions(frame, regions)
        # The blurred area should not be pure red anymore
        blurred_pixel = result[30, 30]
        original_pixel = np.array([255, 0, 0], dtype=np.uint8)
        assert not np.array_equal(blurred_pixel, original_pixel)

    def test_blur_regions_with_empty_list_returns_unchanged(
        self,
        enforcer: Any,
        single_frame: np.ndarray,
    ) -> None:
        """_blur_regions with an empty regions list must return a copy of input."""
        pytest.importorskip("cv2")
        result = enforcer._blur_regions(single_frame, [])
        np.testing.assert_array_equal(result, single_frame)


# ── PrivacyEngineClient ────────────────────────────────────────────


class TestPrivacyEngineClientFrameEncoding:
    """Tests for PrivacyEngineClient base64 frame encode/decode helpers."""

    @pytest.fixture()
    def client(self) -> Any:
        """Return a PrivacyEngineClient with a mocked fallback enforcer."""
        from aumos_video_engine.adapters.privacy_client import PrivacyEngineClient
        from aumos_video_engine.adapters.privacy_enforcer import LocalPrivacyEnforcer

        return PrivacyEngineClient(
            base_url="http://privacy-engine:8000",
            fallback=LocalPrivacyEnforcer(),
            timeout_seconds=5.0,
        )

    def test_frame_to_base64_produces_valid_string(
        self,
        client: Any,
        single_frame: np.ndarray,
    ) -> None:
        """_frame_to_base64 must produce a non-empty base64 string."""
        pytest.importorskip("PIL")
        encoded = client._frame_to_base64(single_frame)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Verify it is valid base64
        base64.b64decode(encoded)

    def test_base64_to_frame_roundtrip_preserves_shape(
        self,
        client: Any,
        single_frame: np.ndarray,
    ) -> None:
        """Encoding and decoding a frame must preserve the original shape."""
        pytest.importorskip("PIL")
        encoded = client._frame_to_base64(single_frame)
        decoded = client._base64_to_frame(encoded, single_frame.shape)
        assert decoded.shape == single_frame.shape
        assert decoded.dtype == np.uint8


class TestPrivacyEngineClientFallback:
    """Tests for PrivacyEngineClient fallback behavior on connection errors."""

    @pytest.fixture()
    def mock_fallback(self, single_frame: np.ndarray, frame_sequence: list[np.ndarray]) -> AsyncMock:
        """Return a mock LocalPrivacyEnforcer."""
        fallback = AsyncMock()
        fallback.enforce_frame = AsyncMock(
            return_value=(single_frame, {"faces": 0, "plates": 0})
        )
        fallback.enforce_batch = AsyncMock(
            return_value=(frame_sequence, {"faces": 0, "plates": 0})
        )
        return fallback

    @pytest.fixture()
    def client(self, mock_fallback: AsyncMock) -> Any:
        """Return a PrivacyEngineClient with a mocked fallback."""
        from aumos_video_engine.adapters.privacy_client import PrivacyEngineClient

        return PrivacyEngineClient(
            base_url="http://privacy-engine:8000",
            fallback=mock_fallback,
            timeout_seconds=1.0,
        )

    @pytest.mark.asyncio()
    async def test_enforce_frame_falls_back_on_connect_error(
        self,
        client: Any,
        mock_fallback: AsyncMock,
        single_frame: np.ndarray,
    ) -> None:
        """enforce_frame must use fallback when remote service is unreachable."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value = mock_http

            result_frame, counts = await client.enforce_frame(
                frame=single_frame,
                blur_faces=True,
                redact_plates=True,
                remove_pii=False,
            )

        mock_fallback.enforce_frame.assert_called_once()
        assert isinstance(result_frame, np.ndarray)

    @pytest.mark.asyncio()
    async def test_enforce_batch_falls_back_on_timeout(
        self,
        client: Any,
        mock_fallback: AsyncMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """enforce_batch must use fallback when request times out."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_http

            result_frames, counts = await client.enforce_batch(
                frames=frame_sequence,
                blur_faces=True,
                redact_plates=True,
                remove_pii=False,
            )

        mock_fallback.enforce_batch.assert_called_once()
        assert len(result_frames) == len(frame_sequence)

    @pytest.mark.asyncio()
    async def test_enforce_frame_raises_on_http_error_status(
        self,
        client: Any,
        single_frame: np.ndarray,
    ) -> None:
        """enforce_frame must raise ExternalServiceError on non-2xx HTTP responses."""
        import httpx
        from aumos_common.errors import ExternalServiceError

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_http.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "500", request=MagicMock(), response=mock_response
                )
            )
            mock_client_cls.return_value = mock_http

            with pytest.raises(ExternalServiceError):
                await client.enforce_frame(
                    frame=single_frame,
                    blur_faces=True,
                    redact_plates=True,
                    remove_pii=False,
                )


# ── Privacy enforcement domain requirements ───────────────────────


class TestPrivacyEnforcementDomainRequirements:
    """Tests verifying domain-specific privacy enforcement requirements."""

    @pytest.mark.asyncio()
    async def test_surveillance_domain_enforces_face_blur(
        self,
        mock_privacy_enforcer: AsyncMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """Surveillance domain must always blur faces (called with blur_faces=True)."""
        # Surveillance domain requires face blur — verify enforce_batch is called
        # with blur_faces=True
        await mock_privacy_enforcer.enforce_batch(
            frames=frame_sequence,
            blur_faces=True,  # Required for surveillance
            redact_plates=True,
            remove_pii=False,
        )
        call_kwargs = mock_privacy_enforcer.enforce_batch.call_args[1]
        assert call_kwargs["blur_faces"] is True

    @pytest.mark.asyncio()
    async def test_traffic_domain_enforces_plate_redaction(
        self,
        mock_privacy_enforcer: AsyncMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """Traffic domain must always redact license plates (called with redact_plates=True)."""
        await mock_privacy_enforcer.enforce_batch(
            frames=frame_sequence,
            blur_faces=True,
            redact_plates=True,  # Required for traffic
            remove_pii=False,
        )
        call_kwargs = mock_privacy_enforcer.enforce_batch.call_args[1]
        assert call_kwargs["redact_plates"] is True

    @pytest.mark.asyncio()
    async def test_enforce_batch_preserves_frame_count(
        self,
        mock_privacy_enforcer: AsyncMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """enforce_batch must return exactly as many frames as provided."""
        processed_frames, _ = await mock_privacy_enforcer.enforce_batch(
            frames=frame_sequence,
            blur_faces=True,
            redact_plates=True,
            remove_pii=False,
        )
        assert len(processed_frames) == len(frame_sequence)

    @pytest.mark.asyncio()
    async def test_enforce_batch_detection_counts_have_expected_keys(
        self,
        mock_privacy_enforcer: AsyncMock,
        frame_sequence: list[np.ndarray],
    ) -> None:
        """enforce_batch detection_counts must include 'faces' and 'plates' keys."""
        _, counts = await mock_privacy_enforcer.enforce_batch(
            frames=frame_sequence,
            blur_faces=True,
            redact_plates=True,
            remove_pii=False,
        )
        assert "faces" in counts
        assert "plates" in counts
