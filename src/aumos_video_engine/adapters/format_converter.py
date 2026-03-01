"""Multi-format video conversion adapter.

Provides bidirectional conversion between video formats (MP4, WebM, AVI, GIF)
with configurable codec selection, bitrate control, and frame rate adjustment.
"""

from __future__ import annotations

import io
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class VideoFormat(str, Enum):
    """Supported video container formats."""

    MP4 = "mp4"
    WEBM = "webm"
    AVI = "avi"
    GIF = "gif"
    MOV = "mov"


class VideoCodecProfile(str, Enum):
    """Predefined codec/quality profiles."""

    WEB_OPTIMIZED = "web_optimized"
    HIGH_QUALITY = "high_quality"
    ARCHIVAL = "archival"
    PREVIEW = "preview"


PROFILE_SETTINGS: dict[VideoCodecProfile, dict[str, Any]] = {
    VideoCodecProfile.WEB_OPTIMIZED: {
        "codec": "libx264",
        "crf": 23,
        "preset": "medium",
        "pix_fmt": "yuv420p",
        "max_bitrate": "5M",
    },
    VideoCodecProfile.HIGH_QUALITY: {
        "codec": "libx264",
        "crf": 18,
        "preset": "slow",
        "pix_fmt": "yuv420p",
        "max_bitrate": "15M",
    },
    VideoCodecProfile.ARCHIVAL: {
        "codec": "libx265",
        "crf": 15,
        "preset": "veryslow",
        "pix_fmt": "yuv420p10le",
        "max_bitrate": None,
    },
    VideoCodecProfile.PREVIEW: {
        "codec": "libx264",
        "crf": 28,
        "preset": "ultrafast",
        "pix_fmt": "yuv420p",
        "max_bitrate": "2M",
    },
}


class FormatConverter:
    """Converts video frame sequences between formats and codec profiles.

    Wraps PyAV for encoding with configurable quality profiles, or falls
    back to OpenCV VideoWriter when PyAV is unavailable.

    Args:
        default_profile: Default codec profile for encoding operations.
    """

    def __init__(
        self,
        default_profile: VideoCodecProfile = VideoCodecProfile.WEB_OPTIMIZED,
    ) -> None:
        self._default_profile = default_profile
        self._av_available = self._check_av_available()

    @staticmethod
    def _check_av_available() -> bool:
        """Check if PyAV is available for video encoding."""
        try:
            import av  # noqa: F401
            return True
        except ImportError:
            return False

    async def convert_frames(
        self,
        frames: list[np.ndarray],
        target_format: VideoFormat,
        fps: int = 24,
        profile: VideoCodecProfile | None = None,
        output_resolution: tuple[int, int] | None = None,
    ) -> bytes:
        """Convert a frame sequence to the specified video format.

        Args:
            frames: RGB uint8 numpy arrays (H, W, 3).
            target_format: Target video container format.
            fps: Output frames per second.
            profile: Codec quality profile (uses default if None).
            output_resolution: Optional target resolution (width, height).

        Returns:
            Encoded video bytes in the target format.
        """
        if not frames:
            raise ValueError("Cannot convert empty frame sequence")

        active_profile = profile or self._default_profile
        settings = PROFILE_SETTINGS[active_profile]

        logger.info(
            "converting_video",
            frame_count=len(frames),
            target_format=target_format.value,
            profile=active_profile.value,
            fps=fps,
        )

        if target_format == VideoFormat.GIF:
            return await self._encode_gif(frames, fps, output_resolution)

        if self._av_available:
            return await self._encode_pyav(
                frames, target_format, fps, settings, output_resolution,
            )

        return await self._encode_opencv(
            frames, target_format, fps, output_resolution,
        )

    async def _encode_pyav(
        self,
        frames: list[np.ndarray],
        target_format: VideoFormat,
        fps: int,
        settings: dict[str, Any],
        output_resolution: tuple[int, int] | None,
    ) -> bytes:
        """Encode frames using PyAV (libav wrapper).

        Args:
            frames: RGB uint8 frames.
            target_format: Container format.
            fps: Target frame rate.
            settings: Codec configuration.
            output_resolution: Optional resize target.

        Returns:
            Encoded video bytes.
        """
        import av

        buf = io.BytesIO()
        container = av.open(buf, mode="w", format=target_format.value)

        codec_name = settings["codec"]
        if target_format == VideoFormat.WEBM:
            codec_name = "libvpx-vp9"

        h, w = frames[0].shape[:2]
        if output_resolution:
            w, h = output_resolution

        stream = container.add_stream(codec_name, rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = settings.get("pix_fmt", "yuv420p")

        if "crf" in settings:
            stream.options = {"crf": str(settings["crf"])}
        if settings.get("preset"):
            stream.options["preset"] = settings["preset"]

        for frame_data in frames:
            if output_resolution and (frame_data.shape[1], frame_data.shape[0]) != output_resolution:
                import cv2
                frame_data = cv2.resize(frame_data, output_resolution)

            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()
        return buf.getvalue()

    async def _encode_opencv(
        self,
        frames: list[np.ndarray],
        target_format: VideoFormat,
        fps: int,
        output_resolution: tuple[int, int] | None,
    ) -> bytes:
        """Fallback encoding using OpenCV VideoWriter.

        Args:
            frames: RGB uint8 frames.
            target_format: Container format.
            fps: Target frame rate.
            output_resolution: Optional resize target.

        Returns:
            Encoded video bytes.
        """
        import cv2

        h, w = frames[0].shape[:2]
        if output_resolution:
            w, h = output_resolution

        fourcc_map = {
            VideoFormat.MP4: "mp4v",
            VideoFormat.AVI: "XVID",
            VideoFormat.MOV: "mp4v",
            VideoFormat.WEBM: "VP90",
        }
        fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(target_format, "mp4v"))

        with tempfile.NamedTemporaryFile(
            suffix=f".{target_format.value}", delete=True,
        ) as tmp:
            writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
            try:
                for frame_data in frames:
                    bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                    if output_resolution:
                        bgr = cv2.resize(bgr, output_resolution)
                    writer.write(bgr)
            finally:
                writer.release()

            return Path(tmp.name).read_bytes()

    async def _encode_gif(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None,
    ) -> bytes:
        """Encode frames as animated GIF using PIL.

        Args:
            frames: RGB uint8 frames.
            fps: Target frame rate (converted to inter-frame delay).
            output_resolution: Optional resize target.

        Returns:
            GIF bytes.
        """
        from PIL import Image

        images: list[Image.Image] = []
        for frame_data in frames:
            img = Image.fromarray(frame_data)
            if output_resolution:
                img = img.resize(output_resolution, Image.LANCZOS)
            images.append(img)

        buf = io.BytesIO()
        duration_ms = int(1000 / fps)
        images[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        return buf.getvalue()

    async def extract_frames(
        self,
        video_bytes: bytes,
        container_format: VideoFormat = VideoFormat.MP4,
        max_frames: int | None = None,
    ) -> tuple[list[np.ndarray], int]:
        """Extract frames from an encoded video.

        Args:
            video_bytes: Encoded video bytes.
            container_format: Source container format hint.
            max_frames: Maximum number of frames to extract.

        Returns:
            Tuple of (frames list, detected fps).
        """
        if not self._av_available:
            raise RuntimeError("PyAV required for frame extraction")

        import av

        buf = io.BytesIO(video_bytes)
        container = av.open(buf, format=container_format.value)
        stream = container.streams.video[0]
        detected_fps = int(stream.average_rate) if stream.average_rate else 24

        extracted: list[np.ndarray] = []
        for frame in container.decode(stream):
            extracted.append(frame.to_ndarray(format="rgb24"))
            if max_frames and len(extracted) >= max_frames:
                break

        container.close()
        return extracted, detected_fps

    async def get_video_info(
        self,
        video_bytes: bytes,
    ) -> dict[str, Any]:
        """Extract metadata from encoded video bytes.

        Args:
            video_bytes: Encoded video bytes.

        Returns:
            Dict with keys: width, height, fps, duration_seconds, frame_count, codec.
        """
        if not self._av_available:
            return {"error": "PyAV required for metadata extraction"}

        import av

        buf = io.BytesIO(video_bytes)
        container = av.open(buf)
        stream = container.streams.video[0]

        info: dict[str, Any] = {
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate) if stream.average_rate else 0,
            "duration_seconds": float(stream.duration * stream.time_base) if stream.duration else 0,
            "frame_count": stream.frames,
            "codec": stream.codec_context.name if stream.codec_context else "unknown",
        }
        container.close()
        return info
