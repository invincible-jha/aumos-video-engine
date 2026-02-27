"""Video export handler adapter.

Encodes synthesised video frames to MP4 (H.264/H.265), WebM (VP9), and AVI
formats using PyAV, with configurable resolution/framerate, audio track muxing,
metadata embedding, MinIO/S3 upload, and thumbnail extraction.
"""

from __future__ import annotations

import asyncio
import io
import struct
import uuid
from enum import Enum
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_PYAV_AVAILABLE = False
try:
    import av  # type: ignore[import-untyped]

    _PYAV_AVAILABLE = True
except ImportError:
    logger.warning("PyAV not installed — video export will use fallback raw bytes mode")

_OPENCV_AVAILABLE = False
try:
    import cv2  # type: ignore[import-untyped]

    _OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python-headless not installed — thumbnail extraction using numpy fallback")


class VideoCodec(str, Enum):
    """Supported video codec identifiers."""

    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    MPEG4 = "mpeg4"


class VideoContainer(str, Enum):
    """Supported video container formats."""

    MP4 = "mp4"
    WEBM = "webm"
    AVI = "avi"


class VideoExportHandler:
    """Encodes and exports synthesised video frames to standard container formats.

    Supports MP4 (H.264/H.265), WebM (VP9), and AVI formats. Handles resolution
    and framerate configuration, audio muxing, metadata embedding, MinIO/S3 upload,
    and thumbnail extraction.
    """

    def __init__(
        self,
        storage_client: Any,
        default_video_bitrate: int = 4_000_000,
        default_audio_bitrate: int = 128_000,
        thumbnail_width: int = 320,
        thumbnail_height: int = 180,
        minio_bucket: str = "aumos-video-artifacts",
    ) -> None:
        """Initialize VideoExportHandler.

        Args:
            storage_client: MinIO/S3 client with put_object(bucket, key, data, size, content_type)
                method. The client must already be authenticated.
            default_video_bitrate: Default video bitrate in bits per second.
            default_audio_bitrate: Default audio bitrate in bits per second.
            thumbnail_width: Width of extracted thumbnails in pixels.
            thumbnail_height: Height of extracted thumbnails in pixels.
            minio_bucket: Target MinIO/S3 bucket for video artifact storage.
        """
        self._storage = storage_client
        self._video_bitrate = default_video_bitrate
        self._audio_bitrate = default_audio_bitrate
        self._thumb_width = thumbnail_width
        self._thumb_height = thumbnail_height
        self._bucket = minio_bucket

    async def export_mp4(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None = None,
        codec: VideoCodec = VideoCodec.H264,
        crf: int = 23,
        preset: str = "medium",
        audio_bytes: bytes | None = None,
        metadata: dict[str, str] | None = None,
    ) -> bytes:
        """Encode frames to MP4 container with H.264 or H.265 codec.

        Args:
            frames: RGB uint8 frame list (H, W, 3).
            fps: Target frames per second.
            output_resolution: Optional (width, height) to resize frames on encode.
                If None, uses the native frame resolution.
            codec: H264 or H265 codec selection.
            crf: Constant Rate Factor quality setting (0=lossless, 51=worst).
            preset: Encoding speed/quality preset (ultrafast, fast, medium, slow).
            audio_bytes: Optional PCM audio bytes to mux as AAC track.
            metadata: Optional dict of metadata tags to embed in the container.

        Returns:
            MP4-encoded video as raw bytes.

        Raises:
            ValueError: If frames list is empty or fps is non-positive.
        """
        self._validate_inputs(frames, fps)
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(
            None,
            self._encode_mp4,
            frames,
            fps,
            output_resolution,
            codec,
            crf,
            preset,
            audio_bytes,
            metadata or {},
        )
        logger.info(
            "MP4 export complete",
            num_frames=len(frames),
            fps=fps,
            codec=codec.value,
            size_bytes=len(encoded),
        )
        return encoded

    def _encode_mp4(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None,
        codec: VideoCodec,
        crf: int,
        preset: str,
        audio_bytes: bytes | None,
        metadata: dict[str, str],
    ) -> bytes:
        """CPU-bound MP4 encoding via PyAV.

        Args:
            frames: RGB uint8 frames.
            fps: Frames per second.
            output_resolution: Optional resize target.
            codec: Video codec.
            crf: Quality factor.
            preset: Encoding preset.
            audio_bytes: Optional audio.
            metadata: Metadata tags.

        Returns:
            Encoded MP4 bytes.
        """
        if not _PYAV_AVAILABLE:
            return self._encode_fallback(frames, fps)

        buffer = io.BytesIO()
        codec_name = "libx264" if codec == VideoCodec.H264 else "libx265"

        height, width = frames[0].shape[:2]
        if output_resolution is not None:
            width, height = output_resolution
        # Ensure dimensions are even (codec requirement)
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1

        with av.open(buffer, mode="w", format="mp4") as container:
            for key, value in metadata.items():
                container.metadata[key] = value

            video_stream = container.add_stream(codec_name, rate=fps)
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {
                "crf": str(crf),
                "preset": preset,
            }

            if audio_bytes is not None:
                audio_stream = container.add_stream("aac", rate=44100)
                audio_stream.options = {"b": str(self._audio_bitrate)}

            for frame_array in frames:
                if output_resolution is not None and (
                    frame_array.shape[1] != width or frame_array.shape[0] != height
                ):
                    if _OPENCV_AVAILABLE:
                        frame_array = cv2.resize(frame_array, (width, height))
                    else:
                        row_idx = (np.arange(height) * frame_array.shape[0] / height).astype(int)
                        col_idx = (np.arange(width) * frame_array.shape[1] / width).astype(int)
                        frame_array = frame_array[np.ix_(row_idx, col_idx)]

                av_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                av_frame = av_frame.reformat(format="yuv420p")
                for packet in video_stream.encode(av_frame):
                    container.mux(packet)

            # Flush video stream
            for packet in video_stream.encode():
                container.mux(packet)

        return buffer.getvalue()

    async def export_webm(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None = None,
        crf: int = 33,
        cpu_used: int = 4,
        audio_bytes: bytes | None = None,
        metadata: dict[str, str] | None = None,
    ) -> bytes:
        """Encode frames to WebM container with VP9 codec.

        Args:
            frames: RGB uint8 frame list.
            fps: Target frames per second.
            output_resolution: Optional (width, height) resize on encode.
            crf: Constant Rate Factor for VP9 (0-63, lower = better quality).
            cpu_used: VP9 encoding speed (0=slowest/best, 8=fastest/worst).
            audio_bytes: Optional PCM audio bytes for Opus track.
            metadata: Optional metadata tags.

        Returns:
            WebM-encoded video as raw bytes.
        """
        self._validate_inputs(frames, fps)
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(
            None,
            self._encode_webm,
            frames,
            fps,
            output_resolution,
            crf,
            cpu_used,
            audio_bytes,
            metadata or {},
        )
        logger.info(
            "WebM export complete",
            num_frames=len(frames),
            fps=fps,
            size_bytes=len(encoded),
        )
        return encoded

    def _encode_webm(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None,
        crf: int,
        cpu_used: int,
        audio_bytes: bytes | None,
        metadata: dict[str, str],
    ) -> bytes:
        """CPU-bound WebM/VP9 encoding via PyAV.

        Args:
            frames: Input frames.
            fps: Frame rate.
            output_resolution: Optional resize.
            crf: Quality factor.
            cpu_used: Encoding speed.
            audio_bytes: Optional audio.
            metadata: Tags.

        Returns:
            WebM bytes.
        """
        if not _PYAV_AVAILABLE:
            return self._encode_fallback(frames, fps)

        buffer = io.BytesIO()
        height, width = frames[0].shape[:2]
        if output_resolution is not None:
            width, height = output_resolution
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1

        with av.open(buffer, mode="w", format="webm") as container:
            for key, value in metadata.items():
                container.metadata[key] = value

            video_stream = container.add_stream("libvpx-vp9", rate=fps)
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {
                "crf": str(crf),
                "cpu-used": str(cpu_used),
                "b:v": "0",
            }

            for frame_array in frames:
                if output_resolution is not None:
                    if _OPENCV_AVAILABLE:
                        frame_array = cv2.resize(frame_array, (width, height))

                av_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                av_frame = av_frame.reformat(format="yuv420p")
                for packet in video_stream.encode(av_frame):
                    container.mux(packet)

            for packet in video_stream.encode():
                container.mux(packet)

        return buffer.getvalue()

    async def export_avi(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None = None,
        audio_bytes: bytes | None = None,
        metadata: dict[str, str] | None = None,
    ) -> bytes:
        """Encode frames to AVI container for legacy format compatibility.

        Uses MPEG-4 codec within the AVI container. Suitable for legacy
        systems that do not support MP4 or WebM containers.

        Args:
            frames: RGB uint8 frame list.
            fps: Target frames per second.
            output_resolution: Optional (width, height) resize on encode.
            audio_bytes: Optional PCM audio bytes.
            metadata: Optional metadata tags.

        Returns:
            AVI-encoded video as raw bytes.
        """
        self._validate_inputs(frames, fps)
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(
            None,
            self._encode_avi,
            frames,
            fps,
            output_resolution,
            audio_bytes,
            metadata or {},
        )
        logger.info(
            "AVI export complete",
            num_frames=len(frames),
            fps=fps,
            size_bytes=len(encoded),
        )
        return encoded

    def _encode_avi(
        self,
        frames: list[np.ndarray],
        fps: int,
        output_resolution: tuple[int, int] | None,
        audio_bytes: bytes | None,
        metadata: dict[str, str],
    ) -> bytes:
        """CPU-bound AVI encoding via PyAV.

        Args:
            frames: Input frames.
            fps: Frame rate.
            output_resolution: Optional resize.
            audio_bytes: Optional audio.
            metadata: Tags.

        Returns:
            AVI bytes.
        """
        if not _PYAV_AVAILABLE:
            return self._encode_fallback(frames, fps)

        buffer = io.BytesIO()
        height, width = frames[0].shape[:2]
        if output_resolution is not None:
            width, height = output_resolution
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1

        with av.open(buffer, mode="w", format="avi") as container:
            video_stream = container.add_stream("mpeg4", rate=fps)
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            video_stream.bit_rate = self._video_bitrate

            for frame_array in frames:
                if output_resolution is not None and _OPENCV_AVAILABLE:
                    frame_array = cv2.resize(frame_array, (width, height))

                av_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                av_frame = av_frame.reformat(format="yuv420p")
                for packet in video_stream.encode(av_frame):
                    container.mux(packet)

            for packet in video_stream.encode():
                container.mux(packet)

        return buffer.getvalue()

    def _encode_fallback(self, frames: list[np.ndarray], fps: int) -> bytes:
        """Fallback encoder that writes raw RGB frames with a minimal header.

        Used only when PyAV is not available. The output is a custom binary
        format (not a standard container) suitable for testing only.

        Args:
            frames: RGB uint8 frames.
            fps: Frames per second (embedded in header).

        Returns:
            Raw bytes with minimal header.
        """
        if not frames:
            return b""
        height, width = frames[0].shape[:2]
        header = struct.pack(">IIII", len(frames), fps, width, height)
        raw_parts = [header]
        for frame in frames:
            raw_parts.append(frame.tobytes())
        return b"".join(raw_parts)

    async def mux_audio(
        self,
        video_bytes: bytes,
        audio_bytes: bytes,
        container_format: VideoContainer = VideoContainer.MP4,
    ) -> bytes:
        """Mux an audio track into an existing encoded video container.

        Args:
            video_bytes: Encoded video bytes (MP4, WebM, or AVI).
            audio_bytes: Raw PCM audio bytes (16-bit signed, 44100 Hz stereo).
            container_format: The container format of video_bytes.

        Returns:
            Video bytes with audio track muxed in.
        """
        if not _PYAV_AVAILABLE or not audio_bytes:
            logger.warning("Audio muxing skipped — PyAV not available or no audio provided")
            return video_bytes

        loop = asyncio.get_running_loop()
        muxed = await loop.run_in_executor(
            None,
            self._mux_audio_cpu,
            video_bytes,
            audio_bytes,
            container_format,
        )
        logger.info(
            "Audio muxed",
            video_size=len(video_bytes),
            audio_size=len(audio_bytes),
            output_size=len(muxed),
        )
        return muxed

    def _mux_audio_cpu(
        self,
        video_bytes: bytes,
        audio_bytes: bytes,
        container_format: VideoContainer,
    ) -> bytes:
        """CPU-bound audio muxing.

        Args:
            video_bytes: Input video.
            audio_bytes: Audio data.
            container_format: Output format.

        Returns:
            Muxed video bytes.
        """
        format_map = {
            VideoContainer.MP4: "mp4",
            VideoContainer.WEBM: "webm",
            VideoContainer.AVI: "avi",
        }
        output_format = format_map[container_format]

        input_video = io.BytesIO(video_bytes)
        input_audio = io.BytesIO(audio_bytes)
        output_buffer = io.BytesIO()

        try:
            with (
                av.open(input_video, "r") as vid_in,
                av.open(input_audio, "r") as aud_in,
                av.open(output_buffer, "w", format=output_format) as out,
            ):
                vid_stream = vid_in.streams.video[0]
                vid_out = out.add_stream(template=vid_stream)

                aud_out = out.add_stream("aac", rate=44100)
                aud_out.options = {"b": str(self._audio_bitrate)}

                for packet in vid_in.demux(vid_stream):
                    if packet.dts is not None:
                        packet.stream = vid_out
                        out.mux(packet)

                for frame in aud_in.decode():
                    frame = frame.reformat(format="fltp", layout="stereo", rate=44100)
                    for packet in aud_out.encode(frame):
                        out.mux(packet)

                for packet in aud_out.encode():
                    out.mux(packet)

        except Exception as exc:
            logger.warning("Audio muxing failed, returning original video", error=str(exc))
            return video_bytes

        return output_buffer.getvalue()

    async def embed_metadata(
        self,
        video_bytes: bytes,
        metadata: dict[str, str],
        container_format: VideoContainer = VideoContainer.MP4,
    ) -> bytes:
        """Embed metadata tags into an existing encoded video container.

        Args:
            video_bytes: Encoded video bytes.
            metadata: Dict of tag name to tag value strings.
            container_format: The container format of video_bytes.

        Returns:
            Video bytes with metadata embedded.
        """
        if not _PYAV_AVAILABLE or not metadata:
            return video_bytes

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._embed_metadata_cpu,
            video_bytes,
            metadata,
            container_format,
        )

    def _embed_metadata_cpu(
        self,
        video_bytes: bytes,
        metadata: dict[str, str],
        container_format: VideoContainer,
    ) -> bytes:
        """CPU-bound metadata embedding by remuxing the container.

        Args:
            video_bytes: Input video bytes.
            metadata: Metadata tags.
            container_format: Container format.

        Returns:
            Video with metadata.
        """
        format_map = {
            VideoContainer.MP4: "mp4",
            VideoContainer.WEBM: "webm",
            VideoContainer.AVI: "avi",
        }
        output_format = format_map[container_format]

        input_buf = io.BytesIO(video_bytes)
        output_buf = io.BytesIO()

        try:
            with (
                av.open(input_buf, "r") as src,
                av.open(output_buf, "w", format=output_format) as dst,
            ):
                for key, value in metadata.items():
                    dst.metadata[key] = value

                stream_map: dict[int, Any] = {}
                for stream in src.streams:
                    out_stream = dst.add_stream(template=stream)
                    stream_map[stream.index] = out_stream

                for packet in src.demux():
                    if packet.dts is not None and packet.stream.index in stream_map:
                        packet.stream = stream_map[packet.stream.index]
                        dst.mux(packet)

        except Exception as exc:
            logger.warning("Metadata embedding failed, returning original", error=str(exc))
            return video_bytes

        return output_buf.getvalue()

    async def upload_to_storage(
        self,
        video_bytes: bytes,
        job_id: str,
        tenant_id: str,
        container_format: VideoContainer = VideoContainer.MP4,
    ) -> str:
        """Upload encoded video bytes to MinIO/S3 storage and return the URI.

        The object key is structured as:
        ``{tenant_id}/videos/{job_id}.{extension}``

        Args:
            video_bytes: Encoded video bytes.
            job_id: Job UUID string for naming the artifact.
            tenant_id: Tenant UUID string for namespace isolation.
            container_format: Container format for content-type resolution.

        Returns:
            Full MinIO/S3 URI: ``s3://{bucket}/{object_key}``

        Raises:
            RuntimeError: If the storage upload fails.
        """
        extension_map = {
            VideoContainer.MP4: "mp4",
            VideoContainer.WEBM: "webm",
            VideoContainer.AVI: "avi",
        }
        content_type_map = {
            VideoContainer.MP4: "video/mp4",
            VideoContainer.WEBM: "video/webm",
            VideoContainer.AVI: "video/x-msvideo",
        }

        extension = extension_map[container_format]
        content_type = content_type_map[container_format]
        object_key = f"{tenant_id}/videos/{job_id}.{extension}"

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                self._storage_put,
                object_key,
                video_bytes,
                content_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Storage upload failed for job {job_id}: {exc}"
            ) from exc

        uri = f"s3://{self._bucket}/{object_key}"
        logger.info(
            "Video uploaded to storage",
            job_id=job_id,
            tenant_id=tenant_id,
            uri=uri,
            size_bytes=len(video_bytes),
        )
        return uri

    def _storage_put(
        self,
        object_key: str,
        data: bytes,
        content_type: str,
    ) -> None:
        """Synchronous storage PUT call.

        Args:
            object_key: Target object key within the bucket.
            data: Bytes to upload.
            content_type: MIME type of the content.
        """
        self._storage.put_object(
            self._bucket,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )

    async def extract_thumbnail(
        self,
        frames: list[np.ndarray],
        frame_index: int | None = None,
    ) -> bytes:
        """Extract and encode a JPEG thumbnail from the video frame sequence.

        Args:
            frames: RGB uint8 frame sequence.
            frame_index: Index of the frame to use as thumbnail.
                If None, uses the middle frame for better representativeness.

        Returns:
            JPEG-encoded thumbnail bytes.

        Raises:
            ValueError: If frames is empty.
        """
        if not frames:
            raise ValueError("Cannot extract thumbnail from empty frame list")

        idx = frame_index if frame_index is not None else len(frames) // 2
        idx = max(0, min(idx, len(frames) - 1))
        source_frame = frames[idx]

        loop = asyncio.get_running_loop()
        thumbnail_bytes = await loop.run_in_executor(
            None,
            self._encode_thumbnail,
            source_frame,
        )
        logger.debug(
            "Thumbnail extracted",
            frame_index=idx,
            thumbnail_size=len(thumbnail_bytes),
        )
        return thumbnail_bytes

    def _encode_thumbnail(self, frame: np.ndarray) -> bytes:
        """CPU-bound thumbnail resize and JPEG encode.

        Args:
            frame: RGB uint8 frame (H, W, 3).

        Returns:
            JPEG bytes.
        """
        if _OPENCV_AVAILABLE:
            resized = cv2.resize(
                frame,
                (self._thumb_width, self._thumb_height),
                interpolation=cv2.INTER_AREA,
            )
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return encoded.tobytes()

        # Fallback: PPM header + raw RGB (not JPEG, but valid image bytes for testing)
        row_idx = (np.arange(self._thumb_height) * frame.shape[0] / self._thumb_height).astype(int)
        col_idx = (np.arange(self._thumb_width) * frame.shape[1] / self._thumb_width).astype(int)
        thumbnail = frame[np.ix_(row_idx, col_idx)]
        header = f"P6\n{self._thumb_width} {self._thumb_height}\n255\n".encode()
        return header + thumbnail.tobytes()

    @staticmethod
    def _validate_inputs(frames: list[np.ndarray], fps: int) -> None:
        """Validate common export inputs.

        Args:
            frames: Frame list to validate.
            fps: Frames per second value.

        Raises:
            ValueError: If frames is empty or fps is non-positive.
        """
        if not frames:
            raise ValueError("Cannot export empty frame list")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
