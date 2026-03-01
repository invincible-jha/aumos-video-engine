"""Annotation exporter for synthetic video — GAP-90 competitive gap implementation.

Exports per-frame bounding box annotations in COCO JSON and YOLO txt formats
for computer vision model training. Annotations are generated from frame
metadata produced by the metadata extractor (known object locations).
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class FrameAnnotation:
    """Per-frame annotation with bounding boxes.

    Attributes:
        frame_index: Zero-based frame index in the video sequence.
        width: Frame width in pixels.
        height: Frame height in pixels.
        objects: List of object annotations with class and bounding box.
    """

    frame_index: int
    width: int
    height: int
    objects: list[dict[str, Any]] = field(default_factory=list)


class AnnotationExporter:
    """Exports video frame annotations in COCO JSON and YOLO txt formats.

    Designed to work with annotations generated from BlenderProc scene
    metadata (ground-truth object positions in 3D space projected to 2D).
    Produces ZIP archives containing per-frame or single consolidated files
    uploaded to MinIO storage.

    Args:
        category_map: Mapping from class name to integer category ID.
            Defaults to a standard synthetic video object taxonomy.
    """

    DEFAULT_CATEGORIES: list[dict[str, Any]] = [
        {"id": 1, "name": "person", "supercategory": "living"},
        {"id": 2, "name": "vehicle", "supercategory": "object"},
        {"id": 3, "name": "defect", "supercategory": "object"},
        {"id": 4, "name": "machinery", "supercategory": "object"},
        {"id": 5, "name": "animal", "supercategory": "living"},
        {"id": 6, "name": "face", "supercategory": "living"},
        {"id": 7, "name": "license_plate", "supercategory": "object"},
        {"id": 8, "name": "hazard", "supercategory": "event"},
    ]

    def __init__(
        self,
        category_map: dict[str, int] | None = None,
    ) -> None:
        if category_map is not None:
            self._category_map = category_map
        else:
            self._category_map = {cat["name"]: cat["id"] for cat in self.DEFAULT_CATEGORIES}
        self._log = logger.bind(component="annotation_exporter")

    async def export_coco(
        self,
        frames: list[FrameAnnotation],
        job_id: uuid.UUID,
    ) -> bytes:
        """Export COCO-format annotation JSON for all frames.

        Args:
            frames: Per-frame annotation data with bounding boxes.
            job_id: Job identifier embedded in COCO info block.

        Returns:
            UTF-8 encoded COCO JSON bytes.
        """
        self._log.info("annotation.export_coco_start", job_id=str(job_id), num_frames=len(frames))

        result = await asyncio.to_thread(self._build_coco_sync, frames, job_id)

        self._log.info(
            "annotation.export_coco_complete",
            job_id=str(job_id),
            num_annotations=len(result.get("annotations", [])),
        )
        return result

    def _build_coco_sync(
        self,
        frames: list[FrameAnnotation],
        job_id: uuid.UUID,
    ) -> bytes:
        """Build COCO JSON synchronously.

        Args:
            frames: Per-frame annotation data.
            job_id: Job identifier for info block.

        Returns:
            COCO JSON as UTF-8 bytes.
        """
        categories = [
            {"id": cat_id, "name": cat_name, "supercategory": "object"}
            for cat_name, cat_id in self._category_map.items()
        ]

        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = 1

        for frame in frames:
            image_id = frame.frame_index + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": f"frame_{frame.frame_index:06d}.jpg",
                    "width": frame.width,
                    "height": frame.height,
                    "frame_index": frame.frame_index,
                }
            )

            for obj in frame.objects:
                class_name: str = obj.get("class", "object")
                category_id: int = self._category_map.get(class_name, 1)

                # Bounding box in [x, y, width, height] (COCO format)
                x1: float = obj.get("x1", 0.0)
                y1: float = obj.get("y1", 0.0)
                x2: float = obj.get("x2", 0.0)
                y2: float = obj.get("y2", 0.0)
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                        "confidence": obj.get("confidence", 1.0),
                        "track_id": obj.get("track_id"),
                    }
                )
                annotation_id += 1

        coco_doc: dict[str, Any] = {
            "info": {
                "description": "AumOS synthetic video annotations",
                "version": "1.0",
                "year": 2026,
                "contributor": "AumOS aumos-video-engine",
                "job_id": str(job_id),
            },
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        return json.dumps(coco_doc, indent=2).encode("utf-8")

    async def export_yolo(
        self,
        frames: list[FrameAnnotation],
        job_id: uuid.UUID,
    ) -> bytes:
        """Export YOLO-format annotation txt files as a ZIP archive.

        Each frame produces one txt file (frame_000000.txt) with one
        annotation per line: <class_id> <cx> <cy> <w> <h> (normalized 0-1).

        Args:
            frames: Per-frame annotation data.
            job_id: Job identifier embedded in the archive comment.

        Returns:
            ZIP archive bytes containing per-frame YOLO txt files and classes.txt.
        """
        self._log.info("annotation.export_yolo_start", job_id=str(job_id), num_frames=len(frames))

        result = await asyncio.to_thread(self._build_yolo_zip_sync, frames, job_id)

        self._log.info("annotation.export_yolo_complete", job_id=str(job_id))
        return result

    def _build_yolo_zip_sync(
        self,
        frames: list[FrameAnnotation],
        job_id: uuid.UUID,
    ) -> bytes:
        """Build YOLO ZIP archive synchronously.

        Args:
            frames: Per-frame annotation data.
            job_id: Job identifier for archive comment.

        Returns:
            ZIP bytes with per-frame YOLO txt files.
        """
        buf = io.BytesIO()

        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.comment = f"AumOS YOLO annotations job={job_id}".encode()

            # Write classes.txt
            class_names = sorted(self._category_map.keys(), key=lambda n: self._category_map[n])
            zf.writestr("classes.txt", "\n".join(class_names))

            for frame in frames:
                lines: list[str] = []
                for obj in frame.objects:
                    class_name: str = obj.get("class", "object")
                    # YOLO class IDs are 0-indexed
                    yolo_class_id = self._category_map.get(class_name, 1) - 1

                    x1: float = obj.get("x1", 0.0)
                    y1: float = obj.get("y1", 0.0)
                    x2: float = obj.get("x2", 0.0)
                    y2: float = obj.get("y2", 0.0)

                    if frame.width == 0 or frame.height == 0:
                        continue

                    # Normalize to [0, 1]
                    cx = ((x1 + x2) / 2.0) / frame.width
                    cy = ((y1 + y2) / 2.0) / frame.height
                    w = (x2 - x1) / frame.width
                    h = (y2 - y1) / frame.height

                    if w <= 0 or h <= 0:
                        continue

                    lines.append(f"{yolo_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                txt_filename = f"labels/frame_{frame.frame_index:06d}.txt"
                zf.writestr(txt_filename, "\n".join(lines))

        return buf.getvalue()

    def frames_from_metadata(
        self,
        objects_per_frame: list[list[dict[str, Any]]],
        width: int,
        height: int,
    ) -> list[FrameAnnotation]:
        """Convert metadata extractor output to FrameAnnotation list.

        Args:
            objects_per_frame: Per-frame list of object dicts with class, x1, y1, x2, y2.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            List of FrameAnnotation objects ready for export.
        """
        return [
            FrameAnnotation(
                frame_index=idx,
                width=width,
                height=height,
                objects=objects,
            )
            for idx, objects in enumerate(objects_per_frame)
        ]
