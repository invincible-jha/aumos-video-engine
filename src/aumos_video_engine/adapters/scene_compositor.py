"""Multi-object scene composition adapter.

Handles spatial reasoning for object placement, depth ordering, occlusion,
lighting consistency, background-foreground blending, scene graph tracking,
and temporal object tracking across frames for synthetic video composition.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_OPENCV_AVAILABLE = False
try:
    import cv2  # type: ignore[import-untyped]

    _OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python-headless not installed — scene compositor using numpy fallbacks")


@dataclass
class SceneObject:
    """Represents a single object within a composed scene.

    Attributes:
        object_id: Unique identifier for tracking across frames.
        label: Semantic class label (e.g., "car", "person", "robot_arm").
        position: (x, y) centre position in normalised [0.0, 1.0] coordinates.
        depth: Depth value in [0.0, 1.0] where 0.0 is closest to camera.
        size: (width, height) in normalised coordinates.
        rotation_degrees: In-plane rotation angle in degrees.
        alpha: Transparency in [0.0, 1.0] (1.0 = fully opaque).
        pixel_data: Optional RGBA numpy array for the object sprite (H, W, 4).
        velocity: (dx, dy) per-frame movement in normalised coordinates.
        attributes: Arbitrary extra attributes for material/lighting config.
    """

    object_id: str
    label: str
    position: tuple[float, float]
    depth: float
    size: tuple[float, float]
    rotation_degrees: float = 0.0
    alpha: float = 1.0
    pixel_data: np.ndarray | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraph:
    """Directed acyclic graph representation of object relationships in a scene.

    Attributes:
        objects: Dict mapping object_id to SceneObject.
        parent_child: List of (parent_id, child_id) edges for attachment.
        lighting: Global lighting parameters for the scene.
        background_config: Background rendering parameters.
    """

    objects: dict[str, SceneObject]
    parent_child: list[tuple[str, str]]
    lighting: dict[str, Any]
    background_config: dict[str, Any]


class SceneCompositor:
    """Composes multi-object scenes with spatial reasoning and temporal tracking.

    Handles object placement with depth-based occlusion, lighting consistency
    across objects, background-foreground blending, and frame-to-frame object
    tracking to produce temporally consistent frame sequences.
    """

    def __init__(
        self,
        canvas_background_colour: tuple[int, int, int] = (128, 128, 128),
        lighting_ambient_intensity: float = 0.4,
        lighting_diffuse_intensity: float = 0.6,
        temporal_smoothing_alpha: float = 0.15,
    ) -> None:
        """Initialize SceneCompositor with rendering defaults.

        Args:
            canvas_background_colour: Default RGB background fill when no background
                image is provided.
            lighting_ambient_intensity: Ambient light strength in [0.0, 1.0].
            lighting_diffuse_intensity: Directional diffuse light strength.
            temporal_smoothing_alpha: EMA weight for position smoothing across frames.
                Higher values cause faster response but less smoothing.
        """
        self._bg_colour = canvas_background_colour
        self._ambient = lighting_ambient_intensity
        self._diffuse = lighting_diffuse_intensity
        self._ema_alpha = temporal_smoothing_alpha

        # Tracks smoothed positions per object_id across frame calls
        self._position_ema: dict[str, tuple[float, float]] = {}

    def build_scene_graph(
        self,
        objects: list[dict[str, Any]],
        scene_config: dict[str, Any],
    ) -> SceneGraph:
        """Construct a SceneGraph from raw object descriptors and scene config.

        Args:
            objects: List of object descriptor dicts. Each must contain:
                - object_id (str), label (str), position ([x, y]),
                  depth (float), size ([w, h]).
                Optional: rotation_degrees, alpha, velocity ([dx, dy]),
                attributes (dict).
            scene_config: Scene-level configuration. Relevant keys:
                - lighting (dict), background (dict), parent_child (list of pairs).

        Returns:
            Structured SceneGraph ready for composition.
        """
        scene_objects: dict[str, SceneObject] = {}
        for descriptor in objects:
            obj = SceneObject(
                object_id=descriptor["object_id"],
                label=descriptor["label"],
                position=tuple(descriptor["position"]),  # type: ignore[arg-type]
                depth=float(descriptor.get("depth", 0.5)),
                size=tuple(descriptor.get("size", [0.1, 0.1])),  # type: ignore[arg-type]
                rotation_degrees=float(descriptor.get("rotation_degrees", 0.0)),
                alpha=float(descriptor.get("alpha", 1.0)),
                velocity=tuple(descriptor.get("velocity", [0.0, 0.0])),  # type: ignore[arg-type]
                attributes=descriptor.get("attributes", {}),
            )
            scene_objects[obj.object_id] = obj

        parent_child: list[tuple[str, str]] = [
            (pair[0], pair[1]) for pair in scene_config.get("parent_child", [])
        ]

        return SceneGraph(
            objects=scene_objects,
            parent_child=parent_child,
            lighting=scene_config.get("lighting", {}),
            background_config=scene_config.get("background", {}),
        )

    async def place_objects_spatially(
        self,
        scene_graph: SceneGraph,
        resolution: tuple[int, int],
    ) -> list[tuple[SceneObject, tuple[int, int, int, int]]]:
        """Compute pixel-space bounding boxes for all objects with depth ordering.

        Args:
            scene_graph: Scene graph with normalised object positions and sizes.
            resolution: Canvas resolution as (width, height) in pixels.

        Returns:
            List of (object, bounding_box) tuples sorted by depth (back-to-front).
            bounding_box is (x1, y1, x2, y2) in pixel coordinates.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._compute_bounding_boxes,
            scene_graph,
            resolution,
        )

    def _compute_bounding_boxes(
        self,
        scene_graph: SceneGraph,
        resolution: tuple[int, int],
    ) -> list[tuple[SceneObject, tuple[int, int, int, int]]]:
        """CPU-bound bounding box computation with depth sort.

        Args:
            scene_graph: Scene graph.
            resolution: Canvas (width, height).

        Returns:
            Back-to-front sorted list of (object, bbox).
        """
        canvas_w, canvas_h = resolution
        placements: list[tuple[SceneObject, tuple[int, int, int, int]]] = []

        for obj in scene_graph.objects.values():
            cx = int(obj.position[0] * canvas_w)
            cy = int(obj.position[1] * canvas_h)
            half_w = int(obj.size[0] * canvas_w / 2)
            half_h = int(obj.size[1] * canvas_h / 2)

            x1 = max(0, cx - half_w)
            y1 = max(0, cy - half_h)
            x2 = min(canvas_w, cx + half_w)
            y2 = min(canvas_h, cy + half_h)

            placements.append((obj, (x1, y1, x2, y2)))

        # Sort back-to-front: highest depth value drawn first (furthest back)
        placements.sort(key=lambda item: item[0].depth, reverse=True)
        return placements

    def apply_lighting_consistency(
        self,
        frame: np.ndarray,
        lighting_config: dict[str, Any],
        object_bboxes: list[tuple[SceneObject, tuple[int, int, int, int]]],
    ) -> np.ndarray:
        """Apply consistent lighting to a composed frame based on scene lighting config.

        Adjusts brightness and adds a directional diffuse gradient to simulate
        a point or directional light source. Applies the same lighting uniformly
        across all rendered objects to maintain visual consistency.

        Args:
            frame: RGB uint8 canvas frame (H, W, 3).
            lighting_config: Dict with optional keys:
                - direction: [x, y] unit vector for directional light.
                - colour: [R, G, B] in [0, 255] for light tint.
                - intensity: float in [0.0, 2.0] for overall brightness scale.
            object_bboxes: Placed objects (for per-object adjustments if needed).

        Returns:
            Lighting-adjusted RGB uint8 frame.
        """
        intensity = float(lighting_config.get("intensity", 1.0))
        light_colour = np.array(
            lighting_config.get("colour", [255, 255, 255]),
            dtype=np.float32,
        ) / 255.0
        direction = lighting_config.get("direction", [0.5, -0.5])

        adjusted = frame.astype(np.float32) / 255.0

        # Apply ambient component
        adjusted = adjusted * self._ambient

        # Apply directional diffuse gradient
        height, width = frame.shape[:2]
        dir_x = float(direction[0])
        dir_y = float(direction[1])

        # Gradient map based on lighting direction
        x_grad = np.linspace(0, 1, width) * dir_x
        y_grad = np.linspace(0, 1, height) * dir_y
        gradient = np.outer(y_grad, np.ones(width)) + np.outer(np.ones(height), x_grad)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
        diffuse_map = gradient[:, :, np.newaxis] * self._diffuse

        adjusted = adjusted + diffuse_map * light_colour[np.newaxis, np.newaxis, :]
        adjusted = adjusted * intensity

        result = np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)
        return result

    def blend_background_foreground(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        alpha_mask: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend foreground objects onto a background canvas.

        Args:
            background: RGB uint8 background frame (H, W, 3).
            foreground: RGB uint8 foreground frame (H, W, 3).
            alpha_mask: Float32 alpha mask (H, W) with values in [0.0, 1.0].
                1.0 = fully foreground, 0.0 = fully background.

        Returns:
            Alpha-composited RGB uint8 frame.
        """
        bg_f = background.astype(np.float32)
        fg_f = foreground.astype(np.float32)
        alpha = alpha_mask[:, :, np.newaxis]

        composited = fg_f * alpha + bg_f * (1.0 - alpha)
        return np.clip(composited, 0, 255).astype(np.uint8)

    def render_object_to_canvas(
        self,
        canvas: np.ndarray,
        alpha_canvas: np.ndarray,
        obj: SceneObject,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render a single scene object onto the canvas at its bounding box.

        When the object has pixel_data (RGBA), it is resized and composited.
        When no pixel_data is provided, a solid-colour placeholder rectangle
        is drawn using the object's alpha and a deterministic colour derived
        from the object label.

        Args:
            canvas: RGB uint8 canvas (H, W, 3) — modified in-place.
            alpha_canvas: Float32 alpha accumulation (H, W) — modified in-place.
            obj: SceneObject to render.
            bbox: Pixel bounding box (x1, y1, x2, y2).

        Returns:
            Updated (canvas, alpha_canvas) tuple.
        """
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return canvas, alpha_canvas

        region_h = y2 - y1
        region_w = x2 - x1

        if obj.pixel_data is not None and obj.pixel_data.ndim == 3 and obj.pixel_data.shape[2] == 4:
            # Resize RGBA sprite to bounding box
            if _OPENCV_AVAILABLE:
                sprite = cv2.resize(obj.pixel_data, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Simple nearest-neighbour resize via slicing indices
                row_indices = (np.arange(region_h) * obj.pixel_data.shape[0] / region_h).astype(int)
                col_indices = (np.arange(region_w) * obj.pixel_data.shape[1] / region_w).astype(int)
                sprite = obj.pixel_data[np.ix_(row_indices, col_indices)]

            sprite_rgb = sprite[..., :3]
            sprite_alpha = sprite[..., 3].astype(np.float32) / 255.0 * obj.alpha
            canvas[y1:y2, x1:x2] = (
                sprite_rgb * sprite_alpha[:, :, np.newaxis]
                + canvas[y1:y2, x1:x2] * (1.0 - sprite_alpha[:, :, np.newaxis])
            ).astype(np.uint8)
            alpha_canvas[y1:y2, x1:x2] = np.maximum(alpha_canvas[y1:y2, x1:x2], sprite_alpha)
        else:
            # Placeholder: solid colour derived from label hash
            label_hash = hash(obj.label) % (256 ** 3)
            colour = np.array([
                (label_hash >> 16) & 0xFF,
                (label_hash >> 8) & 0xFF,
                label_hash & 0xFF,
            ], dtype=np.uint8)
            fill_alpha = obj.alpha
            existing_region = canvas[y1:y2, x1:x2].astype(np.float32)
            placeholder = np.ones((region_h, region_w, 3), dtype=np.float32) * colour
            blended = placeholder * fill_alpha + existing_region * (1.0 - fill_alpha)
            canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            alpha_canvas[y1:y2, x1:x2] = np.maximum(
                alpha_canvas[y1:y2, x1:x2],
                np.full((region_h, region_w), fill_alpha, dtype=np.float32),
            )

        return canvas, alpha_canvas

    def advance_scene_graph(
        self,
        scene_graph: SceneGraph,
        frame_index: int,
    ) -> SceneGraph:
        """Advance object positions by one frame using velocity vectors.

        Applies per-object velocity to update normalised positions and wraps
        positions at canvas boundaries (toroidal topology).

        Args:
            scene_graph: Current scene graph state.
            frame_index: Current frame number (used for deterministic motion).

        Returns:
            Updated SceneGraph with new object positions.
        """
        updated_objects: dict[str, SceneObject] = {}
        for obj_id, obj in scene_graph.objects.items():
            new_x = (obj.position[0] + obj.velocity[0]) % 1.0
            new_y = (obj.position[1] + obj.velocity[1]) % 1.0

            # Apply EMA smoothing to avoid jitter
            if obj_id in self._position_ema:
                prev_x, prev_y = self._position_ema[obj_id]
                new_x = (1.0 - self._ema_alpha) * prev_x + self._ema_alpha * new_x
                new_y = (1.0 - self._ema_alpha) * prev_y + self._ema_alpha * new_y
            self._position_ema[obj_id] = (new_x, new_y)

            updated_obj = SceneObject(
                object_id=obj.object_id,
                label=obj.label,
                position=(new_x, new_y),
                depth=obj.depth,
                size=obj.size,
                rotation_degrees=obj.rotation_degrees,
                alpha=obj.alpha,
                pixel_data=obj.pixel_data,
                velocity=obj.velocity,
                attributes=obj.attributes,
            )
            updated_objects[obj_id] = updated_obj

        return SceneGraph(
            objects=updated_objects,
            parent_child=scene_graph.parent_child,
            lighting=scene_graph.lighting,
            background_config=scene_graph.background_config,
        )

    async def compose_frame(
        self,
        scene_graph: SceneGraph,
        resolution: tuple[int, int],
        background_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compose a single video frame from the current scene graph state.

        Renders objects in depth order (back-to-front), applies lighting,
        and blends onto the background canvas.

        Args:
            scene_graph: Current scene state (object positions, lighting).
            resolution: Output frame size as (width, height).
            background_frame: Optional background image (H, W, 3) RGB uint8.
                If None, a solid-colour background is used.

        Returns:
            Composed RGB uint8 frame (H, W, 3).
        """
        canvas_w, canvas_h = resolution

        if background_frame is not None:
            if _OPENCV_AVAILABLE:
                canvas = cv2.resize(background_frame, (canvas_w, canvas_h))
            else:
                canvas = background_frame[:canvas_h, :canvas_w].copy() if (
                    background_frame.shape[0] >= canvas_h and background_frame.shape[1] >= canvas_w
                ) else np.full((canvas_h, canvas_w, 3), self._bg_colour, dtype=np.uint8)
        else:
            canvas = np.full((canvas_h, canvas_w, 3), self._bg_colour, dtype=np.uint8)

        alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        placements = await self.place_objects_spatially(scene_graph, resolution)
        for obj, bbox in placements:
            canvas, alpha_canvas = self.render_object_to_canvas(canvas, alpha_canvas, obj, bbox)

        canvas = self.apply_lighting_consistency(canvas, scene_graph.lighting, placements)
        return canvas

    async def compose_sequence(
        self,
        objects: list[dict[str, Any]],
        scene_config: dict[str, Any],
        num_frames: int,
        resolution: tuple[int, int],
        background_frames: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Compose a full multi-frame video sequence from object descriptors.

        Builds the scene graph, then advances and renders each frame while
        tracking object positions temporally.

        Args:
            objects: List of object descriptor dicts (see build_scene_graph).
            scene_config: Scene-level configuration dict.
            num_frames: Number of frames to generate.
            resolution: Output frame size as (width, height).
            background_frames: Optional list of background frames per frame.

        Returns:
            List of RGB uint8 composed frames (H, W, 3).
        """
        scene_graph = self.build_scene_graph(objects, scene_config)
        frames: list[np.ndarray] = []

        logger.info(
            "Composing scene sequence",
            num_frames=num_frames,
            num_objects=len(scene_graph.objects),
            resolution=resolution,
        )

        for frame_idx in range(num_frames):
            bg = background_frames[frame_idx] if background_frames and frame_idx < len(background_frames) else None
            composed = await self.compose_frame(scene_graph, resolution, background_frame=bg)
            frames.append(composed)

            if frame_idx < num_frames - 1:
                scene_graph = self.advance_scene_graph(scene_graph, frame_idx)

        return frames

    async def validate_composition_quality(
        self,
        frames: list[np.ndarray],
        scene_graph: SceneGraph,
        resolution: tuple[int, int],
    ) -> dict[str, Any]:
        """Validate composition quality by checking coverage and object visibility.

        Analyses whether objects are properly visible and whether the scene
        has adequate coverage without extreme occlusion.

        Args:
            frames: Composed RGB uint8 frame sequence.
            scene_graph: Scene graph used for composition.
            resolution: Canvas resolution.

        Returns:
            Dict with keys: object_visibility (dict per object_id),
            canvas_coverage_ratio, occlusion_warnings, quality_pass (bool).
        """
        canvas_w, canvas_h = resolution
        total_pixels = canvas_w * canvas_h

        placements = await self.place_objects_spatially(scene_graph, resolution)
        object_visibility: dict[str, float] = {}
        occlusion_warnings: list[str] = []

        # Compute each object's bbox area as fraction of canvas
        for obj, (x1, y1, x2, y2) in placements:
            bbox_area = max(0, (x2 - x1) * (y2 - y1))
            visibility_ratio = bbox_area / total_pixels
            object_visibility[obj.object_id] = round(visibility_ratio, 4)
            if visibility_ratio < 0.001:
                occlusion_warnings.append(
                    f"Object '{obj.object_id}' ({obj.label}) has very small visible area: "
                    f"{visibility_ratio:.4f}"
                )

        total_coverage = sum(object_visibility.values())
        canvas_coverage_ratio = float(np.clip(total_coverage, 0.0, 1.0))
        quality_pass = len(occlusion_warnings) == 0 and canvas_coverage_ratio > 0.01

        logger.info(
            "Composition quality validation complete",
            num_objects=len(scene_graph.objects),
            coverage=canvas_coverage_ratio,
            warnings=len(occlusion_warnings),
            quality_pass=quality_pass,
        )

        return {
            "object_visibility": object_visibility,
            "canvas_coverage_ratio": canvas_coverage_ratio,
            "occlusion_warnings": occlusion_warnings,
            "quality_pass": quality_pass,
        }
