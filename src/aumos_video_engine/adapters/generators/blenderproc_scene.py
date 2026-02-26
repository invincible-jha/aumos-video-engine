"""BlenderProc scene composition adapter.

Wraps BlenderProc (procedural Blender scene generation) to render structured
3D domain scenes into video frame sequences for manufacturing, surveillance,
and traffic domain synthetic data generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_BLENDERPROC_AVAILABLE = False
try:
    import blenderproc  # type: ignore[import-untyped]

    _BLENDERPROC_AVAILABLE = True
except ImportError:
    logger.warning("blenderproc not installed — 3D scene composition unavailable")


class BlenderProcSceneComposer:
    """Renders 3D domain scenes using BlenderProc.

    Composes scenes from structured configuration (lighting, camera path,
    environment, object placement) and renders frame sequences suitable
    for domain-specific synthetic video datasets.

    Note: BlenderProc requires Blender to be installed. When BlenderProc
    is not available, falls back to generating placeholder gradient frames
    for testing and development workflows.
    """

    def __init__(self, blender_path: str | None = None) -> None:
        """Initialize BlenderProcSceneComposer.

        Args:
            blender_path: Optional path to Blender executable. Uses system
                default if not specified.
        """
        self._blender_path = blender_path

    async def compose_scene(
        self,
        scene_config: dict[str, Any],
        objects: list[dict[str, Any]],
        num_frames: int,
        fps: int,
        resolution: tuple[int, int],
    ) -> list[np.ndarray]:
        """Render a 3D scene configuration into a video frame sequence.

        Args:
            scene_config: Scene parameters including:
                - lighting: List of light descriptors (type, position, energy)
                - camera_path: List of camera keyframe positions and orientations
                - environment: HDR environment map path or solid color
                - render_settings: Blender render engine params (samples, denoising)
            objects: List of 3D object descriptors:
                - id: Object identifier
                - model_path: Path to .blend/.obj/.glb asset
                - position: [x, y, z] world position
                - rotation: [rx, ry, rz] Euler rotation in degrees
                - scale: [sx, sy, sz] scale factors
                - material_config: Material overrides dict
            num_frames: Number of frames to render.
            fps: Target frames per second (sets render FPS).
            resolution: Output resolution (width, height).

        Returns:
            List of RGB uint8 numpy arrays (H, W, 3).
        """
        width, height = resolution

        if not _BLENDERPROC_AVAILABLE:
            logger.warning(
                "BlenderProc not available — generating placeholder frames",
                num_frames=num_frames,
                resolution=resolution,
            )
            return self._generate_placeholder_frames(
                num_frames=num_frames,
                width=width,
                height=height,
                scene_config=scene_config,
            )

        return await self._render_with_blenderproc(
            scene_config=scene_config,
            objects=objects,
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
        )

    async def _render_with_blenderproc(
        self,
        scene_config: dict[str, Any],
        objects: list[dict[str, Any]],
        num_frames: int,
        fps: int,
        width: int,
        height: int,
    ) -> list[np.ndarray]:
        """Execute BlenderProc rendering pipeline.

        Args:
            scene_config: Scene configuration dict.
            objects: Object descriptor list.
            num_frames: Number of frames to render.
            fps: Frame rate.
            width: Output width in pixels.
            height: Output height in pixels.

        Returns:
            Rendered frames as RGB uint8 numpy arrays.
        """
        import blenderproc  # type: ignore[import-untyped]

        blenderproc.init()

        # Configure render resolution
        blenderproc.camera.set_resolution(width, height)

        # Load objects into scene
        loaded_objects = []
        for obj_config in objects:
            model_path = obj_config.get("model_path", "")
            if not model_path:
                continue
            try:
                obj = blenderproc.loader.load_obj(model_path)
                position = obj_config.get("position", [0, 0, 0])
                rotation = obj_config.get("rotation", [0, 0, 0])
                scale = obj_config.get("scale", [1, 1, 1])
                for mesh_obj in obj:
                    mesh_obj.set_location(position)
                    mesh_obj.set_rotation_euler([r * 3.14159 / 180.0 for r in rotation])
                    mesh_obj.set_scale(scale)
                loaded_objects.extend(obj)
            except Exception as exc:
                logger.warning("Failed to load object", model_path=model_path, error=str(exc))

        # Configure lighting
        for light_config in scene_config.get("lighting", []):
            light = blenderproc.types.Light()
            light.set_type(light_config.get("type", "POINT").upper())
            light.set_location(light_config.get("position", [0, 5, 5]))
            light.set_energy(light_config.get("energy", 1000))

        # Configure camera path (keyframes)
        camera_path = scene_config.get("camera_path", [{"position": [0, -5, 2], "rotation": [75, 0, 0]}])
        for frame_idx, keyframe in enumerate(camera_path):
            rotation_matrix = blenderproc.camera.rotation_from_forward_vec(
                keyframe.get("forward", [0, 0, -1])
            )
            cam2world_matrix = blenderproc.math.build_transformation_mat(
                keyframe.get("position", [0, -5, 2]),
                rotation_matrix,
            )
            blenderproc.camera.add_camera_pose(cam2world_matrix, frame=frame_idx)

        # Render settings
        render_settings = scene_config.get("render_settings", {})
        blenderproc.renderer.set_max_amount_of_samples(render_settings.get("samples", 64))
        blenderproc.renderer.set_noise_threshold(render_settings.get("noise_threshold", 0.01))

        # Render frames
        render_data = blenderproc.renderer.render()
        color_images = render_data.get("colors", [])

        frames: list[np.ndarray] = []
        for color_image in color_images[:num_frames]:
            frame_array = np.array(color_image, dtype=np.uint8)
            if frame_array.shape[-1] == 4:
                frame_array = frame_array[:, :, :3]  # Drop alpha channel
            frames.append(frame_array)

        logger.info("BlenderProc rendering complete", num_frames=len(frames))
        return frames

    def _generate_placeholder_frames(
        self,
        num_frames: int,
        width: int,
        height: int,
        scene_config: dict[str, Any],
    ) -> list[np.ndarray]:
        """Generate placeholder gradient frames when BlenderProc is unavailable.

        Creates a simple animated gradient sequence for testing pipelines
        without requiring a full Blender installation.

        Args:
            num_frames: Number of frames to generate.
            width: Frame width in pixels.
            height: Frame height in pixels.
            scene_config: Scene config (used for color hints).

        Returns:
            List of gradient frames as RGB uint8 numpy arrays.
        """
        frames = []
        for frame_idx in range(num_frames):
            progress = frame_idx / max(num_frames - 1, 1)
            # Animated gradient: shifts from blue-grey to warm grey
            red = int(80 + 40 * progress)
            green = int(90 + 30 * progress)
            blue = int(120 - 30 * progress)
            frame = np.full((height, width, 3), [red, green, blue], dtype=np.uint8)
            frames.append(frame)
        return frames

    async def is_available(self) -> bool:
        """Check whether BlenderProc rendering is available.

        Returns:
            True if BlenderProc is installed; False otherwise.
        """
        return _BLENDERPROC_AVAILABLE
