"""Video generator model adapter registry — GAP-88: Latest Generation Models.

Implements the same registry pattern as the image-engine (GAP-76) for video
generation. Supports lazy loading, in-process caching, and dynamic adapter
selection by model name.
"""

from __future__ import annotations

import threading
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class VideoModelAdapterRegistry:
    """Thread-safe registry for video generation model adapters.

    Manages lazy loading and caching of video generator adapters. Adapters
    are loaded on first use and reused across requests to avoid repeated
    model weight downloads.

    Registered model names:
        - "open_sora_v13" — Open-Sora v1.3 (default, 204 frames max, 720p)
        - "cogvideox_5b" — CogVideoX-5B (49 frames, 480x720)
        - "svd" — Stable Video Diffusion XT (legacy, 25 frames, 576x1024)

    Args:
        default_model: Name of the model to use when none is specified.
        device: Torch device string ("cuda", "cpu").
        cache_dir: Model weight cache directory.
        allowed_models: Whitelist of allowed model names. All if None.
    """

    REGISTERED_MODELS: dict[str, dict[str, Any]] = {
        "open_sora_v13": {
            "class": "OpenSoraAdapter",
            "module": "aumos_video_engine.adapters.open_sora_adapter",
            "kwargs": {"model_name": "open-sora-v1.3"},
            "max_frames": 204,
            "description": "Open-Sora v1.3 — 204 frames, 720p, DiT architecture",
        },
        "cogvideox_5b": {
            "class": "OpenSoraAdapter",
            "module": "aumos_video_engine.adapters.open_sora_adapter",
            "kwargs": {"model_name": "cogvideox-5b"},
            "max_frames": 49,
            "description": "CogVideoX-5B — 49 frames, 480x720",
        },
        "svd": {
            "class": "OpenSoraAdapter",
            "module": "aumos_video_engine.adapters.open_sora_adapter",
            "kwargs": {"model_name": "svd"},
            "max_frames": 25,
            "description": "Stable Video Diffusion XT — legacy, 25 frames (compat)",
        },
    }

    def __init__(
        self,
        default_model: str = "open_sora_v13",
        device: str = "cpu",
        cache_dir: str = "/tmp/model-cache",
        allowed_models: list[str] | None = None,
    ) -> None:
        if default_model not in self.REGISTERED_MODELS:
            raise ValueError(
                f"Unknown default model '{default_model}'. "
                f"Valid: {list(self.REGISTERED_MODELS)}"
            )
        self._default_model = default_model
        self._device = device
        self._cache_dir = cache_dir
        self._allowed_models = set(allowed_models or list(self.REGISTERED_MODELS))
        self._adapters: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._log = logger.bind(component="video_model_registry")

    def list_models(self) -> list[dict[str, Any]]:
        """Return metadata for all registered and allowed models.

        Returns:
            List of model metadata dicts with name, max_frames, and description.
        """
        return [
            {
                "name": name,
                "max_frames": meta["max_frames"],
                "description": meta["description"],
                "is_default": name == self._default_model,
                "allowed": name in self._allowed_models,
            }
            for name, meta in self.REGISTERED_MODELS.items()
        ]

    async def get_adapter(self, model_name: str | None = None) -> Any:
        """Retrieve a loaded adapter instance by name.

        Loads the adapter on first access (lazy loading). Subsequent calls
        return the cached instance without reloading.

        Args:
            model_name: Registered model name. Uses default if None.

        Returns:
            Loaded adapter instance implementing VideoGeneratorProtocol.

        Raises:
            ValueError: If model_name is not in allowed_models or not registered.
        """
        name = model_name or self._default_model

        if name not in self.REGISTERED_MODELS:
            raise ValueError(f"Unknown model '{name}'. Valid: {list(self.REGISTERED_MODELS)}")

        if name not in self._allowed_models:
            raise ValueError(f"Model '{name}' is not in the allowed models list for this tenant.")

        with self._lock:
            if name not in self._adapters:
                self._log.info("video_registry.load_start", model=name, device=self._device)
                adapter = await self._load_adapter(name)
                self._adapters[name] = adapter
                self._log.info("video_registry.load_complete", model=name)

        return self._adapters[name]

    async def _load_adapter(self, model_name: str) -> Any:
        """Instantiate and warm up a model adapter.

        Args:
            model_name: Registered model name.

        Returns:
            Initialized adapter instance.
        """
        import importlib

        meta = self.REGISTERED_MODELS[model_name]
        module = importlib.import_module(meta["module"])
        cls = getattr(module, meta["class"])

        kwargs: dict[str, Any] = {
            **meta["kwargs"],
            "device": self._device,
            "cache_dir": self._cache_dir,
        }

        adapter = cls(**kwargs)

        if hasattr(adapter, "load_model"):
            await adapter.load_model()

        return adapter

    async def get_default(self) -> Any:
        """Retrieve the default model adapter.

        Returns:
            Default adapter instance.
        """
        return await self.get_adapter(self._default_model)

    def max_frames_for(self, model_name: str | None = None) -> int:
        """Return the maximum frame count for a given model.

        Args:
            model_name: Model name. Defaults to default model.

        Returns:
            Maximum number of frames the model supports.
        """
        name = model_name or self._default_model
        return self.REGISTERED_MODELS.get(name, {}).get("max_frames", 25)
