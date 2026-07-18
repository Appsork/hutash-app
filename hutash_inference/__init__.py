"""Aura Inference Library

Shared framework for Aura model containers. Provides:
- Inference base class that all model inference classes inherit from
- @capability decorator for marking methods as HTTP endpoint handlers
- FastAPI server boilerplate
- I/O helpers (audio, image, file)
- Validation (model.yaml / manifest.json <-> inference.py compatibility)

See docs/strategy/model-ssot-architecture.md for architecture details.
"""

from hutash_inference.base import Inference, capability, resolve_local_weights_dir
from hutash_inference.errors import (
    HutashInferenceError,
    ValidationError,
    ModelLoadError,
    GenerationError,
)
from hutash_inference.vram import get_device_map_kwargs

__version__ = "0.2.0"

__all__ = [
    "Inference",
    "capability",
    "resolve_local_weights_dir",
    "HutashInferenceError",
    "ValidationError",
    "ModelLoadError",
    "GenerationError",
    "get_device_map_kwargs",
]
