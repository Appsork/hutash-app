"""Standardized error types for Aura Inference library."""


class HutashInferenceError(Exception):
    """Base class for all Aura inference errors."""
    pass


class ValidationError(HutashInferenceError):
    """Raised when model.yaml and inference.py don't match."""
    pass


class ModelLoadError(HutashInferenceError):
    """Raised when model fails to load at container startup."""
    pass


class GenerationError(HutashInferenceError):
    """Raised when inference fails during a generation request."""
    pass


class CapabilityNotFoundError(HutashInferenceError):
    """Raised when a requested capability isn't implemented."""
    pass
