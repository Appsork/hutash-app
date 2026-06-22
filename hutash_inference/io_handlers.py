"""I/O handlers for converting between HTTP request data and Python values.

The server uses these to parse incoming request parameters according to
the types declared in manifest.json, and to serialize responses.
"""

import base64
import json
from typing import Any


# Module-level registry — types whose form value carries File bytes
# and that go through the sidecar's PCM-envelope decoder.
#
# Mirror of `api/services/models/schema_builder.py::INPUT_TYPE_MAP`'s
# `carries_file: True` rows. Ships baked into every model image, so
# this set is the in-container view; the sidecar reads its own SSOT.
# Models that introduce a new composite primitive call
# `register_carries_file_type()` at import time to extend the set
# without modifying this file.
_CARRIES_FILE_TYPES: set[str] = {
    "audio_file",
    "image_file",
    "video_file",
    "audio_with_transcript",
    "text_file",
}


def register_carries_file_type(type_id: str) -> None:
    """Register a primitive input type as carrying File bytes.

    Called at import time by model packages that introduce a new
    composite primitive the in-container PCM decoder must recognise.
    """
    _CARRIES_FILE_TYPES.add(type_id)


def carries_file(type_id: str) -> bool:
    """True if the type's form value carries File bytes (PCM envelope)."""
    return type_id in _CARRIES_FILE_TYPES


# Backwards-compatible public name. Existing callers do
# `declared_type in CARRIES_FILE_TYPES`; `in` works identically on a
# set, so the rename to a set semantics is a drop-in.
CARRIES_FILE_TYPES = _CARRIES_FILE_TYPES


def parse_input(value: Any, declared_type: str) -> Any:
    """Parse input value according to declared type from manifest.

    Args:
        value: Raw value from HTTP request
        declared_type: Type string from manifest (e.g., "string", "audio_file")

    Returns:
        Parsed value ready to pass to inference method
    """
    if declared_type in ("string", "text"):
        return str(value)

    if declared_type == "integer":
        return int(value)

    if declared_type == "number":
        return float(value)

    if declared_type == "boolean":
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    if declared_type in CARRIES_FILE_TYPES:
        # Sidecar-preprocessed PCM envelope. Aura's sidecar decodes
        # audio_file inputs via FFmpeg and forwards raw float32 PCM
        # under this shape so containers never need torchaudio /
        # torchcodec / FFmpeg for decoding.
        #
        # Shape:
        #   {
        #     "__pcm__": True,
        #     "pcm_b64": "<base64 raw float32 LE PCM>",
        #     "sample_rate": <int>,
        #     "channels": <int>,    # 1 or 2
        #   }
        #
        # Returns 16-bit PCM WAV bytes (with header) — universal
        # contract that works in every container (no torch import
        # required) and matches the `file: bytes` signature every
        # inference method already declares. Inference code that
        # writes to a temp file and passes the path to its model
        # keeps working unchanged.
        if isinstance(value, dict) and value.get("__pcm__"):
            import io
            import wave
            import numpy as np

            pcm_bytes = base64.b64decode(value["pcm_b64"])
            sample_rate = int(value.get("sample_rate", 16000))
            channels = int(value.get("channels", 1)) or 1
            # Convert float32 PCM → 16-bit WAV bytes
            # Universal contract — works without torch, matches all inference.py signatures
            f32 = np.frombuffer(pcm_bytes, dtype=np.float32)
            s16 = (f32 * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(channels)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(s16)
            return buf.getvalue()

        # Legacy path: raw bytes or base64 string. Kept so any
        # container that still receives undecoded bytes (e.g.
        # a manifest input without target_sample_rate /
        # target_channels declared) continues to work.
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            try:
                return base64.b64decode(value)
            except Exception:
                return value.encode("utf-8")
        return value

    if declared_type == "reference_asset":
        if isinstance(value, dict):
            return value
        return json.loads(value) if isinstance(value, str) else value

    # Unknown type — pass through
    return value


def serialize_output(value: Any, declared_type: str) -> Any:
    """Serialize output value according to declared type for HTTP response.

    Args:
        value: Raw output from inference method
        declared_type: Type string from manifest (e.g., "wav", "png")

    Returns:
        Serialized value for HTTP response (base64 for binary)
    """
    if declared_type in ("wav", "mp3", "flac", "png", "jpg", "webp", "mp4", "webm"):
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("utf-8")
        return value

    if declared_type == "text":
        return str(value)

    if declared_type == "json":
        return value

    return value
