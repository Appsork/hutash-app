"""FastAPI server for Aura model containers.

Every model container runs this server. It:
1. Reads /app/manifest.json to know what endpoints to expose
2. Imports /app/inference.py to find the Inference subclass
3. Instantiates and loads the model
4. Validates inference.py matches manifest.json
5. Wires @capability methods to HTTP endpoints from manifest
6. Starts listening on the configured port

Developers writing model inference code never need to touch this file.
Their only Python file is inference.py.

Container invocation (factory pattern â€” no module-level side effects):
    uvicorn --factory hutash_inference.server:create_app \
        --host 0.0.0.0 --port ${PORT}

The module has no top-level app instantiation, so importing it on the
host (tests, smoke checks) never triggers the container-only startup
path. `create_app()` is only invoked when explicitly called.
"""

import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from hutash_inference.base import Inference, get_capabilities
from hutash_inference.errors import (
    HutashInferenceError,
    GenerationError,
    ModelLoadError,
)
from hutash_inference.io_handlers import parse_input, serialize_output
from hutash_inference.logging import configure_logging
from hutash_inference.validation import validate_inference_matches_manifest


# Paths inside the container
MANIFEST_PATH = Path("/app/manifest.json")
INFERENCE_MODULE_PATH = Path("/app/inference.py")
THIRD_PARTY_LICENSES_PATH = Path("/app/THIRD_PARTY_LICENSES.txt")


# Composite primitive fanout â€” when an input's declared type is a key
# in this map, the dispatcher also copies additional wire fields named
# `<input_name><suffix>` from raw_data into the method kwargs. Mirrors
# the frontend convention where a composite primitive (e.g. the React
# `AudioWithTranscript` component) writes ONE manifest input but TWO
# RHF form keys â†’ splitFormValues sends both to the wire under
# `<input_name>` and `<input_name><suffix>`.
#
# Sibling to `CARRIES_FILE_TYPES` in io_handlers.py and `carriesFile`
# in src/ui/inputs/InputRegistry.ts. When adding a new composite
# primitive that fans out to multiple wire fields, add it here.
COMPOSITE_FANOUT: dict[str, tuple[str, ...]] = {
    "audio_with_transcript": ("_transcript",),
}


def load_manifest() -> dict:
    """Load and parse manifest.json from /app/."""
    if not MANIFEST_PATH.exists():
        raise ModelLoadError(f"Manifest not found at {MANIFEST_PATH}")
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_inference_module():
    """Dynamically import /app/inference.py."""
    if not INFERENCE_MODULE_PATH.exists():
        raise ModelLoadError(f"inference.py not found at {INFERENCE_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "inference", INFERENCE_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["inference"] = module
    spec.loader.exec_module(module)
    return module


def find_inference_class(module) -> type[Inference]:
    """Find the Inference subclass in the given module."""
    for name in dir(module):
        obj = getattr(module, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, Inference)
            and obj is not Inference
        ):
            return obj
    raise ModelLoadError(
        "No Inference subclass found in inference.py. "
        "Define a class inheriting from hutash_inference.Inference."
    )


def make_capability_handler(method, spec: dict):
    """Create an async HTTP handler for a capability method.

    Parses request data according to declared inputs + controls,
    calls the method, serializes the result.

    Controls with implementation_status in {"stub", "planned"} are
    UI-only: the manifest declares them, but inference.py does not
    accept them as kwargs. We inspect the method's signature once
    at wiring time and drop any incoming kwargs the method doesn't
    accept â€” lets Core API forward stub values without erroring.
    """
    inputs_spec = spec.get("inputs", {}) or {}
    controls_spec = spec.get("controls", {}) or {}
    outputs_spec = spec.get("outputs", {}) or {}

    # If the capability declares exactly one output AND its type is a
    # binary format, respond with raw bytes + the matching Content-Type
    # instead of JSON-wrapping base64. Core API's _handle_audio_response
    # does `response.content â†’ write_bytes(...)` and has no decoding
    # step, so JSON-wrapped binary lands on disk verbatim. Raw bytes
    # restores pre-SSOT wire behavior for audio / image / video models
    # with one binary output. Multi-field dicts (STT's text+segments+
    # language+duration) and text/json outputs stay JSON.
    _BINARY_TYPE_TO_MEDIA = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "mp4": "video/mp4",
        "webm": "video/webm",
    }
    single_binary_output = None
    if len(outputs_spec) == 1:
        only_key, only_spec = next(iter(outputs_spec.items()))
        media_type = _BINARY_TYPE_TO_MEDIA.get(only_spec.get("type", ""))
        if media_type is not None:
            single_binary_output = (only_key, media_type)

    sig = inspect.signature(method)
    accepts_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    accepted_params = {
        name for name in sig.parameters if name != "self"
    }
    logger = logging.getLogger("hutash_inference.handler")

    async def handler(request: Request):
        try:
            if request.headers.get("content-type", "").startswith("multipart/"):
                form = await request.form()
                raw_data = {}
                for key in form.keys():
                    value = form[key]
                    # FastAPI UploadFile â†’ read bytes for file-typed inputs.
                    # Detected via duck-typing on read() + filename so we
                    # don't have to import starlette.datastructures here.
                    if hasattr(value, "read") and hasattr(value, "filename"):
                        raw_data[key] = await value.read()
                    else:
                        raw_data[key] = value
            else:
                raw_data = await request.json()

            # Core API's build_json_payload wraps controls under a
            # `parameters` key: {"prompt": "...", "parameters": {...}}.
            # hutash_inference expects flat top-level keys matching the
            # manifest's input + control names. Unwrap `parameters`
            # here so both shapes work â€” this is a transition bridge;
            # once Core API sends SSOT-native flat bodies, the unwrap
            # is a no-op (no `parameters` key present).
            if isinstance(raw_data, dict) and isinstance(raw_data.get("parameters"), dict):
                nested = raw_data.pop("parameters")
                for k, v in nested.items():
                    raw_data.setdefault(k, v)

            # Unwrap fields{} envelope (new wire format from Core API)
            if isinstance(raw_data, dict) and isinstance(raw_data.get("fields"), dict):
                nested = raw_data.pop("fields")
                raw_data.update(nested)

            parsed_args: dict = {}
            for param_name, param_spec in inputs_spec.items():
                if param_name in raw_data:
                    parsed_args[param_name] = parse_input(
                        raw_data[param_name], param_spec["type"],
                    )
                elif param_spec.get("required", False):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required input: {param_name}",
                    )

                # Composite primitive fanout â€” see COMPOSITE_FANOUT
                # above. Pure passthrough (no parse_input â€” the suffix
                # fields are sibling values, not file-shaped). The
                # inference method declares them in its signature
                # (e.g. `voice_ref_transcript: str | None = None`).
                input_type = param_spec.get("type", "")
                if input_type in COMPOSITE_FANOUT:
                    for suffix in COMPOSITE_FANOUT[input_type]:
                        fanout_key = f"{param_name}{suffix}"
                        if fanout_key in raw_data:
                            parsed_args[fanout_key] = raw_data[fanout_key]

            for param_name, param_spec in controls_spec.items():
                if param_name in raw_data:
                    parsed_args[param_name] = parse_input(
                        raw_data[param_name], param_spec["type"],
                    )
                # else: method uses its default

            if not accepts_var_keyword:
                dropped = [k for k in parsed_args if k not in accepted_params]
                for k in dropped:
                    logger.debug(
                        "dropping non-accepted kwarg %r for capability method %s",
                        k, method.__qualname__,
                    )
                    parsed_args.pop(k, None)

            result = method(**parsed_args)

            # Single-binary-output shortcut: return raw bytes with
            # the matching Content-Type. Lets Core API's
            # _handle_audio_response write response.content directly
            # to disk as a valid WAV/PNG/MP4/etc.
            if single_binary_output is not None and isinstance(result, dict):
                key, media_type = single_binary_output
                value = result.get(key)
                if isinstance(value, (bytes, bytearray)):
                    return Response(content=bytes(value), media_type=media_type)
                # Fall through to JSON if the method returned something
                # other than bytes under the declared key (shouldn't
                # happen for a correctly-implemented capability).

            if isinstance(result, dict):
                serialized = {}
                for key, value in result.items():
                    if key in outputs_spec:
                        serialized[key] = serialize_output(
                            value, outputs_spec[key]["type"]
                        )
                    else:
                        serialized[key] = value
                return JSONResponse(content=serialized)
            return JSONResponse(content={"result": result})

        except GenerationError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except HutashInferenceError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 â€” boundary guard
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {type(e).__name__}: {e}",
            ) from e

    return handler


def create_app() -> FastAPI:
    """Build the FastAPI app by reading manifest and wiring inference."""
    manifest = load_manifest()
    model_id = manifest.get("model_id", "unknown")
    configure_logging(model_id)

    module = load_inference_module()
    InferenceClass = find_inference_class(module)

    instance = InferenceClass(config=manifest)

    try:
        instance.load()
        instance.mark_loaded()
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}") from e

    validate_inference_matches_manifest(instance, manifest)

    # Image in manifest is a fully-qualified registry string like
    # "ghcr.io/appsork/aura-kokoro:v0.1.0" â€” the tag is after the last ':'.
    image_ref = manifest.get("image", "")
    image_tag = image_ref.rsplit(":", 1)[-1] if ":" in image_ref else "unknown"
    app = FastAPI(
        title=f"Aura Model: {model_id}",
        version=image_tag,
    )

    @app.get("/health")
    def health():
        """Container readiness gate.

        Returns 200 only when every data path this container exposes is
        ready. Specifically:
          (a) instance._is_loaded â€" the model finished load()
          (b) THIRD_PARTY_LICENSES.txt readable on disk â€" backs
              /third-party-licenses, which the host's install-time
              cache populate fetches once and stores in SQLite KV
          (c) manifest.json readable on disk â€" backs /manifest, which
              the host's install-time manifest cache populate fetches

        503 here causes the host backend's _check_health() probe in
        docker_manager.py â€" and the Docker daemon's HEALTHCHECK â€" to
        mark the container unhealthy. ensure_model_running() in the
        host waits on exactly that 200 transition.

        Contract: when /health returns 200, every other endpoint this
        container serves is ready. A single-shot fetch from the host
        backend will succeed.
        """
        if not instance._is_loaded:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "model_not_loaded"},
            )
        if not THIRD_PARTY_LICENSES_PATH.exists():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "licenses_missing"},
            )
        if not MANIFEST_PATH.exists():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "manifest_missing"},
            )
        return {"status": "ok"}

    @app.get("/manifest")
    def get_manifest():
        return manifest

    @app.get("/third-party-licenses")
    def get_third_party_licenses():
        """Return the bundled THIRD_PARTY_LICENSES.txt as plain text.

        Proxied by Core API at GET /api/v1/models/{id}/third-party-licenses
        so the desktop app's Settings â†’ About / Licenses tab can display
        per-container Python-package attributions for the live fleet.
        """
        if not THIRD_PARTY_LICENSES_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="THIRD_PARTY_LICENSES.txt not present in this image",
            )
        return Response(
            content=THIRD_PARTY_LICENSES_PATH.read_text(encoding="utf-8"),
            media_type="text/plain; charset=utf-8",
        )

    capabilities = get_capabilities(instance)
    declared_capabilities = manifest.get("capabilities", {}) or {}

    for cap_id, cap_spec in declared_capabilities.items():
        endpoint = cap_spec.get("endpoint", f"/{cap_id}")
        method = capabilities[cap_id]
        handler = make_capability_handler(method, cap_spec)
        app.add_api_route(endpoint, handler, methods=["POST"], name=cap_id)

    return app


# Factory pattern only â€” no module-level app instantiation.
# Inside containers, uvicorn loads this via:
#   uvicorn --factory hutash_inference.server:create_app --host 0.0.0.0 --port ${PORT}
# On dev machines and in tests, create_app() is called explicitly when needed.
