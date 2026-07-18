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

import functools
import importlib.util
import inspect
import json
import logging
import os
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low VRAM mode — framework-aware device placement
# ---------------------------------------------------------------------------
# The engine decides, per host, how much VRAM a model may use and sets two env
# vars in the model process when Low VRAM mode is active:
#     HUTASH_VRAM_BUDGET_MB   free VRAM (MB) the model may claim
#     HUTASH_MODEL_FRAMEWORK  huggingface | llama-cpp | faster-whisper | custom
# server.py translates that budget into whatever knob the model's framework
# actually understands, so a model's inference.py needs NO Low-VRAM code of its
# own (the old get_device_map_kwargs import is gone).


def _apply_low_vram() -> None:
    """Translate the engine's VRAM budget into framework-specific signals.

    Runs at import, before any model loads. No budget → no-op (the model loads
    exactly as before). For HuggingFace it records the device_map/max_memory the
    from_pretrained hook injects; for llama-cpp an n_gpu_layers count the model
    reads back from HUTASH_GPU_LAYERS; other frameworks have no partial-load knob
    and just load normally.
    """
    budget = os.environ.get("HUTASH_VRAM_BUDGET_MB")
    framework = os.environ.get("HUTASH_MODEL_FRAMEWORK")
    if not budget:
        return  # Low VRAM not active, load normally

    try:
        budget_mb = int(budget)
    except ValueError:
        logger.warning("Invalid HUTASH_VRAM_BUDGET_MB=%s, ignoring", budget)
        return

    logger.info("Low VRAM mode: %dMB budget, framework=%s", budget_mb, framework)

    if framework == "huggingface":
        # accelerate/transformers do NOT read these as env vars — device_map and
        # max_memory are from_pretrained KWARGS. We stash the intent here and the
        # from_pretrained hook (installed before inference.py loads) reads it back
        # and injects the kwargs. See _install_hf_low_vram_hook.
        os.environ["ACCELERATE_DEVICE_MAP"] = "auto"
        os.environ["ACCELERATE_MAX_MEMORY"] = f'{{"0": "{budget_mb}MB", "cpu": "48GB"}}'
        logger.info("HuggingFace: device_map=auto, max_memory GPU=%dMB", budget_mb)

    elif framework == "llama-cpp":
        # ~300MB per GGUF layer (conservative across quantizations). The model's
        # inference.py reads HUTASH_GPU_LAYERS and passes it as n_gpu_layers.
        gpu_layers = max(1, budget_mb // 300)
        os.environ["HUTASH_GPU_LAYERS"] = str(gpu_layers)
        logger.info("llama-cpp: n_gpu_layers=%d (from %dMB budget)", gpu_layers, budget_mb)

    else:
        # faster-whisper, kokoro, chatterbox, custom — no partial-loading knob.
        # The engine should have blocked this with a red compatibility tag; if we
        # still got here, just load normally.
        logger.info(
            "Framework %s does not support partial loading, loading normally",
            framework,
        )


def _patch_from_pretrained(target_cls, device_map, max_memory) -> None:
    """Wrap one transformers class's ``from_pretrained`` classmethod to inject
    ``device_map`` / ``max_memory`` when the caller didn't pass them.

    Preserves the ``cls`` binding (so a subclass still loads as itself, not the
    patched base) and is idempotent via a sentinel attribute. Classes that only
    inherit ``from_pretrained`` (define none of their own) are skipped — the base
    patch covers them.
    """
    if getattr(target_cls, "_hutash_low_vram_patched", False):
        return
    raw = target_cls.__dict__.get("from_pretrained")
    if raw is None:
        return
    orig_func = raw.__func__ if isinstance(raw, classmethod) else raw

    @functools.wraps(orig_func)
    def wrapper(cls, *args, **kwargs):
        if "device_map" not in kwargs:
            kwargs["device_map"] = device_map
            if max_memory is not None and "max_memory" not in kwargs:
                kwargs["max_memory"] = max_memory
            logger.info(
                "Low VRAM: injected device_map=%s into %s.from_pretrained",
                device_map, getattr(cls, "__name__", cls),
            )
        return orig_func(cls, *args, **kwargs)

    target_cls.from_pretrained = classmethod(wrapper)
    target_cls._hutash_low_vram_patched = True


def _install_hf_low_vram_hook() -> None:
    """Make transformers honour the Low VRAM budget for HuggingFace models.

    RESEARCH NOTE: transformers/accelerate do NOT read ACCELERATE_DEVICE_MAP or
    ACCELERATE_MAX_MEMORY from the environment — ``device_map`` and ``max_memory``
    are ``from_pretrained`` keyword arguments. accelerate's own env vars configure
    the ``Accelerator`` / ``accelerate launch``, not ``from_pretrained``. So to
    place layers across GPU/CPU by budget WITHOUT editing each model's
    inference.py, we wrap ``from_pretrained`` on the common transformers base
    classes to inject those kwargs. Called before the model's inference.py is
    imported (in create_app), so its from_pretrained call gets the budget.

    No-op when Low VRAM huggingface mode is inactive, or when transformers is not
    importable (a non-HuggingFace model venv has nothing to patch).
    """
    device_map = os.environ.get("ACCELERATE_DEVICE_MAP")
    if not device_map:
        return  # only _apply_low_vram's huggingface branch sets this

    max_memory = None
    raw_mm = os.environ.get("ACCELERATE_MAX_MEMORY")
    if raw_mm:
        try:
            parsed = json.loads(raw_mm)
            # transformers wants int keys for GPU ids ("0" -> 0); "cpu"/"disk" stay str.
            max_memory = {
                (int(k) if isinstance(k, str) and k.isdigit() else k): v
                for k, v in parsed.items()
            }
        except (ValueError, AttributeError):
            logger.warning("Low VRAM: could not parse ACCELERATE_MAX_MEMORY=%s", raw_mm)

    try:
        from transformers import PreTrainedModel
    except Exception:  # noqa: BLE001 — transformers absent in non-HF venvs
        logger.info("Low VRAM: transformers not importable; HuggingFace hook skipped")
        return

    _patch_from_pretrained(PreTrainedModel, device_map, max_memory)
    try:
        from transformers.models.auto.auto_factory import _BaseAutoModelClass

        _patch_from_pretrained(_BaseAutoModelClass, device_map, max_memory)
    except Exception:  # noqa: BLE001 — Auto* internals moved; base patch still covers most
        pass
    logger.info(
        "Low VRAM: HuggingFace from_pretrained hook installed (device_map=%s)",
        device_map,
    )


# Apply the framework routing at import — before create_app() imports any model
# inference.py. Only reads/writes env vars, so importing server.py on the host
# (tests, smoke checks) with no budget set stays a no-op.
_apply_low_vram()


def model_dir() -> Path:
    """Directory holding this model's manifest.json + inference.py.

    In a container these live at ``/app`` (the image bakes them there). In a
    native venv there is no ``/app``; hutashd points ``HUTASH_MODEL_DIR`` at the
    model's package directory instead. Resolution order:
    ``HUTASH_MODEL_DIR`` → ``HUTASH_APP_DIR`` → ``/app`` (the container default,
    so existing images are unaffected).
    """
    d = os.environ.get("HUTASH_MODEL_DIR") or os.environ.get("HUTASH_APP_DIR")
    return Path(d) if d else Path("/app")


# Model file locations, resolved at call time so the same server code runs both
# in a container (/app) and in a native venv (HUTASH_MODEL_DIR). Kept as
# module-level accessors for readability at the call sites below.
def _manifest_path() -> Path:
    return model_dir() / "manifest.json"


def _inference_module_path() -> Path:
    return model_dir() / "inference.py"


def _third_party_licenses_path() -> Path:
    return model_dir() / "THIRD_PARTY_LICENSES.txt"


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
    """Load and parse the model's manifest.json (see model_dir())."""
    manifest_path = _manifest_path()
    if not manifest_path.exists():
        raise ModelLoadError(f"Manifest not found at {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)


def load_inference_module():
    """Dynamically import the model's inference.py (see model_dir())."""
    inference_path = _inference_module_path()
    if not inference_path.exists():
        raise ModelLoadError(f"inference.py not found at {inference_path}")

    spec = importlib.util.spec_from_file_location(
        "inference", inference_path
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

    # Low VRAM (HuggingFace): install the from_pretrained device-map hook BEFORE
    # the model's inference.py is imported, so its from_pretrained honours the
    # engine's VRAM budget. No-op unless HF Low VRAM mode is active.
    _install_hf_low_vram_hook()

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
        # THIRD_PARTY_LICENSES.txt is generated at container-build time and may
        # be absent in a native venv install, so it does not gate readiness —
        # /third-party-licenses simply 404s when it is missing. Model-loaded +
        # manifest-present is the readiness contract that holds in both runtimes.
        if not _manifest_path().exists():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "manifest_missing"},
            )
        return {"status": "ok"}

    @app.post("/offload")
    def offload():
        """Move the model off the GPU into RAM to free VRAM, keeping the process
        alive. The engine calls this to make room for another model — far faster
        than killing and reloading. Idempotent.
        """
        try:
            return instance.offload()
        except Exception as e:  # noqa: BLE001 — boundary guard
            raise HTTPException(status_code=500, detail=f"offload failed: {type(e).__name__}: {e}") from e

    @app.post("/reload")
    def reload_model():
        """Move the model back onto the GPU (from RAM). Fast — weights are
        already resident. The engine calls this before the model is next used.
        """
        try:
            return instance.reload()
        except Exception as e:  # noqa: BLE001 — boundary guard
            raise HTTPException(status_code=500, detail=f"reload failed: {type(e).__name__}: {e}") from e

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
        licenses_path = _third_party_licenses_path()
        if not licenses_path.exists():
            raise HTTPException(
                status_code=404,
                detail="THIRD_PARTY_LICENSES.txt not present in this install",
            )
        return Response(
            content=licenses_path.read_text(encoding="utf-8"),
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
