"""Inference base class and @capability decorator.

Every model's inference.py defines a class that inherits from Inference
and uses @capability to mark methods as capability handlers. The
hutash_inference server reads manifest.json, finds the @capability methods,
and wires them to HTTP endpoints.
"""

import io
import logging
import tempfile
from typing import Callable, Optional


def capability(capability_id: str) -> Callable:
    """Decorator marking a method as implementing a capability.

    The capability_id must match a key under 'capabilities' in the
    model's model.yaml file.

    Example:
        @capability("voice-clone")
        def clone_voice(self, text: str, reference_audio: bytes):
            ...

    Args:
        capability_id: ID matching a key in model.yaml capabilities

    Returns:
        The original function with metadata attached
    """
    def decorator(func: Callable) -> Callable:
        func._aura_capability = capability_id
        return func
    return decorator


def resolve_local_weights_dir(model_id: str) -> str:
    """Resolve the mounted weights snapshot dir for ``model_id``'s PINNED commit.

    Weights-out: the model's weights are downloaded on the host and mounted at
    ``HF_HUB_CACHE`` (``/weights``). This returns
    ``<cache>/models--*/snapshots/<commit>/`` for the EXACT pinned commit so a
    model can load from a local path (e.g. ``from_local(ckpt_dir=...)``) instead
    of going through the HuggingFace cache / ``hf_hub_download`` (which needs a
    blob/etag structure the staged cache does not have offline).

    The pinned commit comes from ``HUTASH_HF_REVISION``, injected by
    docker_manager from the catalogue's ``hf_revision``. This deliberately does
    NOT fall back to "the first/only snapshot" — if the pinned commit's dir is
    missing it raises, so a drifted/incomplete mount fails loudly instead of
    silently loading the wrong weights.
    """
    import glob
    import os

    from hutash_inference.errors import ModelLoadError

    cache = os.environ.get("HF_HUB_CACHE", "/weights")
    revision = os.environ.get("HUTASH_HF_REVISION")
    if not revision:
        raise ModelLoadError(
            f"resolve_local_weights_dir({model_id!r}): HUTASH_HF_REVISION is "
            f"not set — cannot resolve the pinned weights snapshot. "
            f"(docker_manager injects it from the catalogue hf_revision.)"
        )
    matches = [
        m for m in glob.glob(os.path.join(cache, "models--*", "snapshots", revision))
        if os.path.isdir(m)
    ]
    if not matches:
        raise ModelLoadError(
            f"resolve_local_weights_dir({model_id!r}): pinned snapshot "
            f"{revision} not found under {cache} — the weights mount is "
            f"missing or not staged at this revision."
        )
    if len(matches) > 1:
        raise ModelLoadError(
            f"resolve_local_weights_dir({model_id!r}): multiple repos contain "
            f"snapshot {revision}: {matches}"
        )
    return matches[0]


class Inference:
    """Base class for all Aura model inference implementations.

    Subclass this in your inference.py and override load() to initialize
    your model. Use @capability decorator to mark methods as capability
    handlers.

    Example:
        class MyModelInference(Inference):
            def load(self):
                from my_library import MyModel
                self.model = MyModel.from_pretrained("...")

            @capability("my-capability")
            def do_something(self, text: str) -> dict:
                result = self.model.run(text)
                return {"output": result}
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize inference with model.yaml config.

        Args:
            config: Parsed model.yaml contents, provided by server
        """
        self.config = config or {}
        self.model_id = self.config.get("model_id", self.__class__.__name__)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_loaded = False
        self._offloaded = False  # True when moved off the GPU into RAM

    def load(self) -> None:
        """Default Tier-1 weights-out loader for standard HuggingFace models.

        Called once at container startup. Models whose weights load via the
        standard HF convention (``SomeClass.from_pretrained(dir)``) need NO
        custom load(): they declare an ``hf_load`` spec in their
        catalogue/manifest and the framework loads them here — offline, from
        the mounted pinned snapshot. Non-standard models (chatterbox,
        kokoro) override load() instead (Tier-2).

        ``hf_load`` spec::

            {"library": "transformers",                  # default
             "auto_class": "AutoModelForSpeechSeq2Seq",  # required
             "processor_class": "AutoProcessor",         # optional
             "tokenizer_class": "AutoTokenizer"}         # optional

            {"library": "diffusers",
             "pipeline_class": "DiffusionPipeline"}       # required

        Everything loads with ``local_files_only=True`` from the mounted
        snapshot — never the network, never a guessed class. Results land on
        ``self.model`` (+ ``self.processor`` / ``self.tokenizer`` when
        declared).
        """
        from hutash_inference.errors import ModelLoadError

        spec = self.config.get("hf_load")
        if not spec:
            raise ModelLoadError(
                f"{self.model_id!r} has no load() override and no 'hf_load' "
                f"spec — a standard HuggingFace model needs an hf_load spec "
                f"(Tier-1) in model.meta.yaml (flows into manifest.json); a "
                f"non-standard model needs a custom load() override (Tier-2)."
            )

        weights_dir = resolve_local_weights_dir(self.model_id)
        library = spec.get("library", "transformers")
        self.logger.info(
            "Tier-1 load: %s via %s from %s", self.model_id, library, weights_dir
        )

        # Level-2 layer offload: when the engine flags HUTASH_CPU_OFFLOAD=1 the
        # model is too big for VRAM, so load it with accelerate's
        # device_map="auto", which splits layers across VRAM + RAM automatically
        # (accelerate ships as a transformers/diffusers dependency, so this needs
        # no extra install and works on NVIDIA/AMD/Apple/CPU alike). A model that
        # fits VRAM loads normally and uses whole-model offload() instead.
        import os as _os

        layer_offload = _os.environ.get("HUTASH_CPU_OFFLOAD", "0") == "1"

        if library == "transformers":
            import transformers

            auto_class = spec.get("auto_class")
            if not auto_class:
                raise ModelLoadError(
                    f"{self.model_id!r} hf_load.library='transformers' requires "
                    f"'auto_class' (e.g. AutoModelForSpeechSeq2Seq)."
                )
            if layer_offload:
                self.model = self._hf_from_pretrained(
                    transformers, auto_class, weights_dir, device_map="auto"
                )
                self.logger.info("Level-2 layer offload: %s via accelerate device_map=auto", self.model_id)
            else:
                self.model = self._hf_from_pretrained(transformers, auto_class, weights_dir)
            if spec.get("processor_class"):
                self.processor = self._hf_from_pretrained(
                    transformers, spec["processor_class"], weights_dir
                )
            if spec.get("tokenizer_class"):
                self.tokenizer = self._hf_from_pretrained(
                    transformers, spec["tokenizer_class"], weights_dir
                )
        elif library == "diffusers":
            import diffusers

            pipeline_class = spec.get("pipeline_class")
            if not pipeline_class:
                raise ModelLoadError(
                    f"{self.model_id!r} hf_load.library='diffusers' requires "
                    f"'pipeline_class' (e.g. DiffusionPipeline)."
                )
            self.model = self._hf_from_pretrained(
                diffusers, pipeline_class, weights_dir
            )
            if layer_offload and hasattr(self.model, "enable_model_cpu_offload"):
                # diffusers' accelerate-backed equivalent of device_map="auto".
                self.model.enable_model_cpu_offload()
                self.logger.info("Level-2 layer offload: %s via enable_model_cpu_offload", self.model_id)
        else:
            raise ModelLoadError(
                f"{self.model_id!r} hf_load.library={library!r} is not supported "
                f"(use 'transformers' or 'diffusers')."
            )

    def _hf_from_pretrained(self, module, class_name: str, weights_dir: str, **kwargs):
        """``getattr(module, class_name).from_pretrained(dir, local_files_only=True, **kwargs)``.

        Offline by construction; raises a clear error if the class name is
        not found in the library (never guesses a default). Extra kwargs (e.g.
        ``device_map="auto"`` for accelerate layer offload) pass straight
        through to from_pretrained.
        """
        from hutash_inference.errors import ModelLoadError

        cls = getattr(module, class_name, None)
        if cls is None:
            raise ModelLoadError(
                f"{self.model_id!r}: {module.__name__}.{class_name} not found "
                f"— check the hf_load spec's class name."
            )
        return cls.from_pretrained(weights_dir, local_files_only=True, **kwargs)

    def unload(self) -> None:
        """Called on graceful shutdown.

        Override for cleanup. Base implementation does nothing.
        """
        pass

    def health_check(self) -> dict:
        """Return container health status.

        Override for custom health logic. Default returns 'ok' if
        load() has been called successfully.
        """
        return {
            "status": "ok" if self._is_loaded else "not_ready",
            "loaded": self._is_loaded,
            # An offloaded model is still healthy — the process is alive with
            # weights resident in RAM, ready to reload to the GPU on demand.
            "offloaded": self._offloaded,
            "device": "cpu" if self._offloaded else self.target_device(),
        }

    def mark_loaded(self) -> None:
        """Called by server after load() completes successfully."""
        self._is_loaded = True

    # ---- VRAM offload / reload (fast model switching) --------------------
    #
    # The engine keeps a model's PROCESS alive but moves it OFF the GPU (into
    # system RAM) to free VRAM for another model, then moves it BACK when the
    # model is next used — seconds, versus minutes to kill the process and
    # reload weights from disk. offload()/reload() are overridable; the default
    # handles any model whose torch Modules are stored as attributes on ``self``
    # (or one/two levels down, e.g. ChatterboxTTS's ``.t3`` / ``.s3gen``).

    def offload(self) -> dict:
        """Move the model off the GPU into RAM, freeing VRAM. The process stays
        alive with weights resident in RAM, so reload() is fast.

        Override for a model whose weights the default cannot reach.
        """
        if self._offloaded:
            return {"status": "offloaded", "device": "cpu", "moved": 0}
        moved = 0
        for module in self._offloadable_modules():
            try:
                module.to("cpu")
                moved += 1
            except Exception as e:  # noqa: BLE001 — best effort per module
                self.logger.warning("offload: could not move %s: %s", type(module).__name__, e)
        self._set_device_holders("cpu")
        self._empty_gpu_cache()
        self._offloaded = True
        self.logger.info("offloaded %s to RAM (%d module(s))", self.model_id, moved)
        return {"status": "offloaded", "device": "cpu", "moved": moved}

    def reload(self) -> dict:
        """Move the model back onto its target device (GPU). Fast — the weights
        are already resident in RAM. Override alongside offload() when custom.
        """
        target = self.target_device()
        moved = 0
        for module in self._offloadable_modules():
            try:
                module.to(target)
                moved += 1
            except Exception as e:  # noqa: BLE001 — best effort per module
                self.logger.warning("reload: could not move %s: %s", type(module).__name__, e)
        self._set_device_holders(target)
        self._offloaded = False
        self.logger.info("reloaded %s to %s (%d module(s))", self.model_id, target, moved)
        return {"status": "reloaded", "device": target, "moved": moved}

    def is_offloaded(self) -> bool:
        return self._offloaded

    def target_device(self) -> str:
        """The concrete device this model runs on when active, resolved from
        HUTASH_DEVICE (cuda | mps | cpu | auto) injected by the engine.
        """
        import os as _os

        want = _os.environ.get("HUTASH_DEVICE", "cuda")
        try:
            import torch

            has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
            if want in ("auto", ""):
                return "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
            if want == "cuda" and not torch.cuda.is_available():
                return "mps" if has_mps else "cpu"
            if want == "mps" and not has_mps:
                return "cpu"
            return want
        except Exception:
            return want or "cpu"

    def _offloadable_modules(self) -> list:
        """torch.nn.Module instances reachable from self's attributes, up to a
        few levels deep. A Module moves its whole parameter tree with one .to(),
        so descent stops at the first Module found on a branch.
        """
        try:
            import torch
        except Exception:
            return []
        found: list = []
        seen: set[int] = set()

        def walk(obj, depth):
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                found.append(obj)
                return  # .to() moves the whole submodule tree
            if depth <= 0:
                return
            d = getattr(obj, "__dict__", None)
            if not isinstance(d, dict):
                return
            for name, val in list(d.items()):
                if name.startswith("_"):
                    continue
                walk(val, depth - 1)

        walk(self, 3)
        return found

    def _device_holders(self) -> list:
        """Objects (self's model attributes, one level deep) that carry a
        writable ``.device`` attribute the model's own code reads to place
        tensors — updated on offload/reload so post-switch generations target
        the right device (e.g. ChatterboxTTS.device).
        """
        holders: list = []
        seen: set[int] = set()
        d = getattr(self, "__dict__", {})
        for name, val in list(d.items()):
            if name.startswith("_") or val is None or id(val) in seen:
                continue
            seen.add(id(val))
            if hasattr(val, "device"):
                holders.append(val)
        return holders

    def _set_device_holders(self, device: str) -> None:
        try:
            import torch

            dev = torch.device(device)
        except Exception:
            dev = device
        for h in self._device_holders():
            try:
                cur = getattr(h, "device", None)
                setattr(h, "device", str(dev) if isinstance(cur, str) else dev)
            except Exception:  # noqa: BLE001 — read-only .device property, skip
                pass

    def _empty_gpu_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available() and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            pass

    # ---- Helper methods for common I/O conversions ----

    def _save_temp_audio(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """Save audio bytes to a temporary file, return path.

        Useful for libraries that require file paths rather than bytes.
        Caller is responsible for cleanup.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name

    def _audio_to_wav_bytes(self, audio, sample_rate: int, channels: int = 1) -> bytes:
        """Convert audio array to WAV bytes.

        Supports numpy arrays, torch tensors, lists.
        """
        import soundfile as sf
        import numpy as np

        try:
            import torch
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
        except ImportError:
            pass

        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        if channels == 1 and audio.ndim > 1:
            audio = audio.squeeze()

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()

    def _image_to_png_bytes(self, image) -> bytes:
        """Convert image to PNG bytes.

        Supports PIL Images, numpy arrays, torch tensors.
        """
        from PIL import Image
        import numpy as np

        if isinstance(image, Image.Image):
            pil_image = image
        else:
            try:
                import torch
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
            except ImportError:
                pass

            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            pil_image = Image.fromarray(image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()


def get_capabilities(instance: Inference) -> dict[str, Callable]:
    """Discover @capability methods on an Inference instance.

    Returns dict mapping capability_id to the bound method.
    """
    capabilities: dict[str, Callable] = {}
    for attr_name in dir(instance):
        if attr_name.startswith("_"):
            continue
        attr = getattr(instance, attr_name)
        if callable(attr) and hasattr(attr, "_aura_capability"):
            capabilities[attr._aura_capability] = attr
    return capabilities
