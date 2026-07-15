"""Kokoro TTS inference.

Implements capabilities declared in model.yaml via the hutash_inference
base class. Boilerplate (FastAPI, routing, HTTP handling) is provided
by the hutash_inference library â€” this file contains only model-specific
code.
"""

import os

import numpy as np

from hutash_inference import Inference, capability, resolve_local_weights_dir
from hutash_inference.errors import GenerationError


def _resolve_device(default: str = "cuda") -> str:
    """Resolve HUTASH_DEVICE (cuda | cuda:N | mps | cpu | auto) to a device.

    The engine is the source of truth: an explicit device (cuda, cuda:N, mps,
    cpu) is returned verbatim without re-checking torch.cuda.is_available().
    Only "auto" resolves to the best available accelerator. Kokoro's default is
    "cpu" (it is a CPU model unless the engine moves it onto an accelerator)."""
    import torch

    want = os.environ.get("HUTASH_DEVICE", default)
    if want == "auto":
        has_mps = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        return "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    return want


class KokoroInference(Inference):
    """Kokoro text-to-speech."""

    def load(self):
        """Load Kokoro from the mounted weights snapshot (offline).

        Weights-out (Tier-2): rather than KPipeline(lang_code="a"), which
        pulls config.json + the .pth from HuggingFace on first call and
        downloads each voice lazily via hf_hub_download, we build a KModel
        from the local snapshot and hand it to KPipeline. With both config
        and model paths supplied, KModel makes zero network calls.
        Voices are resolved to local .pt paths at generation time (see
        text_to_speech) so the lazy per-voice load also stays offline.
        """
        from kokoro import KModel, KPipeline

        weights_dir = resolve_local_weights_dir(self.config.get("model_id", "kokoro"))
        self.voices_dir = os.path.join(weights_dir, "voices")
        config_path = os.path.join(weights_dir, "config.json")
        model_path = os.path.join(weights_dir, "kokoro-v1_0.pth")
        self.logger.info("Loading Kokoro from local weights: %s", weights_dir)
        # Kokoro is a CPU model by default; HUTASH_DEVICE can still move it
        # onto an accelerator (cuda|mps) when the engine selects one.
        device = _resolve_device(default="cpu")
        kmodel = KModel(repo_id=None, config=config_path, model=model_path).to(device).eval()
        self.pipeline = KPipeline(lang_code="a", repo_id=None, model=kmodel)  # 'a' = American English
        self.logger.info("Kokoro pipeline loaded (offline)")

    @capability("tts")
    def text_to_speech(
        self,
        prompt: str,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> dict:
        """Generate speech audio from text.

        Parameters match model.yaml capabilities.tts.inputs and
        capabilities.tts.controls. Return dict keys match
        capabilities.tts.outputs.
        """
        if not prompt or not prompt.strip():
            raise GenerationError("prompt is required and cannot be empty")

        self.logger.info(
            "Generating TTS: voice=%s, speed=%s, prompt_len=%d",
            voice, speed, len(prompt),
        )

        # Resolve voice name(s) to local .pt paths so KPipeline's lazy
        # per-voice load reads from the mounted snapshot instead of
        # hf_hub_download (which fails under HF_HUB_OFFLINE). Comma-
        # separated names are averaged by KPipeline; map each one.
        voice_paths = []
        for name in voice.split(","):
            name = name.strip()
            path = os.path.join(self.voices_dir, f"{name}.pt")
            if not os.path.isfile(path):
                raise GenerationError(
                    f"Voice '{name}' not found in local weights ({self.voices_dir})"
                )
            voice_paths.append(path)
        voice_arg = ",".join(voice_paths)

        # KPipeline returns a generator of (text_chunk, phoneme_chunk, audio_chunk)
        generator = self.pipeline(prompt, voice=voice_arg, speed=speed)

        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            raise GenerationError("No audio generated")

        full_audio = np.concatenate(audio_chunks)
        wav_bytes = self._audio_to_wav_bytes(full_audio, sample_rate=24000)

        return {"audio": wav_bytes}
