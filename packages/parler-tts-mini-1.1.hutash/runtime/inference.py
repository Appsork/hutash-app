"""Parler-TTS Mini 1.1 inference for Aura.

Source file â€” hand-written, not generated.
Implements the hutash_inference contract for capability: tts

Parler-TTS steers the speaker with a free-text natural-language
description (the `voice_description` control) rather than a fixed
preset list. Boilerplate (FastAPI, routing, HTTP) is provided by the
hutash_inference library â€” this file contains only model-specific code.
"""

import os

import torch

from hutash_inference import Inference, capability, resolve_local_weights_dir
from hutash_inference.errors import GenerationError

# Hugging Face repo for Parler-TTS Mini v1.1. Used only in load().
_HF_REPO = "parler-tts/parler-tts-mini-v1.1"

# Mirrors the model.meta.yaml control default so a missing
# voice_description still produces sensible audio.
_DEFAULT_VOICE = (
    "A warm female voice, slightly expressive, moderate pace, "
    "clear studio quality."
)


class ParlerTTSMiniInference(Inference):
    """Parler-TTS Mini 1.1 text-to-speech."""

    def load(self) -> None:
        """Load model + tokenizers at container startup.

        Weights-out: load from the mounted snapshot (the exact pinned commit)
        rather than the _HF_REPO id, which would hit HuggingFace offline. The
        description_tokenizer's separate repo (google/flan-t5-large's tokenizer)
        is baked into the image at /app/flan-t5-tokenizer — see below.
        (NEEDS GPU RUNTIME TEST.)
        """
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(
            "Loading Parler-TTS Mini 1.1 on %sâ€¦", self.device,
        )

        weights_dir = resolve_local_weights_dir(self.model_id)
        # Load at the checkpoint's native precision (float32). Under the
        # on-demand model lifecycle (Ollama-style), only one GPU model is
        # resident at a time — Studio stops other GPU containers before starting
        # this one — so Parler-TTS Mini's ~10 GB load peak fits a 12 GB card
        # without the float16 cast. Keeping fp32 avoids any half-precision
        # quality/compat caveats.
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            weights_dir,
        ).to(self.device)
        # Prompt tokenizer = the text to be spoken.
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        # Description tokenizer = the natural-language voice style. Parler v1.1
        # uses the text encoder's own tokenizer (google/flan-t5-large) — a
        # SEPARATE HF repo. The text-encoder WEIGHTS are already inside this
        # model's safetensors; only the tokenizer is needed, so it is baked
        # bundled alongside this model at <model_dir>/flan-t5-tokenizer and
        # loaded offline rather than from
        # `self.model.config.text_encoder._name_or_path`, which would hit HF.
        # The model dir is /app inside a container image and the package's run
        # dir in a native venv — HUTASH_MODEL_DIR points at whichever applies.
        model_dir = os.environ.get("HUTASH_MODEL_DIR", "/app")
        tokenizer_path = os.path.join(model_dir, "flan-t5-tokenizer")
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True,
        )
        self.sample_rate = self.model.config.sampling_rate
        self.logger.info(
            "Parler-TTS Mini 1.1 loaded (sample_rate=%d)",
            self.sample_rate,
        )

    @capability("tts")
    def tts(
        self,
        prompt: str,
        voice_description: str = _DEFAULT_VOICE,
        **kwargs,
    ) -> dict:
        """Synthesize speech from `prompt` in the described voice.

        Args match capabilities.tts.inputs + .controls in
        model.meta.yaml; return-dict keys match .outputs (audio).
        """
        if not prompt or not prompt.strip():
            raise GenerationError("prompt is required and cannot be empty")

        description = (voice_description or _DEFAULT_VOICE).strip()
        self.logger.info(
            "Generating TTS: prompt_len=%d, description_len=%d",
            len(prompt), len(description),
        )

        try:
            input_ids = self.description_tokenizer(
                description, return_tensors="pt",
            ).input_ids.to(self.device)
            prompt_input_ids = self.tokenizer(
                prompt, return_tensors="pt",
            ).input_ids.to(self.device)

            with torch.inference_mode():
                generation = self.model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                )

            audio = generation.cpu().numpy().squeeze()
        except Exception as e:  # noqa: BLE001 â€” surfaced as GenerationError
            raise GenerationError(
                f"Parler-TTS generation failed: {e}",
            ) from e

        if audio is None or getattr(audio, "size", 0) == 0:
            raise GenerationError("No audio generated")

        wav_bytes = self._audio_to_wav_bytes(
            audio, sample_rate=self.sample_rate,
        )
        return {"audio": wav_bytes}
