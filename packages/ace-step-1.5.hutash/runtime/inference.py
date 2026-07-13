"""ACE-Step v1 3.5B music generation via acestep.pipeline_ace_step.

Capability `music` at endpoint /generate â€” wire-compatible with the
pre-SSOT server so api/routers/generation.py's /generate/music route
continues to work via the generic handler.

Module-level torchaudio monkey-patches: torchaudio 2.11+ removed its
multi-backend dispatch and routes both save and load through
torchcodec, which requires FFmpeg shared libs not in the container.
Replace save with soundfile.write and load with soundfile.read BEFORE
ACE-Step or the Inference class touches torchaudio.
"""

import os

import torch
import torchaudio
import soundfile as sf

_original_torchaudio_save = torchaudio.save
_original_torchaudio_load = torchaudio.load


def _soundfile_save(filepath, src, sample_rate, **_kw):
    src_np = src.cpu().numpy()
    if src_np.ndim == 2:
        src_np = src_np.T  # soundfile expects (samples, channels)
    sf.write(str(filepath), src_np, sample_rate)


def _soundfile_load(filepath, *_args, **_kw):
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    # soundfile â†’ (samples, channels); torchaudio â†’ (channels, samples)
    return torch.from_numpy(data.T).contiguous(), sr


torchaudio.save = _soundfile_save
torchaudio.load = _soundfile_load

from hutash_inference import (  # noqa: E402
    Inference,
    capability,
    resolve_local_weights_dir,
)
from hutash_inference.errors import GenerationError  # noqa: E402


class ACEStepInference(Inference):
    """ACE-Step music generator â€” text prompt + optional lyrics."""

    def load(self) -> None:
        from acestep.pipeline_ace_step import ACEStepPipeline

        # Weights-out: the multi-component checkpoint (ace_step_transformer/,
        # umt5-base/, music_dcae_f8c8/, music_vocoder/) is no longer baked at
        # /app/models/ace-step. resolve_local_weights_dir returns the mounted
        # pinned snapshot, whose nested layout matches what ACEStepPipeline
        # expects. An explicit ACE_STEP_CHECKPOINT override still wins.
        # UNTESTED: ACEStepPipeline may try to fetch any missing component from
        # HuggingFace at load — under HF_HUB_OFFLINE=1 that would fail. Verify
        # offline on a GPU box before release.
        checkpoint = os.environ.get("ACE_STEP_CHECKPOINT") or resolve_local_weights_dir(
            self.model_id
        )
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint,
            device_id=0 if torch.cuda.is_available() else -1,
        )

    @capability("music")
    def text_to_music(
        self,
        prompt: str,
        duration: float = 30.0,
        lyrics: str = "",
        inference_steps: int = 8,
        seed: int = -1,
        instrumental: bool = False,
        bpm: int | None = None,
        keyscale: str = "",
        infer_method: str = "ode",
        reference_audio: bytes = b"",
        audio_cover_strength: float = 0.5,
    ) -> dict:
        """Generate a music clip from a prompt.

        `seed != -1` pins the diffusion RNG via `manual_seeds`.
        `instrumental=True` forces empty lyrics regardless of the
        `lyrics` argument. `reference_audio` (WAV bytes â€” the
        backend's audio_converter already normalises uploads to WAV
        before they reach this container) routes to the pipeline's
        `ref_audio_input` with `audio2audio_enable=True` and
        `ref_audio_strength` from `audio_cover_strength`.

        `bpm` + `keyscale` are appended to the caption text â€” the
        ACE-Step pipeline has no dedicated kwargs for them, but the
        upstream LM reads tempo and key cues from caption text. The
        XL variant exposes these as real parameters via
        `GenerationParams` (Batch 4).

        `infer_method` maps the YAML's `ode`/`sde` option vocabulary
        onto the pipeline's `scheduler_type` argument (`euler` /
        `sde`).
        """
        if not prompt.strip():
            raise GenerationError("Prompt is required")

        # Map infer_method to pipeline's scheduler_type vocabulary
        SCHEDULER_MAP = {"ode": "euler", "sde": "sde"}
        scheduler_type = SCHEDULER_MAP.get(infer_method, "euler")

        # Inject bpm/keyscale into prompt
        # (pipeline has no dedicated params â€” LM reads these from caption)
        enriched_prompt = prompt
        meta_hints: list[str] = []
        if bpm:
            meta_hints.append(f"{int(bpm)} BPM")
        if keyscale and keyscale.strip().lower() not in ("", "auto"):
            meta_hints.append(keyscale.strip())
        if meta_hints:
            enriched_prompt = prompt.rstrip(", ") + ", " + ", ".join(meta_hints)

        effective_lyrics = "" if instrumental else (lyrics or "")

        ref_audio_path: str | None = None
        if reference_audio:
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(reference_audio)
            tmp.close()
            ref_audio_path = tmp.name

        try:
            result = self.pipeline(
                prompt=enriched_prompt,
                lyrics=effective_lyrics,
                audio_duration=float(duration),
                infer_step=int(inference_steps),
                manual_seeds=[int(seed)] if seed != -1 else None,
                scheduler_type=scheduler_type,
                ref_audio_input=ref_audio_path,
                audio2audio_enable=bool(ref_audio_path),
                ref_audio_strength=float(audio_cover_strength),
            )

            # Pipeline returns a list: [wav_path, ..., params_json_str].
            # Find the first .wav path; read its bytes.
            wav_path = None
            if isinstance(result, (list, tuple)):
                for item in result:
                    if isinstance(item, str) and item.endswith(".wav"):
                        wav_path = item
                        break
            if not wav_path:
                raise GenerationError("ACE-Step pipeline returned no audio file")

            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
            return {"audio": wav_bytes}
        finally:
            if ref_audio_path:
                from pathlib import Path

                Path(ref_audio_path).unlink(missing_ok=True)
