"""Whisper Tiny inference via faster-whisper (CTranslate2, CPU/INT8).

Source file â€” hand-written, not generated.
Implements the hutash_inference contract for capability: stt.

Internal infrastructure â€” invoked by
api/services/transcription/transcriber.py when a user-facing model
declares `auto_fill: {via: stt}` on an input. Returns just the
transcript string under the `transcript` key (no segments, no word
timestamps â€” those belong to the user-facing whisper-large-v3 model).
"""

import os

from hutash_inference import Inference, capability, resolve_local_weights_dir


def _ct2_device_compute(default: str = "auto") -> tuple:
    """Map HUTASH_DEVICE (cuda|cuda:N|auto|cpu|mps) to faster-whisper's
    CTranslate2 (device, compute_type). CTranslate2 has no MPS backend, so
    mps -> cpu/int8; "auto" lets CTranslate2 pick cuda/float16 or cpu/int8 from
    the libs present. Any explicit "cuda" flavour (incl. "cuda:N") -> cuda."""
    want = os.environ.get("HUTASH_DEVICE", default)
    if want.startswith("cuda"):
        return ("cuda", "float16")
    return {
        "cpu": ("cpu", "int8"),
        "auto": ("auto", "auto"),
        "mps": ("cpu", "int8"),
    }.get(want, ("auto", "auto"))


class WhisperTinyInference(Inference):
    """Whisper Tiny CPU-only STT (~40 MB INT8). Internal system component."""

    def load(self) -> None:
        from faster_whisper import WhisperModel

        # CPU + int8 â€” this image is CPU-only and the tiny model is
        # small enough that quantisation has no audible quality cost
        # for the short clips this component sees (5â€“30 s voice refs).
        # Weights-out: load from the mounted CT2 snapshot (the exact pinned
        # commit) rather than the "Systran/faster-whisper-tiny" repo id, which
        # would hit HuggingFace and fail offline. resolve_local_weights_dir
        # returns /weights/models--*/snapshots/<hf_revision>/, whose root holds
        # the CT2 files faster-whisper expects (model.bin, config.json,
        # tokenizer.json, vocabulary.txt).
        # Internal CPU component: HUTASH_DEVICE defaults to cpu/int8, but the
        # engine can still select cuda/float16 via HUTASH_DEVICE if desired.
        device, compute_type = _ct2_device_compute(default="cpu")
        self.model = WhisperModel(
            resolve_local_weights_dir(self.model_id),
            device=device,
            compute_type=compute_type,
        )

    @capability("stt")
    def transcribe(
        self,
        file: bytes,
        language: str = "auto",
    ) -> dict:
        """Transcribe an audio file into a plain string.

        Args match model.meta.yaml capabilities.stt.inputs.

        `language` is an ISO-639-1 code passed through to faster-whisper.
        The sentinel `"auto"` (plus the empty string and `"Auto-detect"`
        label) map to `None`, which triggers automatic language detection.
        """
        language_code = (
            None
            if language in ("auto", "Auto-detect", "", None)
            else language
        )
        temp_path = self._save_temp_audio(file)

        try:
            segments_gen, _info = self.model.transcribe(
                temp_path,
                beam_size=1,
                vad_filter=True,
                language=language_code,
            )
            text = " ".join(s.text.strip() for s in segments_gen).strip()
            return {"transcript": text}
        finally:
            import os

            try:
                os.unlink(temp_path)
            except OSError:
                pass
