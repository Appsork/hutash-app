"""Whisper Tiny inference via faster-whisper (CTranslate2, CPU/INT8).

Source file â€” hand-written, not generated.
Implements the hutash_inference contract for capability: stt.

Internal infrastructure â€” invoked by
api/services/transcription/transcriber.py when a user-facing model
declares `auto_fill: {via: stt}` on an input. Returns just the
transcript string under the `transcript` key (no segments, no word
timestamps â€” those belong to the user-facing whisper-large-v3 model).
"""

from hutash_inference import Inference, capability, resolve_local_weights_dir


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
        self.model = WhisperModel(
            resolve_local_weights_dir(self.model_id),
            device="cpu",
            compute_type="int8",
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
