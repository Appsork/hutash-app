"""Whisper large-v3 inference via faster-whisper (CTranslate2 backend).

Source file â€” hand-written, not generated.
Implements the hutash_inference contract for capability: stt.

No runtime device detection: docker_manager pulls the GPU or CPU image
(get_model_image(prefer_cpu=bool)), so the correct CUDA libs are either
present or not. faster-whisper's CTranslate2 backend with device="auto"
+ compute_type="auto" then uses whatever the image provides â€” CUDA in
the GPU image, CPU in the slim image.
"""

import os

from hutash_inference import Inference, capability, resolve_local_weights_dir


def _ct2_device_compute(default: str = "auto") -> tuple:
    """Map HUTASH_DEVICE (cuda|auto|cpu|mps) to faster-whisper's CTranslate2
    (device, compute_type). CTranslate2 has no MPS backend, so mps -> cpu/int8;
    "auto" lets CTranslate2 pick cuda/float16 or cpu/int8 from the libs present."""
    want = os.environ.get("HUTASH_DEVICE", default)
    return {
        "cuda": ("cuda", "float16"),
        "cpu": ("cpu", "int8"),
        "auto": ("auto", "auto"),
        "mps": ("cpu", "int8"),
    }.get(want, ("auto", "auto"))


class WhisperInference(Inference):
    """Whisper large-v3 speech-to-text (99 languages, translateâ†’English)."""

    def load(self) -> None:
        from faster_whisper import WhisperModel

        # Weights-out: load from the mounted CT2 snapshot (the exact pinned
        # commit) rather than the "Systran/faster-whisper-large-v3" repo id,
        # which would hit HuggingFace and fail offline. The snapshot root holds
        # the CT2 files faster-whisper expects (model.bin, config.json,
        # tokenizer.json, vocabulary.json, preprocessor_config.json).
        # device="auto" + compute_type="auto" resolve correctly in both the
        # GPU image (CUDA libs present â†’ cuda/float16) and the CPU image
        # (no CUDA â†’ cpu/int8). No torch, no manual detection.
        # HUTASH_DEVICE selects the CTranslate2 device + compute_type.
        device, compute_type = _ct2_device_compute(default="auto")
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
        task: str = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: str = "",
        beam_size: int = 5,
    ) -> dict:
        """Transcribe or translate an audio file.

        Args match model.meta.yaml capabilities.stt.{inputs, controls}.

        `language` is the ISO-639-1 code from the YAML's value/label
        split (`{value: "en", label: "English"}` etc.) â€” passed through
        to faster-whisper directly. The sentinel `"auto"` (and the
        legacy label-form `"Auto-detect"`, plus the empty string from a
        cleared field) all map to `None`, which triggers faster-whisper's
        automatic language detection.
        """
        language_code = (
            None
            if language in ("auto", "Auto-detect", "", None)
            else language
        )
        temp_path = self._save_temp_audio(file)

        try:
            segments_gen, info = self.model.transcribe(
                temp_path,
                beam_size=int(beam_size),
                vad_filter=True,
                language=language_code,
                task=task,
                word_timestamps=bool(word_timestamps),
                initial_prompt=initial_prompt or None,
            )

            segments: list[dict] = []
            full_text_parts: list[str] = []
            for segment in segments_gen:
                seg_dict = {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                }
                if word_timestamps and segment.words:
                    seg_dict["words"] = [
                        {
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "word": w.word,
                            "probability": round(w.probability, 3),
                        }
                        for w in segment.words
                    ]
                segments.append(seg_dict)
                full_text_parts.append(segment.text.strip())

            return {
                "text": " ".join(full_text_parts),
                "segments": segments,
                "language": info.language,
                "duration": round(info.duration, 2),
            }
        finally:
            import os

            try:
                os.unlink(temp_path)
            except OSError:
                pass
