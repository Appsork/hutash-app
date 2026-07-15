"""Chatterbox voice cloning + TTS via chatterbox-tts.

Two capabilities:
- voice-clone (primary) â€” upload reference audio, synthesize text
  in that voice. Endpoint /generate (wire-compatible with the
  pre-SSOT server and api/routers/generation.py's /generate/clone).
- tts â€” text-only, Chatterbox's default voice. Endpoint /tts.

chatterbox-tts==0.1.7 has a hard torch<=2.6.0 pin that causes pip to
downgrade off the shared-base's torch 2.11.0+cu128 during image build.
Runtime torch is 2.6.0+cu124 â€” see Dockerfile labels.
"""

import io
import os
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch

from hutash_inference import Inference, capability
from hutash_inference.errors import GenerationError


def _ensure_wav(audio_bytes: bytes) -> str:
    """Write `audio_bytes` to a temp WAV file, transcoding if needed.

    Two-stage decode:

      1. Try `soundfile.read` first. libsndfile (the underlying
         library) handles WAV / FLAC / OGG / AIFF natively, plus
         MP3 on libsndfile 1.1+. Re-encodes to WAV at the source
         sample-rate and returns the temp path.
      2. Fallback to FFmpeg subprocess for everything libsndfile
         doesn't claim â€” M4A, WebM, MP4, AAC, etc. Forces 22050
         Hz mono WAV output (Chatterbox's expected reference
         format).

    Returns the absolute path to a `.wav` file. Caller is
    responsible for unlinking it. Raises `GenerationError` with a
    user-readable message if both paths fail.
    """
    # --- Stage 1: soundfile fast path -----------------------------
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, data, sr)
        return tmp.name
    except Exception:
        pass

    # --- Stage 2: FFmpeg fallback ---------------------------------
    tmp_in = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp_in.write(audio_bytes)
        tmp_in.flush()
        tmp_in.close()
        tmp_out.close()
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", tmp_in.name,
                "-ar", "22050", "-ac", "1", "-f", "wav",
                tmp_out.name,
            ],
            check=True,
            capture_output=True,
        )
        return tmp_out.name
    except FileNotFoundError as e:
        os.unlink(tmp_out.name)
        raise GenerationError(
            "FFmpeg not available in container â€” non-WAV/FLAC/OGG "
            "uploads can't be transcoded. Reinstall the model image."
        ) from e
    except subprocess.CalledProcessError as e:
        os.unlink(tmp_out.name)
        stderr = e.stderr.decode("utf-8", errors="replace")[:200]
        raise GenerationError(
            f"FFmpeg could not decode the reference audio: {stderr}"
        ) from e
    finally:
        # `tmp_in` is always disposable; `tmp_out` is only kept on
        # success (we returned its path) and cleaned by the caller.
        try:
            os.unlink(tmp_in.name)
        except OSError:
            pass


def _resolve_device(default: str = "cuda") -> str:
    """Resolve HUTASH_DEVICE (cuda | cuda:N | mps | cpu | auto) to a device.

    The engine injects HUTASH_DEVICE from its detected compute mode and is the
    source of truth. An explicit device (cuda, cuda:N, mps, cpu) is returned
    verbatim — we do NOT re-check torch.cuda.is_available() to second-guess the
    engine. Only "auto" (the engine explicitly asking us to detect) resolves to
    the best available accelerator.
    """
    import torch

    want = os.environ.get("HUTASH_DEVICE", default)
    if want == "auto":
        has_mps = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        return "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    return want


def _cpu_offload() -> bool:
    """True when the engine asks accelerate to offload weights to CPU RAM
    (HUTASH_CPU_OFFLOAD=1), used with device_map="auto" model loads."""
    return os.environ.get("HUTASH_CPU_OFFLOAD", "0") == "1"



class ChatterboxInference(Inference):
    """Chatterbox voice-clone + TTS."""

    def load(self) -> None:
        from chatterbox.tts import ChatterboxTTS

        from hutash_inference import resolve_local_weights_dir

        device = _resolve_device()  # HUTASH_DEVICE (cuda|auto|cpu|mps)
        # Weights-out: load from the mounted weights snapshot (the exact
        # pinned commit) rather than ChatterboxTTS.from_pretrained, which
        # calls hf_hub_download and fails offline against the staged,
        # blob-less HF cache. resolve_local_weights_dir returns
        # /weights/models--*/snapshots/<hf_revision>/.
        ckpt_dir = resolve_local_weights_dir(self.config.get("model_id", "chatterbox"))
        self.model = ChatterboxTTS.from_local(ckpt_dir=ckpt_dir, device=device)

    def _generate_audio(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        max_new_tokens: int = 4096,
    ):
        """Synthesize speech, exposing `max_new_tokens` to callers.

        Reimplements `ChatterboxTTS.generate()` because upstream
        hardcodes `max_new_tokens=1000` with an explicit
        `# TODO: use the value in config` comment. Body is otherwise
        a faithful copy of upstream â€” same conditioning prep, same
        CFG batch duplication, same SOT/EOT padding, same drop-
        invalid-tokens post-processing, same watermark application.

        All upstream helpers (`punc_norm`, `drop_invalid_tokens`,
        `T3Cond`, `F` = `torch.nn.functional`) are re-exported from
        `chatterbox.tts` at module level, so we don't need to know
        the deeper submodule paths.
        """
        from chatterbox.tts import (
            F,
            T3Cond,
            drop_invalid_tokens,
            punc_norm,
        )

        model = self.model

        if audio_prompt_path:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            if model.conds is None:
                raise GenerationError(
                    "No reference voice prepared and no audio_prompt_path "
                    "given â€” Chatterbox needs at least one."
                )

        # Refresh exaggeration on the cached conds if it changed
        # since the last call (upstream does this so the slider takes
        # effect across consecutive generations without re-uploading).
        if exaggeration != model.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = model.conds.t3
            model.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=model.device)

        text = punc_norm(text)
        text_tokens = model.tokenizer.text_to_tokens(text).to(model.device)

        if cfg_weight > 0.0:
            # Two seqs in the batch â€” one CFG branch, one unconditional.
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = model.t3.hp.start_text_token
        eot = model.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = model.t3.inference(
                t3_cond=model.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                cfg_weight=float(cfg_weight),
                repetition_penalty=1.2,
                min_p=0.05,
                top_p=1.0,
            )
            # Conditional batch only.
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            # Token-id range guard from upstream â€” values >=6561 are
            # codebook-out-of-range and will crash s3gen.
            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = speech_tokens.to(model.device)

            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=model.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = model.watermarker.apply_watermark(
                wav, sample_rate=model.sr,
            )
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    @capability("voice-clone")
    def voice_clone(
        self,
        text: str,
        file: bytes,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        seed: int = -1,
        max_new_tokens: int = 4096,
    ) -> dict:
        """Clone the voice in `file` and speak `text`.

        `seed != -1` makes the synthesis reproducible by seeding
        torch's global RNG before the t3 transformer runs.
        `temperature` AND `max_new_tokens` are now both forwarded
        through the t3-stage inference call (we bypass
        `ChatterboxTTS.generate()` and call its internals directly
        â€” see `_generate_audio` for the body and the upstream-
        equivalence note).
        `language_id` stays declared as `implementation_status: stub`
        in the YAML â€” chatterbox-tts 0.1.7 has no multilingual
        synthesis hook.
        """
        if not text.strip():
            raise GenerationError("Text is required")
        if not file:
            raise GenerationError("Reference audio file is required")

        if seed != -1:
            torch.manual_seed(int(seed))

        # `_ensure_wav` handles every accepted format the YAML
        # declares (wav, mp3, m4a, flac, ogg, webm) â€” soundfile
        # reads the common ones natively, FFmpeg covers the rest.
        wav_path = _ensure_wav(file)
        try:
            wav = self._generate_audio(
                text,
                audio_prompt_path=wav_path,
                exaggeration=float(exaggeration),
                cfg_weight=float(cfg_weight),
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
            )
            return {"audio": self._wav_to_bytes(wav)}
        finally:
            Path(wav_path).unlink(missing_ok=True)

    @capability("tts")
    def text_to_speech(
        self,
        text: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        seed: int = -1,
        max_new_tokens: int = 4096,
    ) -> dict:
        """Speak `text` with Chatterbox's default voice (no reference).

        Same seed / temperature / max_new_tokens contract as
        `voice_clone`. The default voice's conditionals are loaded
        from `conds.pt` at `from_pretrained` time, so no
        `audio_prompt_path` is needed.
        """
        if not text.strip():
            raise GenerationError("Text is required")

        if seed != -1:
            torch.manual_seed(int(seed))

        wav = self._generate_audio(
            text,
            audio_prompt_path=None,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
        )
        return {"audio": self._wav_to_bytes(wav)}

    def _wav_to_bytes(self, wav) -> bytes:
        """Serialize a Chatterbox waveform tensor to WAV bytes.

        Uses soundfile (via the base helper) rather than torchaudio.save:
        torchaudio 2.11 routes save() through TorchCodec, which needs
        FFmpeg libraries that aren't in the container — without them
        save() raises and /tts and /clone return 500. soundfile writes
        the WAV directly with no codec dependency.
        """
        return self._audio_to_wav_bytes(wav, self.model.sr)
