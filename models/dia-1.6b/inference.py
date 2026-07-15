"""Dia 1.6B dialogue TTS for Aura.

Source file â€” hand-written, not generated.
Implements the hutash_inference contract for capability: voice-clone.

Backend: HuggingFace transformers (DiaForConditionalGeneration +
AutoProcessor). Dia generates multi-speaker dialogue from a [S1]/[S2]
script in a single pass and can clone a voice from a short reference
clip. Upstream is GPU-only (CPU support pending), but the device is
detected via torch.cuda.is_available() so the same code path runs
locally without a GPU (slowly, unsupported).
"""

import io
import os

import soundfile as sf
import torch

from hutash_inference import Inference, capability, resolve_local_weights_dir
from hutash_inference.errors import GenerationError

_MODEL_ID = "nari-labs/Dia-1.6B-0626"
_SAMPLE_RATE = 44100  # Dia / Descript Audio Codec native rate (44.1 kHz mono)
_DAC_REPO = "descript/dac_44khz"


def _resolve_dac_dir() -> str:
    """Return a local directory holding the DAC codec, resolved fully offline.

    Resolution order:
      1. A legacy baked ``/app/dac`` directory, if the image carries one.
      2. The companion repo staged into the HF cache by the weights downloader
         (``extra_hf_repos``) — found by globbing
         ``$HF_HUB_CACHE/models--descript--dac_44khz/snapshots/<commit>/``.
         Loading the snapshot DIR (not the repo id) avoids the missing
         ``refs/main`` pointer that a commit-pinned download leaves out, which
         would otherwise make a by-id load reach the network and fail offline.
      3. Fall back to the bare repo id (online path / dev box).
    """
    import glob
    import os

    if os.path.isdir("/app/dac"):
        return "/app/dac"
    cache = os.environ.get("HF_HUB_CACHE") or "/weights"
    repo_cache = os.path.join(cache, "models--descript--dac_44khz", "snapshots")
    snaps = sorted(glob.glob(os.path.join(repo_cache, "*")))
    snaps = [s for s in snaps if os.path.isdir(s)]
    if snaps:
        return snaps[0]
    return _DAC_REPO


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



class Dia16BInference(Inference):
    """Dia 1.6B multi-speaker dialogue TTS."""

    def load(self) -> None:
        from transformers import (
            AutoFeatureExtractor,
            AutoTokenizer,
            DacModel,
            DiaForConditionalGeneration,
            DiaProcessor,
        )

        self.device = _resolve_device()  # HUTASH_DEVICE (cuda|auto|cpu|mps)
        self.logger.info("Loading Dia 1.6B on %sâ€¦", self.device)

        # fp16 on GPU (~4.4 GB VRAM); fp32 on CPU fallback. Matches the
        # original Nari Labs reference (compute_dtype="float16") and the
        # known-working :1.0.12 state; this is what every published Dia
        # deployment we've cross-checked uses (Koyeb's hosted Dia ships
        # cuda=float16 + cpu=float32 too). Earlier this session we tried
        # bfloat16 (:1.0.17) and torch_dtype="auto" (:1.0.18) â€” both
        # produced lower-quality output than this configuration.
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # Weights-out: load from the mounted snapshot (the exact pinned commit)
        # rather than the _MODEL_ID repo, which would hit HuggingFace offline.
        weights_dir = resolve_local_weights_dir(self.model_id)

        # The Dia processor needs the DAC audio codec (descript/dac_44khz) — a
        # SEPARATE HF repo. AutoProcessor.from_pretrained ALWAYS rebuilds the
        # audio_tokenizer from audio_tokenizer_config (fetching that repo, and
        # ignoring any injected instance), which fails under HF_HUB_OFFLINE=1.
        # The codec's 300 MB safetensors is too large to bake into the image or
        # commit to the catalogue, so it is staged into the mounted weights cache
        # as a companion repo (weights_downloader `extra_hf_repos`). It is fetched
        # pinned to a commit, so the HF cache has a snapshots/<commit>/ dir but no
        # refs/main pointer — meaning a load BY REPO ID would still try the network
        # to resolve "main" and fail offline. So resolve the snapshot DIRECTORY
        # directly from the cache and load from that path. A legacy baked /app/dac
        # is honoured first. Build the DiaProcessor explicitly from the local
        # sub-components + the local DAC so nothing hits the network.
        dac_src = _resolve_dac_dir()
        feature_extractor = AutoFeatureExtractor.from_pretrained(weights_dir)
        text_tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        dac = DacModel.from_pretrained(dac_src, local_files_only=True)
        self.processor = DiaProcessor(
            feature_extractor, text_tokenizer, audio_tokenizer=dac
        )
        # HUTASH_CPU_OFFLOAD=1 → device_map="auto" offloads across CPU/GPU;
        # otherwise load fully onto the resolved device.
        if _cpu_offload():
            self.model = DiaForConditionalGeneration.from_pretrained(
                weights_dir, torch_dtype=dtype, device_map="auto"
            )
        else:
            self.model = DiaForConditionalGeneration.from_pretrained(
                weights_dir, torch_dtype=dtype
            ).to(self.device)

    @capability("voice-clone")
    def tts(
        self,
        prompt: str,
        voice_ref: bytes | None = None,
        voice_ref_transcript: str | None = None,
        seed: int = -1,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        max_tokens: int = 3072,
        **kwargs,
    ) -> dict:
        """Generate multi-speaker dialogue audio from a [S1]/[S2] script.

        Args mirror the inputs + controls in model.meta.yaml:
          prompt                â€” dialogue script with [S1]/[S2] tags + nonverbals
          voice_ref             â€” optional reference clip for in-context voice
                                  cloning (bytes; ~5â€“10 s recommended)
          voice_ref_transcript  â€” exact transcript of voice_ref; required for
                                  cloning to work (else the audio is ignored)
          seed                  â€” fix (!= -1) to reproduce a voice; the
                                  control was named `voice_seed` originally
                                  but renamed to the standard `seed` so the
                                  batch-generation UI (which gates on
                                  `c.name === "seed"`) picks it up
          cfg_scale    â€” classifier-free guidance scale
          temperature  â€” sampling temperature
          max_tokens   â€” max new audio tokens (~86 tokens â‰ˆ 1 s)

        Returns {"audio": <wav bytes>} matching the declared output.
        """
        if not prompt.strip():
            raise GenerationError("Dialogue script is required")

        if seed != -1:
            torch.manual_seed(int(seed))

        # Voice cloning needs the reference audio's transcript prepended to
        # the script (blank-line separated), or Dia echoes the reference
        # instead of using it as voice conditioning. A reference clip given
        # without a transcript is dropped â€” run text-only and warn.
        if voice_ref and voice_ref_transcript:
            full_prompt = f"{voice_ref_transcript.strip()}\n\n{prompt.strip()}"
        elif voice_ref and not voice_ref_transcript:
            self.logger.warning(
                "voice_ref provided without transcript â€” ignoring audio "
                "conditioning"
            )
            voice_ref = None
            full_prompt = prompt.strip()
        else:
            full_prompt = prompt.strip()

        try:
            proc_kwargs = {
                "text": [full_prompt],
                "padding": True,
                "return_tensors": "pt",
            }

            # In-context voice cloning: decode the reference clip to a mono
            # waveform and pass it as the audio prompt. The processor
            # keyword ("audio") should be confirmed against the pinned
            # transformers version at first build.
            if voice_ref:
                ref_audio, ref_sr = sf.read(io.BytesIO(voice_ref), dtype="float32")
                if ref_audio.ndim > 1:
                    ref_audio = ref_audio.mean(axis=1)
                # DiaProcessor requires audio at exactly _SAMPLE_RATE
                # (44.1 kHz). Reference clips uploaded by users come at
                # whatever the source recorded â€” 16 / 22.05 / 48 kHz are
                # all common. Without this resample the processor either
                # raises or interprets the clip at the wrong pitch, and
                # the voice clone fails silently.
                if ref_sr != _SAMPLE_RATE:
                    import scipy.signal as sps

                    num_samples = int(len(ref_audio) * _SAMPLE_RATE / ref_sr)
                    ref_audio = sps.resample(ref_audio, num_samples)
                    ref_sr = _SAMPLE_RATE
                proc_kwargs["audio"] = [ref_audio]

            inputs = self.processor(**proc_kwargs).to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_tokens),
                    guidance_scale=float(cfg_scale),
                    temperature=float(temperature),
                    top_p=0.95,
                )

            decoded = self.processor.batch_decode(outputs)
            audio = decoded[0] if isinstance(decoded, (list, tuple)) else decoded

            # When voice cloning, Dia echoes the reference clip at the
            # start of the output. Trim it so only the script plays.
            # ref_audio is already at _SAMPLE_RATE after the resample block,
            # so len(ref_audio) gives the correct number of output samples to trim.
            if voice_ref:
                trim_samples = len(ref_audio)
                # Safety guard â€” never trim more than 50% of output
                if trim_samples and len(audio) > trim_samples * 2:
                    audio = audio[trim_samples:]

            return {
                "audio": self._audio_to_wav_bytes(audio, _SAMPLE_RATE, channels=1)
            }
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(str(e)) from e
