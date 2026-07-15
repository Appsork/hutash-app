"""Prompt-engine: Qwen3 1.7B GGUF via llama-cpp-python.

Exposes one synthetic capability â€” "improve" â€” on the /generate
endpoint. Wire format kept identical to the pre-SSOT server so
api/services/prompts/improver.py works unchanged:
POST body {prompt, system, max_tokens, temperature} â†’ {"response": text}.
"""

import glob
import os
import re

from hutash_inference import Inference, capability, resolve_local_weights_dir


class PromptEngineInference(Inference):
    """Qwen3 1.7B prompt-improvement LLM (infrastructure, no UI)."""

    def load(self) -> None:
        from llama_cpp import Llama

        # Weights-out: the GGUF is no longer baked at /app/models. It is
        # downloaded at install time and mounted; resolve_local_weights_dir
        # returns the pinned snapshot dir, whose root holds the single
        # *.gguf file (allow_patterns keeps only Qwen3-1.7B-Q4_K_M.gguf).
        weights_dir = resolve_local_weights_dir(self.model_id)
        model_files = glob.glob(os.path.join(weights_dir, "*.gguf"))
        if not model_files:
            raise RuntimeError(f"No GGUF model file found in {weights_dir}")

        # HUTASH_DEVICE (cuda|cuda:N|auto|cpu|mps) -> llama.cpp GPU offload.
        # Internal CPU component by default (cpu/auto -> 0 GPU layers); any
        # explicit cuda flavour (incl. "cuda:0") or mps offloads all layers when
        # the llama build supports it. startswith so "cuda:N" isn't misread as
        # CPU-only.
        _device = os.environ.get("HUTASH_DEVICE", "cpu")
        n_gpu_layers = -1 if (_device.startswith("cuda") or _device == "mps") else 0
        self.model = Llama(
            model_path=model_files[0],
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    @capability("improve")
    def improve(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict:
        """Rewrite a user prompt. Matches improver.py's wire contract."""
        # Qwen3 recognizes `/no_think` (with underscore) to skip the
        # chain-of-thought <think> block. Without it, the model spends
        # its token budget inside <think> and often gets truncated
        # before emitting the actual improved prompt.
        user_prompt = (
            prompt if "/no_think" in prompt.lower()
            else f"{prompt} /no_think"
        )

        completion = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )

        text = completion["choices"][0]["message"]["content"]
        # Strip full closed <think>...</think> blocks, then any
        # unclosed <think>... trailing chunk if generation truncated
        # mid-reasoning.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return {"response": text.strip()}
