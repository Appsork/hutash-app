"""Tests for framework-aware Low VRAM in hutash_inference.server.

Stdlib unittest so it runs in any model venv without pytest. Run from the
package parent (avoids the flat-layout logging.py shadowing stdlib logging):
    python -m unittest hutash_inference.test_low_vram
"""

import os
import unittest

from hutash_inference.server import _apply_low_vram, _patch_from_pretrained


class ApplyLowVramTest(unittest.TestCase):
    """_apply_low_vram translates the engine's budget per framework."""

    _VARS = [
        "HUTASH_VRAM_BUDGET_MB",
        "HUTASH_MODEL_FRAMEWORK",
        "ACCELERATE_DEVICE_MAP",
        "ACCELERATE_MAX_MEMORY",
        "HUTASH_GPU_LAYERS",
    ]

    def setUp(self):
        # Snapshot and clear so each case controls the env in isolation.
        self._saved = {k: os.environ.pop(k, None) for k in self._VARS}

    def tearDown(self):
        for k in self._VARS:
            os.environ.pop(k, None)
            if self._saved[k] is not None:
                os.environ[k] = self._saved[k]

    def test_huggingface_sets_device_map_env(self):
        os.environ["HUTASH_VRAM_BUDGET_MB"] = "4000"
        os.environ["HUTASH_MODEL_FRAMEWORK"] = "huggingface"
        _apply_low_vram()
        self.assertEqual(os.environ["ACCELERATE_DEVICE_MAP"], "auto")
        self.assertEqual(
            os.environ["ACCELERATE_MAX_MEMORY"],
            '{"0": "4000MB", "cpu": "48GB"}',
        )

    def test_llama_cpp_calculates_gpu_layers(self):
        os.environ["HUTASH_VRAM_BUDGET_MB"] = "3000"
        os.environ["HUTASH_MODEL_FRAMEWORK"] = "llama-cpp"
        _apply_low_vram()
        # 3000 // 300 == 10 layers; no HuggingFace env set.
        self.assertEqual(os.environ["HUTASH_GPU_LAYERS"], "10")
        self.assertNotIn("ACCELERATE_DEVICE_MAP", os.environ)

    def test_faster_whisper_does_nothing(self):
        os.environ["HUTASH_VRAM_BUDGET_MB"] = "2000"
        os.environ["HUTASH_MODEL_FRAMEWORK"] = "faster-whisper"
        _apply_low_vram()
        self.assertNotIn("ACCELERATE_DEVICE_MAP", os.environ)
        self.assertNotIn("HUTASH_GPU_LAYERS", os.environ)

    def test_no_budget_does_nothing(self):
        # Framework set but no budget → Low VRAM inactive, load normally.
        os.environ["HUTASH_MODEL_FRAMEWORK"] = "huggingface"
        _apply_low_vram()
        self.assertNotIn("ACCELERATE_DEVICE_MAP", os.environ)
        self.assertNotIn("HUTASH_GPU_LAYERS", os.environ)

    def test_invalid_budget_is_ignored(self):
        os.environ["HUTASH_VRAM_BUDGET_MB"] = "not-a-number"
        os.environ["HUTASH_MODEL_FRAMEWORK"] = "huggingface"
        _apply_low_vram()  # must not raise
        self.assertNotIn("ACCELERATE_DEVICE_MAP", os.environ)


class PatchFromPretrainedTest(unittest.TestCase):
    """The from_pretrained wrapper injects kwargs, preserves cls, is idempotent."""

    @staticmethod
    def _make_cls():
        class Fake:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return {"cls": cls.__name__, "path": path, "kwargs": kwargs}

        return Fake

    def test_injects_device_map_and_max_memory(self):
        Fake = self._make_cls()
        _patch_from_pretrained(Fake, "auto", {0: "4000MB", "cpu": "48GB"})
        out = Fake.from_pretrained("model")
        self.assertEqual(out["kwargs"]["device_map"], "auto")
        self.assertEqual(out["kwargs"]["max_memory"], {0: "4000MB", "cpu": "48GB"})
        self.assertEqual(out["cls"], "Fake")  # cls binding preserved

    def test_preserves_subclass_binding(self):
        Fake = self._make_cls()
        _patch_from_pretrained(Fake, "auto", None)

        class Sub(Fake):
            pass

        out = Sub.from_pretrained("model")
        self.assertEqual(out["cls"], "Sub")  # loads as Sub, not the patched base
        self.assertEqual(out["kwargs"]["device_map"], "auto")

    def test_respects_explicit_device_map(self):
        Fake = self._make_cls()
        _patch_from_pretrained(Fake, "auto", {0: "4000MB"})
        out = Fake.from_pretrained("model", device_map="cuda:0")
        self.assertEqual(out["kwargs"]["device_map"], "cuda:0")
        self.assertNotIn("max_memory", out["kwargs"])  # not injected when device_map given

    def test_idempotent(self):
        Fake = self._make_cls()
        _patch_from_pretrained(Fake, "auto", None)
        raw1 = Fake.__dict__["from_pretrained"]
        _patch_from_pretrained(Fake, "auto", None)  # second call no-ops
        raw2 = Fake.__dict__["from_pretrained"]
        self.assertIs(raw1, raw2)  # not double-wrapped


if __name__ == "__main__":
    unittest.main()
