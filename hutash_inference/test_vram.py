"""Tests for hutash_inference.vram.get_device_map_kwargs.

Stdlib unittest so it runs in any model venv without pytest. Run with:
    python -m unittest hutash_inference.test_vram
    python hutash_inference/test_vram.py
"""

import os
import unittest

from hutash_inference.vram import get_device_map_kwargs

_ENV = "HUTASH_VRAM_BUDGET_MB"


class GetDeviceMapKwargsTest(unittest.TestCase):
    def setUp(self):
        # Snapshot and clear so each case controls the env var in isolation.
        self._saved = os.environ.pop(_ENV, None)

    def tearDown(self):
        os.environ.pop(_ENV, None)
        if self._saved is not None:
            os.environ[_ENV] = self._saved

    def test_unset_returns_empty(self):
        self.assertEqual(get_device_map_kwargs(), {})

    def test_valid_budget_returns_device_map(self):
        os.environ[_ENV] = "4000"
        kwargs = get_device_map_kwargs()
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertEqual(kwargs["max_memory"], {0: "4000MB", "cpu": "48GB"})

    def test_invalid_budget_returns_empty_and_warns(self):
        os.environ[_ENV] = "not-a-number"
        with self.assertLogs("hutash_inference.vram", level="WARNING") as cm:
            self.assertEqual(get_device_map_kwargs(), {})
        self.assertTrue(any("Invalid" in m for m in cm.output))

    def test_zero_budget_returns_empty(self):
        os.environ[_ENV] = "0"
        self.assertEqual(get_device_map_kwargs(), {})

    def test_negative_budget_returns_empty(self):
        os.environ[_ENV] = "-100"
        self.assertEqual(get_device_map_kwargs(), {})

    def test_empty_string_returns_empty(self):
        os.environ[_ENV] = ""
        self.assertEqual(get_device_map_kwargs(), {})


if __name__ == "__main__":
    unittest.main()
