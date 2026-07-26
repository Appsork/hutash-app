"""Tests for the /park and /unpark endpoints (Sarathi Step 2 Part C).

Builds a real app via create_app() over a temp model dir, so the actual route
wiring is exercised — not a hand-rolled copy. No torch needed: the base
offload()/reload() degrade to a no-op move (0 modules) when torch is absent, which
is exactly the path a CPU-only test host takes.

Stdlib unittest so it runs in any model venv without pytest:
    python -m unittest hutash_inference.test_park_unpark
"""

import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient


_INFERENCE_PY = '''\
from hutash_inference.base import Inference


class TinyModel(Inference):
    def load(self):
        # No weights to load; the park/unpark path does not need real modules.
        self.calls = []

    def offload(self):
        self.calls.append("offload")
        return {"status": "offloaded", "device": "cpu", "moved": 3}

    def reload(self):
        self.calls.append("reload")
        return {"status": "reloaded", "device": "cuda:0", "moved": 3}
'''

_MANIFEST = {
    "model_id": "tiny",
    "image": "example/tiny:v0",
    "capabilities": {},
}


class ParkUnparkEndpointTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        d = self._dir.name
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(_MANIFEST, f)
        with open(os.path.join(d, "inference.py"), "w") as f:
            f.write(_INFERENCE_PY)
        self._saved = os.environ.get("HUTASH_MODEL_DIR")
        os.environ["HUTASH_MODEL_DIR"] = d

        from hutash_inference.server import create_app

        self.client = TestClient(create_app())

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("HUTASH_MODEL_DIR", None)
        else:
            os.environ["HUTASH_MODEL_DIR"] = self._saved
        self._dir.cleanup()

    def test_park_calls_offload(self):
        r = self.client.post("/park")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "offloaded", "device": "cpu", "moved": 3})

    def test_unpark_calls_reload(self):
        r = self.client.post("/unpark")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "reloaded")

    def test_offload_alias_matches_park(self):
        self.assertEqual(
            self.client.post("/offload").json(),
            self.client.post("/park").json(),
        )

    def test_reload_alias_matches_unpark(self):
        self.assertEqual(
            self.client.post("/reload").json()["status"],
            self.client.post("/unpark").json()["status"],
        )


class BaseOffloadPinningTest(unittest.TestCase):
    """base.offload()/reload() must never crash on the pinning path — on a host
    without torch/CUDA, parking still succeeds with 0 storages pinned (Part F).
    """

    def _model(self):
        from hutash_inference.base import Inference

        class M(Inference):
            def load(self):
                pass

        return M(config={"model_id": "t"})

    def test_offload_reports_zero_pinned_without_cuda(self):
        m = self._model()
        out = m.offload()
        self.assertEqual(out["status"], "offloaded")
        self.assertEqual(out["pinned"], 0)  # no CUDA host-register available here
        self.assertEqual(m._pinned_host, [])

    def test_reload_clears_pins_and_succeeds(self):
        m = self._model()
        m.offload()
        out = m.reload()
        self.assertEqual(out["status"], "reloaded")
        self.assertEqual(m._pinned_host, [])  # unpin drained the list


if __name__ == "__main__":
    unittest.main()
