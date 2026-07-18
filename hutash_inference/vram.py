"""Low VRAM support for Hutash model loading.

When the engine enables Low VRAM mode it sets HUTASH_VRAM_BUDGET_MB in the
model process's environment. A model's inference.py imports
get_device_map_kwargs() and splats the result into its transformers
from_pretrained() (or equivalent) call. When the budget is set, this asks
accelerate to place as many layers as fit within the VRAM budget on GPU and
spill the rest to CPU RAM (device_map="auto" + max_memory). When it is not
set, the returned empty dict is a no-op — the model loads exactly as before
(full GPU or full CPU, per the model's own logic).
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_device_map_kwargs():
    """Return kwargs for model loading that enable partial GPU/RAM splitting.

    When HUTASH_VRAM_BUDGET_MB is set (engine Low VRAM mode), returns
    device_map="auto" with max_memory so PyTorch places as many layers
    as fit in the VRAM budget on GPU, rest on CPU RAM.

    When not set, returns empty dict — model loads normally (full GPU or full CPU).
    """
    budget_mb = os.environ.get("HUTASH_VRAM_BUDGET_MB")
    if not budget_mb:
        return {}
    try:
        budget = int(budget_mb)
    except ValueError:
        logger.warning("Invalid HUTASH_VRAM_BUDGET_MB=%s, ignoring", budget_mb)
        return {}
    if budget <= 0:
        return {}
    logger.info("Low VRAM mode: budget %dMB — using device_map=auto", budget)
    return {
        "device_map": "auto",
        "max_memory": {0: f"{budget}MB", "cpu": "48GB"},
    }
