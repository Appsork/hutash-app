"""Logging configuration for Aura model containers.

Ensures consistent log format across all containers so Aura's Core API
can parse them uniformly.
"""

import logging
import sys


def configure_logging(model_id: str, level: str = "INFO") -> None:
    """Set up logging for a model container.

    Args:
        model_id: The model's ID (appears in log lines)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    log_format = (
        f"%(asctime)s [{model_id}] %(name)s %(levelname)s: %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Silence overly chatty third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
