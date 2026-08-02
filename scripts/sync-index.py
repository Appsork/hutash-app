#!/usr/bin/env python3
"""Publish entry point: sync index.json's display summary with manifest.yaml.

Reads every ``.hutash`` in ``pipelines/`` and ``apps/``, extracts each one's
display fields from its ``manifest.yaml`` (+ ``resources/weights.yaml`` for a
pipeline's download size), and writes them into that package's ``index.json``
entry — so the index never drifts out of sync with the manifests it
summarises.

This is a thin entry point, not a second implementation: the actual
extraction and index-writing logic is ``build_index.py``'s (`display_fields`,
`entry_for`, `build`) — the repo's one generator, the one CI checks via
``--check``. A separate implementation here would risk exactly the drift this
script exists to prevent (two readers of the same manifests, disagreeing
after one changes) and — concretely — would break CI the moment it ran:
`build_index.py --check` rebuilds every entry from scratch on every run, so
any field written by a second, independent script would immediately read
back as "stale" against that rebuild.

Usage:  python scripts/sync-index.py [--check]
"""
from __future__ import annotations

import sys

from build_index import main

if __name__ == "__main__":
    sys.exit(main())
