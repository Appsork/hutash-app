#!/usr/bin/env python3
"""Generate the root ``index.json`` from what is actually on disk.

One discovery index for every consumer, built by scanning ``pipelines/``,
``apps/`` and ``plugins/`` (future) and reading each ``.hutash``'s
``manifest.yaml`` for identity — so the index can never claim a package that
is not there, and a new package is published by dropping the file in and
re-running this.

The index answers exactly one question: what packages exist, and where is
each one's ``.hutash`` file? Nothing else. Every other fact about a package —
name, version, license, quality score, description, the UI contract — lives
in that package's own ``manifest.yaml`` (``docs/HUTASH_FORMAT.md``'s
three-layer format); this script never copies those fields into the index,
so there is nothing here that can drift out of sync with the package it
describes. See HUTASH_FORMAT.md's "Package Registry — index.json".

Two on-disk forms are read, because both exist in this repo:

- **v2.0+ package** — a zip whose ``manifest.yaml`` carries identity.
- **flat YAML** — a single document (the older third-party declarations).

``modalities`` and ``features`` are Studio's UI-routing tables — not
per-package data, so they have no home in any single ``manifest.yaml``.
They ride alongside ``packages`` at the index's top level, carried forward
from whatever the currently-committed index already has (there is nothing
else to source them from once they are written once).

Usage:  python scripts/build_index.py [--check]

``--check`` regenerates in memory and exits non-zero if the committed index is
stale, which is what CI runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.json"

# Section name -> (folder, package type). "community" isn't its own type —
# third-party apps published there are applications like anything in apps/;
# only the folder they're grouped under differs.
SECTIONS: dict[str, tuple[Path, str]] = {
    "pipelines": (ROOT / "pipelines", "pipeline"),
    "apps": (ROOT / "apps", "application"),
    "community": (ROOT / "community", "application"),
    "plugins": (ROOT / "plugins", "plugin"),  # future — empty until any ship
}

# Raw-content base for absolute URLs that consumers still resolve eagerly.
# NOTE: this repo has not actually been renamed/moved to hutash-public — that
# repo does not exist on GitHub (confirmed 404 at the repo root). This was set
# prematurely for an unfinished rename; every package_url in the generated
# index was broken from the commit that introduced this constant. Point at
# the repo this actually lives in until a real rename happens.
RAW_BASE = "https://raw.githubusercontent.com/Appsork/hutash-app/main"


def read_manifest(path: Path) -> dict[str, Any]:
    """Identity mapping from a .hutash, whichever on-disk form it uses."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            member = "manifest.yaml" if "manifest.yaml" in names else None
            if member is None:
                return {}
            data = yaml.safe_load(z.read(member).decode("utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def category_for(manifest: dict[str, Any], package_type: str) -> str:
    """The index's coarse `category` field for one package.

    Pipelines: `metadata.weight_category` when the manifest declares one
    (matches docs/ARCHITECTURE.md's categorised weights layout — "audio",
    "LLM", …), else the modality of its first capability, else "pipeline".

    Applications: `metadata.category` (the real display category — "photo",
    "creative", …; the manifest's own top-level `category` is usually just
    the literal string "app" and not useful for grouping), else "app".
    """
    meta = manifest.get("metadata") or {}
    if package_type == "pipeline":
        weight_category = meta.get("weight_category")
        if weight_category:
            return str(weight_category)
        caps = manifest.get("capabilities") or []
        if caps and isinstance(caps[0], dict) and caps[0].get("modality"):
            return str(caps[0]["modality"])
        return "pipeline"
    return str(meta.get("category") or "app")


def entry_for(path: Path, section: str, package_type: str) -> dict[str, Any]:
    """One minimal index entry: id, type, category, and where to find it.

    No display or install metadata — that all lives in the package's own
    manifest.yaml, which a consumer reads directly (see HUTASH_FORMAT.md).
    """
    manifest = read_manifest(path)
    app_id = str(manifest.get("id") or path.stem)
    return {
        "id": app_id,
        "type": package_type,
        "category": category_for(manifest, package_type),
        "hutash": f"{section}/{path.name}",
    }


def legacy_blocks() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`modalities` + `features` — Studio's UI routing, carried across verbatim.

    Neither has a per-package home, so once written they are simply carried
    forward from whatever is already committed. Edit them directly in
    index.json (or via a future dedicated file) — this script never
    regenerates their content, only preserves it across a rebuild.
    """
    if INDEX.is_file():
        data = json.loads(INDEX.read_text(encoding="utf-8"))
        return data.get("modalities", {}), data.get("features", [])
    return {}, []


def build() -> dict[str, Any]:
    modalities, features = legacy_blocks()
    packages: list[dict[str, Any]] = []
    for section, (folder, package_type) in SECTIONS.items():
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.hutash")):
            packages.append(entry_for(p, section, package_type))

    return {
        "version": "1.0",
        "modalities": modalities,
        "features": features,
        "packages": packages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()

    built = build()
    text = json.dumps(built, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not INDEX.is_file():
            print("index.json is missing — run scripts/build_index.py")
            return 1
        if INDEX.read_text(encoding="utf-8") != text:
            print("index.json is stale — run scripts/build_index.py")
            return 1
        print("index.json is up to date")
        return 0

    INDEX.write_text(text, encoding="utf-8")
    by_type: dict[str, int] = {}
    for pkg in built["packages"]:
        by_type[pkg["type"]] = by_type.get(pkg["type"], 0) + 1
    counts = ", ".join(f"{k} {v}" for k, v in sorted(by_type.items()))
    print(f"wrote {INDEX.relative_to(ROOT)} — {len(built['packages'])} packages ({counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
