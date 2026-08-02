#!/usr/bin/env python3
"""Generate the root ``index.json`` from what is actually on disk.

One discovery index for every consumer, built by scanning ``pipelines/``,
``apps/`` and ``plugins/`` (future) and reading each ``.hutash``'s
``manifest.yaml`` for identity — so the index can never claim a package that
is not there, and a new package is published by dropping the file in and
re-running this.

The index carries identity (id, type, category, where to find the ``.hutash``)
PLUS a derived summary of display fields — name, description, version,
license, min_vram_gb, disk_size_gb, quality_score, speed, modality,
hardware_label — read straight from that package's own ``manifest.yaml``
(+ ``resources/weights.yaml`` for a pipeline's download size). ``manifest.yaml``
stays the source of truth for the full UI contract and every other fact; this
script re-derives the summary from it on every run, so the two can never drift
out of sync as long as this is re-run whenever a package changes. This lets a
client render the marketplace instantly from ``index.json`` alone — no
per-package ``.hutash`` fetch until the user actually installs something.

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


def read_weights(path: Path) -> dict[str, Any]:
    """A pipeline's ``resources/weights.yaml`` (download size, HF source), or
    ``{}`` — absent for a flat-YAML package, an application, or any package
    with no external weights."""
    if not zipfile.is_zipfile(path):
        return {}
    with zipfile.ZipFile(path) as z:
        if "resources/weights.yaml" not in z.namelist():
            return {}
        data = yaml.safe_load(z.read("resources/weights.yaml").decode("utf-8"))
    return data if isinstance(data, dict) else {}


def display_fields(manifest: dict[str, Any], weights: dict[str, Any]) -> dict[str, Any]:
    """The derived display summary for one package, read straight from its own
    manifest (+ weights.yaml for download size). Every field defaults to a
    predictable empty value rather than being omitted, so a consumer never has
    to branch on whether a key is present.

    ``speed`` has no home in any current manifest — passed through from
    ``metadata.speed`` if a future manifest ever adds one, never fabricated.
    ``hardware_label`` has no manifest field either; it is DERIVED from the
    real ``gpu``/``min_vram_gb`` fields, not copied from anywhere.

    Beyond the display fields themselves, this also carries ``internal``,
    ``hf_repo``, ``hf_revision``, ``allow_patterns`` and ``weights_external``
    — not display data, but every field a consumer (Studio) needs to build
    its full model-catalogue row from the index ALONE, with no per-package
    ``.hutash`` fetch. Without these a consumer skipping the zip fetch would
    silently lose real behaviour: ``internal`` gates a model out of the
    user-facing list (prompt-engine), and ``hf_repo``/``hf_revision``/
    ``weights_external`` drive an already-installed model's weights-presence
    checks. Omitting them here would not shrink the index (they're already
    read from the same manifest/weights.yaml this function already opened);
    it would just move the zip fetch to break invisibly downstream.
    """
    meta = manifest.get("metadata") or {}
    caps = manifest.get("capabilities") or []
    modality = ""
    if caps and isinstance(caps[0], dict):
        modality = str(caps[0].get("modality") or "")

    min_vram = manifest.get("min_vram_gb")
    min_vram_gb = min_vram if isinstance(min_vram, (int, float)) else 0
    if manifest.get("gpu") == "required":
        hardware_label = f"GPU ({min_vram_gb:g}GB+)" if min_vram_gb > 0 else "GPU"
    else:
        hardware_label = "CPU"

    sources = weights.get("sources") or []
    first_source = sources[0] if sources and isinstance(sources[0], dict) else {}

    return {
        "name": str(manifest.get("name") or ""),
        "description": str(manifest.get("description") or meta.get("description") or ""),
        "version": str(manifest.get("version") or ""),
        "license": str(manifest.get("license") or ""),
        "min_vram_gb": min_vram_gb,
        "disk_size_gb": weights.get("download_size_gb"),
        "quality_score": meta.get("quality_score"),
        "speed": meta.get("speed"),
        "modality": modality,
        "hardware_label": hardware_label,
        "internal": bool(meta.get("internal", False)),
        "hf_repo": first_source.get("repo"),
        "hf_revision": first_source.get("revision"),
        "allow_patterns": first_source.get("allow_patterns"),
        "weights_external": bool(sources),
    }


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
    """One index entry: identity (id, type, category, where to find the
    ``.hutash``) plus the derived display summary (see ``display_fields``).

    Install/UI-contract metadata beyond the display summary still lives only
    in the package's own manifest.yaml, fetched at install time.
    """
    manifest = read_manifest(path)
    weights = read_weights(path)
    app_id = str(manifest.get("id") or path.stem)
    entry = {
        "id": app_id,
        "type": package_type,
        "category": category_for(manifest, package_type),
        "hutash": f"{section}/{path.name}",
    }
    entry.update(display_fields(manifest, weights))
    return entry


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
