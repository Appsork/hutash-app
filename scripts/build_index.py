#!/usr/bin/env python3
"""Generate the root ``index.json`` from what is actually on disk.

One browse index for every consumer, built by scanning ``pipelines/``, ``apps/``
and ``community/`` and reading each ``.hutash`` manifest — so the index can never
claim a package that is not there, and a new package is published by dropping
the file in and re-running this.

Two on-disk forms are read, because both exist in this repo:

- **v2.0 package** — a zip whose ``manifest.yaml`` carries identity.
- **flat YAML** — a single document (the older third-party declarations).

Display fields the manifests do not carry (modality, quality score, weight
provenance…) come from the previous ``catalogue/catalogue.json``, which this
index replaces. That file is the historical source for pipeline metadata; once
it is gone the values live here, and this script preserves whatever it finds in
the existing index so a regeneration is never lossy.

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
LEGACY_CATALOGUE = ROOT / "catalogue" / "catalogue.json"

SECTIONS = {
    "pipelines": ROOT / "pipelines",
    "apps": ROOT / "apps",
    "community": ROOT / "community",
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


def human_size(gb: float | None) -> str:
    """Size for a browse listing: '500 MB' / '1.4 GB'. Empty when unknown."""
    if not gb or gb <= 0:
        return ""
    if gb < 1:
        return f"{int(round(gb * 1024))} MB"
    return f"{gb:.1f} GB".replace(".0 GB", " GB")


def legacy_models() -> dict[str, dict[str, Any]]:
    """Pipeline metadata from the catalogue.json this index replaces."""
    if not LEGACY_CATALOGUE.is_file():
        return {}
    data = json.loads(LEGACY_CATALOGUE.read_text(encoding="utf-8"))
    return {m["id"]: m for m in data.get("models", []) if m.get("id")}


def legacy_blocks() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`modalities` + `features` — Studio's UI routing, carried across verbatim."""
    if LEGACY_CATALOGUE.is_file():
        data = json.loads(LEGACY_CATALOGUE.read_text(encoding="utf-8"))
        return data.get("modalities", {}), data.get("features", [])
    if INDEX.is_file():
        data = json.loads(INDEX.read_text(encoding="utf-8"))
        return data.get("modalities", {}), data.get("features", [])
    return {}, []


def entry_for(path: Path, section: str, prior: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One index entry: manifest identity, enriched with prior metadata."""
    manifest = read_manifest(path)
    app_id = str(manifest.get("id") or path.stem)
    old = prior.get(app_id, {})
    meta = manifest.get("metadata") or {}

    description = (
        manifest.get("description")
        or meta.get("description")
        or old.get("description")
        or ""
    )
    license_ = manifest.get("license") or old.get("license") or ""
    name = manifest.get("name") or old.get("display_name") or app_id
    rel = f"{section}/{path.name}"

    entry: dict[str, Any] = {
        "id": app_id,
        "name": name,
        "description": description,
        # `type` is the browse-level kind. Pipelines keep their modality (tts,
        # stt, music, clone) because that is what the Studio marketplace filters
        # on; everything else is an "app" — the manifests spell that
        # "application" or "app" interchangeably, so it is normalised here.
        "type": (
            old.get("modality")
            if section == "pipelines"
            else "app"
        ) or manifest.get("type") or "app",
        "license": license_,
        "size_display": human_size(old.get("disk_size_gb")),
        "hutash_file": rel,
        # Kept for consumers that join on a bare filename rather than a path.
        "file": path.name,
    }

    if section == "pipelines":
        # Fields Studio's catalogue validator and installer require. Absent
        # values are omitted rather than guessed — a wrong version or hash is
        # worse than a missing one.
        entry["display_name"] = old.get("display_name") or name
        entry["modality"] = old.get("modality", "")
        entry["version"] = manifest.get("version") or old.get("version") or ""
        # The installable package now lives under pipelines/.
        entry["package_url"] = f"{RAW_BASE}/{rel}"
        for key in (
            "weight_category",
            "min_vram_gb",
            "disk_size_gb",
            "hf_repo",
            "hf_revision",
            "weights_external",
            "allow_patterns",
            "quality_score",
            "requires_aura",
            "docker_image",
            "ghcr_image",
        ):
            if key in old:
                entry[key] = old[key]

    if section == "community":
        # Externally-managed apps carry their reason so the OS can badge them.
        if meta.get("managed_externally"):
            entry["managed_externally"] = True
            entry["advanced_reason"] = meta.get("advanced_reason", "")

    return entry


def build() -> dict[str, Any]:
    prior = legacy_models()
    if not prior and INDEX.is_file():
        # Regenerating after catalogue.json is gone: the index is its own prior.
        current = json.loads(INDEX.read_text(encoding="utf-8"))
        prior = {e["id"]: e for e in current.get("pipelines", []) if e.get("id")}

    modalities, features = legacy_blocks()
    out: dict[str, Any] = {
        "version": "1.0",
        "modalities": modalities,
        "features": features,
    }
    for section, folder in SECTIONS.items():
        entries = [
            entry_for(p, section, prior)
            for p in sorted(folder.glob("*.hutash"))
        ]
        out[section] = entries
    return out


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
    counts = ", ".join(f"{k} {len(built[k])}" for k in SECTIONS)
    print(f"wrote {INDEX.relative_to(ROOT)} — {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
