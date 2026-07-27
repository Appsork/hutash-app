# Community packages

Third-party applications packaged for Hutash. **In development — not active.**

These are not part of the shipping product yet. They are published here so the
work is visible and reviewable, and so anyone already running one keeps its
display metadata, but the Hutash OS App Store does not list them by default.

## Why they are held back

Every app here manages its own models. Each downloads weights to its own
location and deletes them on its own terms, outside the shared CAMS shelf that
Hutash uses to give every app one library and one disk budget. That means
Hutash cannot honestly tell you what is installed, cannot reclaim space
reliably, and a shelf delete can orphan an app's models.

Four of them declare this explicitly in their manifest:

```yaml
metadata:
  managed_externally: true
  advanced_reason: ComfyUI downloads and deletes models through its own manager.
```

| Package | Why it is set apart |
|---|---|
| `comfyui` | Downloads and deletes models through its own manager |
| `forge` | Downloads and deletes models through its own UI |
| `invokeai` | Manages its own model library and install paths |
| `rvc` | Downloads its own weights outside the shared shelf |

The rest (`a1111`, `anythingllm`, `localai`, `ollama`, `openwebui`, `whisper`)
are packaged but unproven — they have not been through the install, health and
lifecycle testing the first-party apps get.

## How to see them anyway

Hutash OS has an **Advanced** toggle in the App Store. It lists everything here,
each card badged with why it is set apart. An app installed this way runs
normally; Hutash simply does not claim to manage its models.

## What would move one out of here

1. Its models install to, and delete from, the shared shelf.
2. Its install, health check and stop are driven by the engine, not by a
   bespoke script.
3. It passes the same lifecycle tests the first-party apps do.

At that point the package moves to `apps/` and the `managed_externally` flag
comes off.

## Adding a package

Drop the `.hutash` in this directory and run `python scripts/build_index.py`
from the repo root. The index is generated from what is on disk, so it can never
advertise a package that is not there.
