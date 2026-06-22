"""Validate that inference.py matches model.yaml capabilities.

Runs at container startup. If mismatch, raises ValidationError with
clear message so operator knows what to fix.
"""

import inspect

from hutash_inference.base import Inference, get_capabilities
from hutash_inference.errors import ValidationError


def validate_inference_matches_manifest(
    instance: Inference,
    manifest: dict,
) -> None:
    """Verify Inference instance implements all capabilities in manifest.

    Checks:
    - Every capability declared in manifest has a @capability method
    - Method parameter names match declared inputs + controls
    - Extra @capability methods without manifest declaration are tolerated
      (developer may be iterating)

    Args:
        instance: Loaded Inference subclass instance
        manifest: Parsed manifest.json contents

    Raises:
        ValidationError: If any capability mismatch detected
    """
    declared_capabilities = manifest.get("capabilities", {}) or {}
    implemented_capabilities = get_capabilities(instance)

    # Every declared capability must have an implementation
    missing = [
        cap_id for cap_id in declared_capabilities
        if cap_id not in implemented_capabilities
    ]
    if missing:
        raise ValidationError(
            f"Manifest declares capabilities {missing} but inference.py "
            f"has no @capability methods for them. "
            f"Implemented: {list(implemented_capabilities.keys())}"
        )

    # Verify parameter names for each implemented capability
    for cap_id, method in implemented_capabilities.items():
        if cap_id not in declared_capabilities:
            continue
        validate_method_signature(cap_id, method, declared_capabilities[cap_id])


def validate_method_signature(
    capability_id: str,
    method: callable,
    spec: dict,
) -> None:
    """Verify method parameter names match declared inputs + wired controls.

    Only controls whose implementation_status is "wired" (or unspecified)
    must appear in the method signature. Stub and planned controls are
    declarative-only â€” UI renders them, but inference.py need not accept
    them as kwargs, and the server filters unexpected kwargs out before
    calling the method.

    Raises:
        ValidationError: If required parameters are missing from the method
    """
    expected_params: set[str] = set()
    expected_params.update((spec.get("inputs") or {}).keys())
    for ctrl_id, ctrl_spec in (spec.get("controls") or {}).items():
        status = (ctrl_spec or {}).get("implementation_status", "wired")
        if status == "wired":
            expected_params.add(ctrl_id)

    sig = inspect.signature(method)
    actual_params: set[str] = set()
    for param_name in sig.parameters:
        if param_name == "self":
            continue
        actual_params.add(param_name)

    missing = expected_params - actual_params
    if missing:
        raise ValidationError(
            f"Capability '{capability_id}' method missing parameters: "
            f"{missing}. Manifest declares inputs + wired controls: "
            f"{expected_params}. Method signature has: {actual_params}"
        )

    # Extra actual params are OK (could be internal defaults)
