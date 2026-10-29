"""Shared operation selection for artifact setup and request admission."""

import os
from collections.abc import Mapping

DEFAULT_OPERATIONS = "llmlingua2,sot,embed,verify"


def configured_operations(artifacts: Mapping) -> tuple[str, ...]:
    # Empty entries are harmless (including a trailing comma); an explicitly
    # blank configuration disables all operations instead of choosing defaults.
    configured = os.environ.get("OPTIMIZATION_OPERATIONS", DEFAULT_OPERATIONS)
    operations = tuple(dict.fromkeys(key.strip() for key in configured.split(",") if key.strip()))
    unknown = [key for key in operations if key not in artifacts]
    if unknown:
        raise RuntimeError(
            "Unknown OPTIMIZATION_OPERATIONS: " + ", ".join(unknown)
            + ". Supported operations: " + ", ".join(sorted(artifacts))
        )
    return operations
