"""Startup-only import speed patch for Cosmos training jobs.

The shared venv can make importlib.metadata.packages_distributions() scan
every installed distribution over a slow filesystem. Diffusers calls it during
module import only to precompute an optional top-level-package map. Returning
an empty map preserves normal per-package version checks while avoiding the
global metadata walk.
"""

from __future__ import annotations

import os


_FALSE_VALUES = {"0", "FALSE", "NO", "OFF"}


def _enabled() -> bool:
    return os.environ.get("COSMOS_SKIP_PACKAGE_DISTRIBUTION_SCAN", "1").upper() not in _FALSE_VALUES


def _empty_packages_distributions() -> dict[str, list[str]]:
    return {}


if _enabled():
    import importlib.metadata as _stdlib_metadata

    _stdlib_metadata.packages_distributions = _empty_packages_distributions

    try:
        import importlib_metadata as _backport_metadata
    except Exception:
        _backport_metadata = None
    if _backport_metadata is not None:
        _backport_metadata.packages_distributions = _empty_packages_distributions
