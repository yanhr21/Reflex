#!/usr/bin/env python3
"""Small compute-node probe for Phase 03 runner import latency."""

from __future__ import annotations

import importlib
import time


MODULES = [
    "json",
    "os",
    "subprocess",
    "time",
    "dataclasses",
    "pathlib",
    "typing",
    "gymnasium",
    "imageio.v2",
    "mani_skill.envs",
    "numpy",
    "torch",
    "tyro",
    "gymnasium.vector",
    "mani_skill.sensors.camera",
    "mani_skill.utils.common",
    "mani_skill.utils.sapien_utils",
    "mani_skill.utils.structs",
    "mani_skill.utils.wrappers",
    "train",
]


def main() -> int:
    for module_name in MODULES:
        start = time.time()
        print(f"import_start {module_name}", flush=True)
        importlib.import_module(module_name)
        print(f"import_done {module_name} {time.time() - start:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
