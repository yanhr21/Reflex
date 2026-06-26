#!/usr/bin/env python3
"""Keep allocated GPUs warm until a claim root reaches a done-count target."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-root", type=Path, required=True)
    parser.add_argument("--min-done", type=int, default=1)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--inner-loops", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    parser.add_argument("--max-loops", type=int, default=500000)
    return parser.parse_args()


def done_count(claim_root: Path) -> int:
    if not claim_root.exists():
        return 0
    return sum(1 for _ in claim_root.glob("*/done.txt"))


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    num_devices = torch.cuda.device_count()
    if num_devices < 1:
        raise RuntimeError("no CUDA devices visible")

    state = []
    for idx in range(num_devices):
      dev = torch.device(f"cuda:{idx}")
      a = torch.randn((args.matrix_size, args.matrix_size), device=dev, dtype=torch.float16)
      b = torch.randn((args.matrix_size, args.matrix_size), device=dev, dtype=torch.float16)
      out = torch.empty((args.matrix_size, args.matrix_size), device=dev, dtype=torch.float16)
      state.append((dev, a, b, out))

    print(f"gpu_keepalive_start devices={num_devices} min_done={args.min_done}", flush=True)
    loops = 0
    while loops < args.max_loops:
        count = done_count(args.claim_root)
        stop_exists = bool(args.stop_file and args.stop_file.exists())
        if stop_exists or count >= args.min_done:
            break
        for dev, a, b, out in state:
            with torch.cuda.device(dev):
                for _ in range(args.inner_loops):
                    torch.mm(a, b, out=out)
        for dev, *_ in state:
            with torch.cuda.device(dev):
                torch.cuda.synchronize()
        loops += 1
        if loops % 100 == 0:
            print(
                f"gpu_keepalive loops={loops} done={count} stop_file={stop_exists}",
                flush=True,
            )
        time.sleep(args.sleep_seconds)

    print(
        f"gpu_keepalive_stop loops={loops} done={done_count(args.claim_root)} "
        f"stop_file={bool(args.stop_file and args.stop_file.exists())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
