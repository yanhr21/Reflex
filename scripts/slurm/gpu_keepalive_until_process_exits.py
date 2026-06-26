#!/usr/bin/env python3
"""Keep a Slurm allocation's GPU active while a target process is alive."""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pattern", required=True)
    parser.add_argument("--matmul-size", type=int, default=2048)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def target_alive(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    self_pid = os.getpid()
    for line in result.stdout.splitlines():
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if "gpu_keepalive_until_process_exits.py" in command:
            continue
        if "pgrep -af" in command:
            continue
        return True
    return False


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("cuda_not_available")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    torch.set_grad_enabled(False)
    print(
        "keepalive_start "
        f"device={torch.cuda.get_device_name(0)} "
        f"target_pattern={args.target_pattern!r} "
        f"matmul_size={args.matmul_size} "
        f"sleep_seconds={args.sleep_seconds} "
        f"dtype={args.dtype}",
        flush=True,
    )
    iterations = 0
    while target_alive(args.target_pattern):
        a = torch.randn((args.matmul_size, args.matmul_size), device="cuda", dtype=dtype)
        b = torch.randn((args.matmul_size, args.matmul_size), device="cuda", dtype=dtype)
        _ = a @ b
        torch.cuda.synchronize()
        iterations += 1
        if iterations % 100 == 0:
            print(f"keepalive_iterations={iterations}", flush=True)
        time.sleep(args.sleep_seconds)
    print(f"keepalive_stop iterations={iterations} reason=target_process_not_found", flush=True)


if __name__ == "__main__":
    main()
