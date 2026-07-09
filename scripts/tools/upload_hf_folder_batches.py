#!/usr/bin/env python3
"""Batch Hugging Face folder uploader with resumable JSONL logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def load_done(log_paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for log_path in log_paths:
        if not log_path.exists():
            continue
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "done" and isinstance(record.get("path"), str):
                    done.add(record["path"])
    return done


def write_log(log_path: Path, record: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_files(root: Path, exclude_parts: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in exclude_parts for part in rel.parts):
            continue
        files.append(path)
    return files


def parse_retry_after(error_text: str, default: float) -> float:
    match = re.search(r"Retry after ([0-9]+) seconds", error_text)
    if match:
        return max(default, float(match.group(1)) + 5.0)
    if "128 per hour" in error_text or "rate limit" in error_text.lower():
        return max(default, 3700.0)
    return default


def make_batches(paths: list[Path], root: Path, max_files: int, max_bytes: int) -> list[list[Path]]:
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in paths:
        size = path.stat().st_size
        if current and (len(current) >= max_files or current_bytes + size > max_bytes):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += size
        if size > max_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
    if current:
        batches.append(current)
    return batches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--path-prefix", default="")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--extra-done-log", action="append", default=[], type=Path)
    parser.add_argument("--exclude-part", action="append", default=[".cache", "__pycache__"])
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sort", choices=["name", "size-desc"], default="name")
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--batch-files", type=int, default=100)
    parser.add_argument("--batch-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--retry-sleep", type=float, default=300.0)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")
    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard-index must satisfy 0 <= index < count")

    done = load_done([args.log, *args.extra_done_log])
    files = iter_files(root, set(args.exclude_part))
    files = [path for path in files if path.stat().st_size >= args.min_size]
    if args.max_size is not None:
        files = [path for path in files if path.stat().st_size <= args.max_size]
    if args.sort == "size-desc":
        files.sort(key=lambda path: (-path.stat().st_size, str(path.relative_to(root))))
    else:
        files.sort(key=lambda path: str(path.relative_to(root)))
    files = [path for idx, path in enumerate(files) if idx % args.shard_count == args.shard_index]
    pending = [path for path in files if str(path.relative_to(root)) not in done]
    batches = make_batches(pending, root, args.batch_files, args.batch_bytes)

    total_bytes = sum(path.stat().st_size for path in files)
    pending_bytes = sum(path.stat().st_size for path in pending)
    print(
        f"repo={args.repo_id} root={root} shard={args.shard_index}/{args.shard_count} "
        f"files={len(files)} done_known={len(done)} pending={len(pending)} "
        f"batches={len(batches)} total_bytes={total_bytes} pending_bytes={pending_bytes}",
        flush=True,
    )
    write_log(
        args.log,
        {
            "status": "scan",
            "files": len(files),
            "done_known": len(done),
            "pending": len(pending),
            "batches": len(batches),
            "total_bytes": total_bytes,
            "pending_bytes": pending_bytes,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
    )

    api = HfApi()
    for batch_idx, batch in enumerate(batches, start=1):
        batch_bytes = sum(path.stat().st_size for path in batch)
        rels = [str(path.relative_to(root)) for path in batch]
        operations = [
            CommitOperationAdd(
                path_in_repo=(
                    f"{args.path_prefix.rstrip('/')}/{rel}" if args.path_prefix else rel
                ),
                path_or_fileobj=str(root / rel),
            )
            for rel in rels
        ]
        write_log(args.log, {"status": "batch_start", "batch": batch_idx, "files": len(batch), "bytes": batch_bytes})
        print(f"[batch {batch_idx}/{len(batches)}] files={len(batch)} bytes={batch_bytes}", flush=True)

        attempt = 0
        while True:
            attempt += 1
            try:
                info = api.create_commit(
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    operations=operations,
                    commit_message=f"Upload experiment batch {args.shard_index}-{batch_idx}",
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - long upload, log and retry.
                text = repr(exc)
                sleep_s = parse_retry_after(text, args.retry_sleep)
                write_log(
                    args.log,
                    {
                        "status": "batch_error",
                        "batch": batch_idx,
                        "attempt": attempt,
                        "files": len(batch),
                        "bytes": batch_bytes,
                        "sleep": sleep_s,
                        "error": text,
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"batch_error batch={batch_idx} attempt={attempt} sleep={sleep_s}: {exc!r}", flush=True)
                time.sleep(sleep_s)
                continue

            for rel, path in zip(rels, batch):
                write_log(
                    args.log,
                    {
                        "status": "done",
                        "path": rel,
                        "path_in_repo": f"{args.path_prefix.rstrip('/')}/{rel}" if args.path_prefix else rel,
                        "size": path.stat().st_size,
                        "url": str(info.commit_url),
                        "batch": batch_idx,
                    },
                )
            print(f"done batch={batch_idx} url={info.commit_url}", flush=True)
            break

    write_log(args.log, {"status": "complete", "batches": len(batches)})
    print("complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
