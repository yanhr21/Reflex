#!/usr/bin/env python3
"""Sequential Hugging Face folder uploader with a resumable JSONL log."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import HfApi


def load_done(log_path: Path) -> set[str]:
    done: set[str] = set()
    if not log_path.exists():
        return done
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "done" and isinstance(record.get("path"), str):
                done.add(record["path"])
    return done


def load_done_many(log_paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for log_path in log_paths:
        done.update(load_done(log_path))
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
    return sorted(files, key=lambda p: str(p.relative_to(root)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--path-prefix", default="")
    parser.add_argument("--max-retries", type=int, default=1000000)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    parser.add_argument("--extra-done-log", action="append", default=[], type=Path)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sort", choices=["name", "size-desc"], default="name")
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument(
        "--exclude-part",
        action="append",
        default=[".cache", "__pycache__"],
        help="Skip files whose relative path contains this path component.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < count")

    done = load_done_many([args.log, *args.extra_done_log])
    files = iter_files(root, set(args.exclude_part))
    files = [path for path in files if path.stat().st_size >= args.min_size]
    if args.max_size is not None:
        files = [path for path in files if path.stat().st_size <= args.max_size]
    if args.sort == "size-desc":
        files.sort(key=lambda path: (-path.stat().st_size, str(path.relative_to(root))))
    files = [path for idx, path in enumerate(files) if idx % args.shard_count == args.shard_index]
    pending = [path for path in files if str(path.relative_to(root)) not in done]
    total_bytes = sum(path.stat().st_size for path in files)
    pending_bytes = sum(path.stat().st_size for path in pending)

    print(
        f"repo={args.repo_id} root={root} files={len(files)} "
        f"done={len(done)} pending={len(pending)} total_bytes={total_bytes} "
        f"pending_bytes={pending_bytes} shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )
    write_log(
        args.log,
        {
            "status": "scan",
            "files": len(files),
            "done": len(done),
            "pending": len(pending),
            "total_bytes": total_bytes,
            "pending_bytes": pending_bytes,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
    )

    api = HfApi()
    uploaded = 0
    for index, path in enumerate(pending, start=1):
        rel = str(path.relative_to(root))
        path_in_repo = f"{args.path_prefix.rstrip('/')}/{rel}" if args.path_prefix else rel
        size = path.stat().st_size
        write_log(args.log, {"status": "start", "path": rel, "path_in_repo": path_in_repo, "size": size})
        print(f"[{index}/{len(pending)}] upload {rel} size={size}", flush=True)

        for attempt in range(1, args.max_retries + 1):
            try:
                url = api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path_in_repo,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    commit_message=f"Upload {path_in_repo}",
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - log and retry long network transfers.
                write_log(
                    args.log,
                    {
                        "status": "error",
                        "path": rel,
                        "attempt": attempt,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"error path={rel} attempt={attempt}: {exc!r}", flush=True)
                time.sleep(min(args.retry_sleep * attempt, 900.0))
                continue

            uploaded += 1
            write_log(args.log, {"status": "done", "path": rel, "path_in_repo": path_in_repo, "size": size, "url": str(url)})
            print(f"done {rel} -> {url}", flush=True)
            break
        else:
            write_log(args.log, {"status": "failed", "path": rel, "size": size})
            return 1

    write_log(args.log, {"status": "complete", "uploaded": uploaded})
    print(f"complete uploaded={uploaded}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
