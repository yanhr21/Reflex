#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py


def copy_h5_group(src: h5py.File, dst: h5py.File, name: str) -> None:
    if name not in src:
        raise KeyError(f"missing H5 group: {name}")
    src.copy(name, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-h5", required=True)
    parser.add_argument("--src-json", required=True)
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-count", type=int, required=True)
    args = parser.parse_args()

    src_h5 = Path(args.src_h5)
    src_json = Path(args.src_json)
    out_h5 = Path(args.out_h5)
    out_json = Path(args.out_json)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with src_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("source JSON missing episodes list")
    if args.episode_start < 0:
        raise ValueError("--episode-start must be nonnegative")
    if args.episode_count <= 0:
        raise ValueError("--episode-count must be positive")

    selected = episodes[args.episode_start : args.episode_start + args.episode_count]
    if not selected:
        raise ValueError("selected shard has no episodes")

    out_data = dict(data)
    out_data["episodes"] = selected
    out_data["source_episode_start"] = args.episode_start
    out_data["source_episode_count_requested"] = args.episode_count
    out_data["source_episode_count_written"] = len(selected)
    out_data["source_json"] = str(src_json)
    out_data["source_h5"] = str(src_h5)

    tmp_h5 = out_h5.with_suffix(out_h5.suffix + ".tmp")
    tmp_json = out_json.with_suffix(out_json.suffix + ".tmp")
    if tmp_h5.exists():
        tmp_h5.unlink()
    if tmp_json.exists():
        tmp_json.unlink()

    with h5py.File(src_h5, "r") as src, h5py.File(tmp_h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for episode in selected:
            episode_id = episode.get("episode_id")
            if episode_id is None:
                raise ValueError(f"selected episode missing episode_id: {episode}")
            copy_h5_group(src, dst, f"traj_{episode_id}")

    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
        f.write("\n")

    shutil.move(str(tmp_h5), str(out_h5))
    shutil.move(str(tmp_json), str(out_json))

    print(f"static_replay_shard_written=true")
    print(f"source_episode_start={args.episode_start}")
    print(f"source_episode_count_requested={args.episode_count}")
    print(f"source_episode_count_written={len(selected)}")
    print(f"out_h5={out_h5}")
    print(f"out_json={out_json}")


if __name__ == "__main__":
    main()
