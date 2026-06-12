#!/usr/bin/env python3
"""Assemble separate world-model member runs into one inspectable ensemble."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import shutil

import tyro


@dataclass
class Args:
    member_dirs_file: str
    output_dir: str
    export_source_job_id: str = "102294"
    min_predicted_slot_files: int = 1000
    expected_predicted_slot_files: int = 1000
    predicted_slot_file_count: int = 1000
    min_train_gpus: int = 1
    min_train_seconds: int = 10800
    overwrite: bool = False


def _read_member_dirs(path: Path) -> list[Path]:
    members = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not members:
        raise SystemExit(f"No member dirs listed in {path}")
    for member in members:
        if not member.is_dir():
            raise SystemExit(f"Missing member dir: {member}")
        if not (member / "manifest.json").exists():
            raise SystemExit(f"Member has no manifest.json yet: {member}")
    return members


def main() -> None:
    args = tyro.cli(Args)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        existing_members = sorted(output_dir.glob("member_*"))
        if existing_members:
            raise SystemExit(f"{output_dir} already has member_* entries; pass --overwrite to replace them")
    output_dir.mkdir(parents=True, exist_ok=True)

    members = _read_member_dirs(Path(args.member_dirs_file))
    for old in output_dir.glob("member_*"):
        if old.is_symlink() or old.is_file():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
    for index, member in enumerate(members):
        target = output_dir / f"member_{index}"
        target.symlink_to(member.resolve())

    (output_dir / "member_dirs.txt").write_text("\n".join(str(member.resolve()) for member in members) + "\n")
    manifest_lines = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "job_id": "tmux_assembled_members",
        "node_list": "multiple_tmux_allocations",
        "output_dir": str(output_dir.resolve()),
        "member_dirs_file": str(Path(args.member_dirs_file).resolve()),
        "export_source_job_id": args.export_source_job_id,
        "requested_gres": "gpu:NVIDIAH200:1",
        "cluster_gpu_type_expected": "NVIDIAH200",
        "ntasks": "1",
        "min_train_gpus": str(args.min_train_gpus),
        "min_train_seconds": str(args.min_train_seconds),
        "min_predicted_slot_files": str(args.min_predicted_slot_files),
        "expected_predicted_slot_files": str(args.expected_predicted_slot_files),
        "predicted_slot_file_count": str(args.predicted_slot_file_count),
        "input_representation": "rgbd_predicted_slots",
        "world_model_input_group": "slots",
        "oracle_slots_not_used": "true",
        "oracle_slots_group": "inspection_only",
        "assembly_boundary": (
            "symlink_only_assembly_of_completed_rgbd_predicted_slot_world_model_members;"
            "does_not_change_training_metrics_or_controller_inputs"
        ),
    }
    (output_dir / "slurm_manifest.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in manifest_lines.items()) + "\n"
    )
    print(
        {
            "output_dir": str(output_dir),
            "num_members": len(members),
            "manifest": asdict(args),
        }
    )


if __name__ == "__main__":
    main()
