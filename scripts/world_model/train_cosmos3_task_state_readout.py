#!/usr/bin/env python3
"""Train a controller-facing task-state readout for Cosmos3 videos.

This is not a replacement world model. Cosmos3 remains the future video/latent
predictor; this head decodes a Cosmos/reference video frame into the task-state
quantities that the rebinding controller already needs.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tyro


TARGET_CONT_NAMES = (
    "hole_pos_x",
    "hole_pos_y",
    "hole_pos_z",
    "hole_quat_w",
    "hole_quat_x",
    "hole_quat_y",
    "hole_quat_z",
    "peg_pos_x",
    "peg_pos_y",
    "peg_pos_z",
    "peg_quat_w",
    "peg_quat_x",
    "peg_quat_y",
    "peg_quat_z",
    "tcp_pos_x",
    "tcp_pos_y",
    "tcp_pos_z",
    "tcp_quat_w",
    "tcp_quat_x",
    "tcp_quat_y",
    "tcp_quat_z",
    "peg_head_hole_x",
    "peg_head_hole_y",
    "peg_head_hole_z",
    "hole_radius",
)
TARGET_BIN_NAMES = ("grasped", "inserted")


@dataclass
class Args:
    dataset_manifest: str
    output_dir: str
    checkpoint_path: str | None = None
    predict_video: str | None = None
    predict_output_json: str | None = None
    predict_reference_h5: str | None = None
    predict_reference_start_frame: int = 0
    num_frames: int = 301
    future_start_frame: int = 29
    image_size: int = 160
    max_train_trajectories: int = 0
    max_val_trajectories: int = 0
    steps: int = 2000
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    cnn_channels: int = 48
    bin_loss_weight: float = 0.2
    hole_loss_weight: float = 1.0
    peg_loss_weight: float = 1.0
    tcp_loss_weight: float = 1.0
    peg_head_hole_loss_weight: float = 1.0
    seed: int = 7
    cuda: bool = True
    require_cuda: bool = True
    require_exact_video_frames: bool = True
    num_workers: int = 0
    video_cache_size: int = 4
    frame_mode: bool = False
    log_every: int = 50
    eval_every: int = 250
    max_eval_batches: int = 40


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    @classmethod
    def from_targets(cls, targets: np.ndarray) -> "Normalizer":
        mean = targets.mean(axis=0).astype(np.float32)
        std = np.maximum(targets.std(axis=0), 1e-6).astype(np.float32)
        return cls(mean=mean.tolist(), std=std.tolist())

    def normalize(self, array: np.ndarray) -> np.ndarray:
        return ((array - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)).astype(
            np.float32
        )

    def denormalize_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        return tensor * std.view(1, -1) + mean.view(1, -1)


class FrameReadout(nn.Module):
    def __init__(self, cont_dim: int, bin_dim: int, cnn_channels: int, hidden_dim: int):
        super().__init__()
        c = int(cnn_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, c, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.GELU(),
            nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        feat_dim = c * 4 * 4 * 4
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cont_head = nn.Linear(hidden_dim, cont_dim)
        self.bin_head = nn.Linear(hidden_dim, bin_dim)

    def _coord_channels(self, image: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        y = torch.linspace(-1.0, 1.0, height, dtype=image.dtype, device=image.device)
        x = torch.linspace(-1.0, 1.0, width, dtype=image.dtype, device=image.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return coords.expand(batch, -1, -1, -1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(torch.cat((image, self._coord_channels(image)), dim=1))
        z = self.head(z)
        return self.cont_head(z), self.bin_head(z)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    rows = list(manifest.get("videos", []))
    if not rows:
        raise ValueError(f"{path} has no videos")
    return rows


def _trajectory_name(h5: h5py.File, expected: str | None = None) -> str:
    names = sorted([key for key in h5.keys() if key.startswith("traj_")])
    if expected and expected in h5:
        return expected
    if len(names) != 1:
        raise ValueError(f"expected exactly one trajectory, found {names}")
    return names[0]


def _slot_targets(h5_path: Path, trajectory: str | None, frame_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        group = h5[_trajectory_name(h5, trajectory)]
        slots = group["slots"]
        frame_count = int(slots["hole_pose"].shape[0])
        bad_indices = [int(raw_idx) for raw_idx in frame_indices if int(raw_idx) < 0 or int(raw_idx) >= frame_count]
        if bad_indices:
            raise ValueError(
                "slot target frame range mismatch: "
                f"h5={h5_path}, trajectory={trajectory}, available_frames={frame_count}, "
                f"requested_min={min(frame_indices)}, requested_max={max(frame_indices)}, "
                f"bad_examples={bad_indices[:5]}"
            )
        cont_rows = []
        bin_rows = []
        for raw_idx in frame_indices:
            idx = int(raw_idx)
            hole_pose = np.asarray(slots["hole_pose"][idx], dtype=np.float32).reshape(-1)[:7]
            peg_pose = np.asarray(slots["peg_pose"][idx], dtype=np.float32).reshape(-1)[:7]
            tcp_pose = np.asarray(slots["tcp_pose"][idx], dtype=np.float32).reshape(-1)[:7]
            peg_head_hole = np.asarray(slots["peg_head_at_hole"][idx], dtype=np.float32).reshape(-1)[:3]
            hole_radius = np.asarray([slots["hole_radius"][idx]], dtype=np.float32)
            cont_rows.append(np.concatenate([hole_pose, peg_pose, tcp_pose, peg_head_hole, hole_radius]))
            bin_rows.append(
                np.asarray([float(bool(slots["grasped"][idx])), float(bool(slots["inserted"][idx]))], dtype=np.float32)
            )
    return np.stack(cont_rows).astype(np.float32), np.stack(bin_rows).astype(np.float32)


def _slot_frame_count(h5_path: Path, trajectory: str | None) -> int:
    with h5py.File(h5_path, "r") as h5:
        group = h5[_trajectory_name(h5, trajectory)]
        return int(group["slots"]["hole_pose"].shape[0])


def _resize_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    frame = frame[:, :, :3]
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _read_video_frames(
    video_path: Path,
    num_frames: int,
    image_size: int,
    require_exact_frames: bool = True,
) -> np.ndarray:
    reader = imageio.get_reader(video_path)
    frames = []
    last = None
    try:
        for idx, frame in enumerate(reader):
            if idx >= num_frames:
                if require_exact_frames:
                    raise ValueError(
                        f"video has more frames than expected: {video_path}, "
                        f"expected={num_frames}, decoded_at_least={idx + 1}"
                    )
                break
            last = frame
            frames.append(_resize_rgb(frame, image_size))
        if not frames:
            raise ValueError(f"no frames decoded from {video_path}")
        if require_exact_frames and len(frames) != num_frames:
            raise ValueError(
                f"video frame count mismatch: {video_path}, decoded={len(frames)}, expected={num_frames}"
            )
        while len(frames) < num_frames:
            frames.append(_resize_rgb(last, image_size))
    finally:
        reader.close()
    return np.stack(frames).transpose(0, 3, 1, 2).astype(np.float32)


class CosmosTaskStateClipDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        normalizer: Normalizer,
        num_frames: int,
        image_size: int,
        cache_size: int,
        require_exact_video_frames: bool,
    ):
        self.rows = rows
        self.normalizer = normalizer
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.cache_size = max(0, int(cache_size))
        self.require_exact_video_frames = bool(require_exact_video_frames)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _frames(self, idx: int) -> np.ndarray:
        if self.cache_size > 0 and idx in self._cache:
            frames = self._cache.pop(idx)
            self._cache[idx] = frames
            return frames
        frames = _read_video_frames(
            Path(self.rows[idx]["video"]),
            self.num_frames,
            self.image_size,
            self.require_exact_video_frames,
        )
        if self.cache_size > 0:
            self._cache[idx] = frames
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        frame_indices = list(range(self.num_frames))
        cont, bins = _slot_targets(Path(row["input_h5"]), row.get("trajectory"), frame_indices)
        cont_norm = self.normalizer.normalize(cont)
        return {
            "image": torch.from_numpy(self._frames(idx)),
            "target_cont": torch.from_numpy(cont_norm),
            "target_cont_raw": torch.from_numpy(cont),
            "target_bin": torch.from_numpy(bins),
        }


class CosmosTaskStateFrameDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        normalizer: Normalizer,
        num_frames: int,
        image_size: int,
        cache_size: int,
        require_exact_video_frames: bool,
    ):
        self.rows = rows
        self.normalizer = normalizer
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.cache_size = max(0, int(cache_size))
        self.require_exact_video_frames = bool(require_exact_video_frames)
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._target_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows) * self.num_frames

    def _row_frame(self, idx: int) -> tuple[int, int]:
        row_idx = int(idx) // self.num_frames
        frame_idx = int(idx) % self.num_frames
        return row_idx, frame_idx

    def _frames(self, row_idx: int) -> np.ndarray:
        if self.cache_size > 0 and row_idx in self._frame_cache:
            frames = self._frame_cache.pop(row_idx)
            self._frame_cache[row_idx] = frames
            return frames
        frames = _read_video_frames(
            Path(self.rows[row_idx]["video"]),
            self.num_frames,
            self.image_size,
            self.require_exact_video_frames,
        )
        if self.cache_size > 0:
            self._frame_cache[row_idx] = frames
            while len(self._frame_cache) > self.cache_size:
                self._frame_cache.popitem(last=False)
        return frames

    def _targets(self, row_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cache_size > 0 and row_idx in self._target_cache:
            targets = self._target_cache.pop(row_idx)
            self._target_cache[row_idx] = targets
            return targets
        cont, bins = _slot_targets(Path(self.rows[row_idx]["input_h5"]), self.rows[row_idx].get("trajectory"), list(range(self.num_frames)))
        cont_norm = self.normalizer.normalize(cont)
        targets = (cont_norm, cont, bins)
        if self.cache_size > 0:
            self._target_cache[row_idx] = targets
            while len(self._target_cache) > self.cache_size:
                self._target_cache.popitem(last=False)
        return targets

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row_idx, frame_idx = self._row_frame(idx)
        cont_norm, cont_raw, bins = self._targets(row_idx)
        return {
            "image": torch.from_numpy(self._frames(row_idx)[frame_idx]),
            "target_cont": torch.from_numpy(cont_norm[frame_idx]),
            "target_cont_raw": torch.from_numpy(cont_raw[frame_idx]),
            "target_bin": torch.from_numpy(bins[frame_idx]),
            "frame_id": torch.as_tensor(frame_idx, dtype=torch.long),
        }


def _limit(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def _target_stats(rows: list[dict[str, Any]], num_frames: int) -> Normalizer:
    arrays = []
    frame_indices = list(range(num_frames))
    for row in rows:
        cont, _ = _slot_targets(Path(row["input_h5"]), row.get("trajectory"), frame_indices)
        arrays.append(cont)
    return Normalizer.from_targets(np.concatenate(arrays, axis=0))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _flatten_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    image = batch["image"].to(device=device, dtype=torch.float32)
    target_cont = batch["target_cont"].to(device=device, dtype=torch.float32)
    target_cont_raw = batch["target_cont_raw"].to(device=device, dtype=torch.float32)
    target_bin = batch["target_bin"].to(device=device, dtype=torch.float32)
    if image.ndim == 5:
        batch_size, frames, channels, height, width = image.shape
        image = image.view(batch_size * frames, channels, height, width)
        frame_ids = torch.arange(frames, device=device).repeat(batch_size)
        return (
            image,
            target_cont.view(batch_size * frames, -1),
            target_cont_raw.view(batch_size * frames, -1),
            target_bin.view(batch_size * frames, -1),
            frame_ids,
        )
    if image.ndim == 4:
        frame_ids = batch["frame_id"].to(device=device, dtype=torch.long)
        return image, target_cont.view(image.shape[0], -1), target_cont_raw.view(image.shape[0], -1), target_bin.view(
            image.shape[0], -1
        ), frame_ids
    raise ValueError(f"unexpected image batch shape {tuple(image.shape)}")


def _continuous_loss_weights(args: Args, device: torch.device) -> torch.Tensor:
    weights = torch.ones(len(TARGET_CONT_NAMES), dtype=torch.float32, device=device)
    weights[0:3] = float(args.hole_loss_weight)
    weights[7:10] = float(args.peg_loss_weight)
    weights[14:17] = float(args.tcp_loss_weight)
    weights[21:24] = float(args.peg_head_hole_loss_weight)
    return weights


def _rmse(pred: torch.Tensor, target: torch.Tensor, cols: slice) -> float:
    diff = pred[:, cols] - target[:, cols]
    return float(torch.sqrt(torch.mean(torch.sum(diff * diff, dim=1))).detach().cpu())


@torch.no_grad()
def evaluate(
    model: FrameReadout,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    max_batches: int,
    future_start_frame: int,
    num_frames: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    pred_cont_all = []
    target_cont_all = []
    pred_bin_all = []
    target_bin_all = []
    frame_ids_all = []
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        image, target_cont, target_cont_raw, target_bin, frame_ids = _flatten_batch(batch, device)
        pred_cont_norm, pred_bin_logits = model(image)
        loss = F.mse_loss(pred_cont_norm, target_cont, reduction="sum")
        total_loss += float(loss.detach().cpu())
        total_items += int(target_cont.shape[0])
        pred_cont_all.append(normalizer.denormalize_torch(pred_cont_norm).detach().cpu())
        target_cont_all.append(target_cont_raw.detach().cpu())
        pred_bin_all.append(torch.sigmoid(pred_bin_logits).detach().cpu())
        target_bin_all.append(target_bin.detach().cpu())
        frame_ids_all.append(frame_ids.detach().cpu())
    pred_cont = torch.cat(pred_cont_all, dim=0)
    target_cont = torch.cat(target_cont_all, dim=0)
    pred_bin = torch.cat(pred_bin_all, dim=0)
    target_bin = torch.cat(target_bin_all, dim=0)
    frame_ids = torch.cat(frame_ids_all, dim=0)
    future_mask = frame_ids >= int(future_start_frame)

    def metrics_for(mask: torch.Tensor) -> dict[str, Any]:
        pred_c = pred_cont[mask]
        target_c = target_cont[mask]
        pred_b = pred_bin[mask]
        target_b = target_bin[mask]
        bin_pred = (pred_b >= 0.5).float()
        return {
            "num_frames": int(mask.sum().item()),
            "hole_pos_rmse_m": _rmse(pred_c, target_c, slice(0, 3)),
            "peg_pos_rmse_m": _rmse(pred_c, target_c, slice(7, 10)),
            "tcp_pos_rmse_m": _rmse(pred_c, target_c, slice(14, 17)),
            "peg_head_hole_rmse_m": _rmse(pred_c, target_c, slice(21, 24)),
            "grasped_accuracy": float((bin_pred[:, 0] == target_b[:, 0]).float().mean().item()),
            "inserted_accuracy": float((bin_pred[:, 1] == target_b[:, 1]).float().mean().item()),
        }

    return {
        "loss_cont_norm_mean": total_loss / max(1, total_items),
        "all": metrics_for(torch.ones_like(future_mask, dtype=torch.bool)),
        "future": metrics_for(future_mask),
    }


@torch.no_grad()
def predict_video(
    model: FrameReadout,
    normalizer: Normalizer,
    video_path: Path,
    output_json: Path,
    device: torch.device,
    num_frames: int,
    image_size: int,
    reference_h5: Path | None,
    reference_start_frame: int = 0,
    require_exact_video_frames: bool = True,
) -> dict[str, Any]:
    model.eval()
    frames = torch.from_numpy(
        _read_video_frames(video_path, num_frames, image_size, require_exact_video_frames)
    ).to(device=device)
    pred_norm, pred_bin_logits = model(frames)
    pred_cont = normalizer.denormalize_torch(pred_norm).detach().cpu().numpy()
    pred_bin = torch.sigmoid(pred_bin_logits).detach().cpu().numpy()
    report: dict[str, Any] = {
        "video_path": str(video_path),
        "num_frames": int(num_frames),
        "target_cont_names": TARGET_CONT_NAMES,
        "target_bin_names": TARGET_BIN_NAMES,
        "pred_cont": pred_cont.tolist(),
        "pred_bin_probability": pred_bin.tolist(),
        "reference_start_frame": int(reference_start_frame),
        "require_exact_video_frames": bool(require_exact_video_frames),
        "boundary": (
            "Cosmos video task-state readout prediction. This is controller-facing "
            "readout evidence only when paired with a Cosmos-generated video/latent; "
            "it is not task success by itself."
        ),
    }
    if reference_h5 is not None:
        if int(reference_start_frame) < 0:
            raise ValueError(f"reference_start_frame must be non-negative, got {reference_start_frame}")
        reference_frame_count = _slot_frame_count(reference_h5, None)
        reference_end_exclusive = int(reference_start_frame) + int(num_frames)
        if reference_end_exclusive > reference_frame_count:
            raise ValueError(
                "reference H5 frame range mismatch for readout prediction: "
                f"reference_start={reference_start_frame}, num_frames={num_frames}, "
                f"required_end={reference_end_exclusive}, available={reference_frame_count}, "
                f"h5={reference_h5}"
            )
        target_cont, target_bin = _slot_targets(
            reference_h5,
            None,
            list(range(int(reference_start_frame), int(reference_start_frame) + int(num_frames))),
        )
        pred_t = torch.from_numpy(pred_cont)
        target_t = torch.from_numpy(target_cont)
        pred_b = torch.from_numpy(pred_bin)
        target_b = torch.from_numpy(target_bin)
        report["reference_h5"] = str(reference_h5)
        report["reference_frame_count"] = int(reference_frame_count)
        report["reference_end_exclusive"] = int(reference_end_exclusive)
        report["reference_metrics"] = {
            "hole_pos_rmse_m": _rmse(pred_t, target_t, slice(0, 3)),
            "peg_pos_rmse_m": _rmse(pred_t, target_t, slice(7, 10)),
            "tcp_pos_rmse_m": _rmse(pred_t, target_t, slice(14, 17)),
            "peg_head_hole_rmse_m": _rmse(pred_t, target_t, slice(21, 24)),
            "grasped_accuracy": float(((pred_b[:, 0] >= 0.5) == (target_b[:, 0] >= 0.5)).float().mean().item()),
            "inserted_accuracy": float(((pred_b[:, 1] >= 0.5) == (target_b[:, 1] >= 0.5)).float().mean().item()),
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    return report


def _load_checkpoint(path: Path, device: torch.device) -> tuple[FrameReadout, Normalizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    normalizer = Normalizer(**checkpoint["normalizer"])
    cfg = checkpoint["model_config"]
    model = FrameReadout(
        cont_dim=len(TARGET_CONT_NAMES),
        bin_dim=len(TARGET_BIN_NAMES),
        cnn_channels=int(cfg["cnn_channels"]),
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, normalizer, checkpoint


def main() -> None:
    args = tyro.cli(Args)
    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = bool(args.cuda and torch.cuda.is_available())
    if args.require_cuda and not use_cuda:
        raise SystemExit("CUDA is required for this training run but is not available.")
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.checkpoint_path:
        model, normalizer, checkpoint = _load_checkpoint(Path(args.checkpoint_path), device)
        manifest = dict(checkpoint.get("manifest", {}))
    else:
        rows = _read_manifest(Path(args.dataset_manifest))
        train_rows = _limit([row for row in rows if row.get("split") == "train"], args.max_train_trajectories)
        val_rows = _limit([row for row in rows if row.get("split") == "val"], args.max_val_trajectories)
        if not train_rows or not val_rows:
            raise ValueError(f"need train and val rows, got train={len(train_rows)} val={len(val_rows)}")
        normalizer = _target_stats(train_rows, args.num_frames)
        dataset_cls = CosmosTaskStateFrameDataset if args.frame_mode else CosmosTaskStateClipDataset
        train_ds = dataset_cls(
            train_rows,
            normalizer,
            args.num_frames,
            args.image_size,
            args.video_cache_size,
            args.require_exact_video_frames,
        )
        val_ds = dataset_cls(
            val_rows,
            normalizer,
            args.num_frames,
            args.image_size,
            args.video_cache_size,
            args.require_exact_video_frames,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=not args.frame_mode,
            num_workers=args.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        model = FrameReadout(
            cont_dim=len(TARGET_CONT_NAMES),
            bin_dim=len(TARGET_BIN_NAMES),
            cnn_channels=args.cnn_channels,
            hidden_dim=args.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        manifest = {
            "args": asdict(args),
            "dataset_manifest": str(Path(args.dataset_manifest)),
            "num_train_trajectories": len(train_rows),
            "num_val_trajectories": len(val_rows),
            "frame_mode": bool(args.frame_mode),
            "target_cont_names": TARGET_CONT_NAMES,
            "target_bin_names": TARGET_BIN_NAMES,
            "physical_reason": (
                "Decode Cosmos3 predicted/reference video frames into hole, peg, TCP, "
                "peg-head-in-hole, grasp, and insertion task state so the rebinding "
                "controller can consume the foundation world model output."
            ),
            "method_boundary": (
                "This readout is not a world model and must not be reported as "
                "controller or task-completion evidence by itself. It is valid only "
                "as the controller-facing decoder on top of Cosmos3 video/latent "
                "prediction."
            ),
        }
        (output_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")

        iterator = iter(train_loader)
        cont_loss_weights = _continuous_loss_weights(args, device)
        best_future = float("inf")
        latest_metrics: dict[str, Any] | None = None
        for step in range(1, int(args.steps) + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            model.train()
            image, target_cont, _, target_bin, _ = _flatten_batch(batch, device)
            pred_cont, pred_bin = model(image)
            loss_cont = torch.mean(
                (pred_cont - target_cont) ** 2 * cont_loss_weights.reshape(1, -1)
            )
            loss_bin = F.binary_cross_entropy_with_logits(pred_bin, target_bin)
            loss = loss_cont + float(args.bin_loss_weight) * loss_bin
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % max(1, args.log_every) == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.detach().cpu()),
                            "loss_cont": float(loss_cont.detach().cpu()),
                            "loss_bin": float(loss_bin.detach().cpu()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if step % max(1, args.eval_every) == 0 or step == int(args.steps):
                latest_metrics = evaluate(
                    model, val_loader, normalizer, device, args.max_eval_batches, args.future_start_frame, args.num_frames
                )
                latest_metrics["step"] = step
                (output_dir / "metrics_latest.json").write_text(
                    json.dumps(_jsonable(latest_metrics), indent=2, sort_keys=True) + "\n"
                )
                future_rmse = float(latest_metrics["future"]["peg_head_hole_rmse_m"])
                print(json.dumps({"event": "eval", **latest_metrics}, sort_keys=True), flush=True)
                checkpoint = {
                    "model": model.state_dict(),
                    "normalizer": asdict(normalizer),
                    "model_config": {
                        "cnn_channels": args.cnn_channels,
                        "hidden_dim": args.hidden_dim,
                        "image_size": args.image_size,
                    },
                    "manifest": manifest,
                    "metrics": latest_metrics,
                }
                torch.save(checkpoint, output_dir / "model_latest.pt")
                if future_rmse < best_future:
                    best_future = future_rmse
                    torch.save(checkpoint, output_dir / "best_model.pt")
                    (output_dir / "metrics_best.json").write_text(
                        json.dumps(_jsonable(latest_metrics), indent=2, sort_keys=True) + "\n"
                    )
        if latest_metrics is None:
            latest_metrics = evaluate(
                model, val_loader, normalizer, device, args.max_eval_batches, args.future_start_frame, args.num_frames
            )
            (output_dir / "metrics_latest.json").write_text(
                json.dumps(_jsonable(latest_metrics), indent=2, sort_keys=True) + "\n"
            )
        (output_dir / "readout_completed").write_text(
            json.dumps(
                {
                    "timestamp_note": "written after task-state readout training loop exits",
                    "steps": int(args.steps),
                    "best_model_exists": (output_dir / "best_model.pt").exists(),
                    "metrics_latest_exists": (output_dir / "metrics_latest.json").exists(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    if args.predict_video:
        if not args.predict_output_json:
            raise ValueError("--predict-output-json is required when --predict-video is set")
        report = predict_video(
            model=model,
            normalizer=normalizer,
            video_path=Path(args.predict_video),
            output_json=Path(args.predict_output_json),
            device=device,
            num_frames=args.num_frames,
            image_size=args.image_size,
            reference_h5=Path(args.predict_reference_h5) if args.predict_reference_h5 else None,
            reference_start_frame=args.predict_reference_start_frame,
            require_exact_video_frames=args.require_exact_video_frames,
        )
        print(json.dumps({"event": "predict_video", "output_json": args.predict_output_json, **report}, sort_keys=True))


if __name__ == "__main__":
    main()
