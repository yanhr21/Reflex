#!/usr/bin/env python3
"""Train a diagnostic scorer from live action-bank replay labels.

This is a calibration/debug tool. It learns executed action consequences from
real live simulator snapshots. It is not closed-loop method evidence by
itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cosmos3_live_receding_loop import require_compute_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-outcome-jsonl", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-action-steps", type=int, default=24)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every-steps", type=int, default=20)
    parser.add_argument("--error-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.5)
    parser.add_argument("--delta-loss-weight", type=float, default=0.5)
    parser.add_argument("--binary-loss-weight", type=float, default=0.25)
    parser.add_argument("--score-gate-weight", type=float, default=0.75)
    parser.add_argument("--score-success-weight", type=float, default=1.0)
    parser.add_argument("--score-grasped-weight", type=float, default=0.25)
    parser.add_argument("--score-contact-stable-weight", type=float, default=0.25)
    parser.add_argument("--score-delta-weight", type=float, default=1.0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_vec(value: Any, width: int) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    if arr.size < width:
        arr = np.pad(arr, (0, width - arr.size), mode="constant")
    return arr[:width].astype(np.float32)


def metric_values(metrics: dict[str, Any] | None) -> np.ndarray:
    metrics = metrics or {}
    keys = ("abs_x", "abs_y", "abs_z", "abs_yz_sum", "yz_l2", "weighted_abs_xyz")
    return np.asarray([float(metrics.get(key, 0.0) or 0.0) for key in keys], dtype=np.float32)


def gate_values(gate: dict[str, Any] | None) -> np.ndarray:
    gate = gate or {}
    checks = gate.get("checks") or {}
    keys = ("grasped", "hole_speed", "rel_x_max", "rel_x_min", "rel_y_abs", "rel_z_abs")
    return np.asarray(
        [float(bool(gate.get("ok", False)))] + [float(bool(checks.get(key, False))) for key in keys],
        dtype=np.float32,
    )


def candidate_descriptor(name: str, selected: bool) -> np.ndarray:
    raw_name = str(name or "")
    short_match = re.match(r"^short(?P<n>\d+)_(?P<rest>.*)$", raw_name)
    short_steps = float(short_match.group("n")) if short_match else 0.0
    name = short_match.group("rest") if short_match else raw_name
    is_dp = name == "dp_prior"
    is_mean = name == "mean"
    is_scale = name.startswith("scale_")
    is_sample = name.startswith("sample_t")
    scale_value = 0.0
    if is_scale:
        try:
            scale_value = float(name.rsplit("_", 1)[-1])
        except ValueError:
            scale_value = 0.0
    sample_temp = 0.0
    sample_index = -1.0
    if is_sample:
        match = re.search(r"sample_t(?P<temp>[-+0-9.eE]+)_(?P<idx>\d+)$", name)
        if match:
            sample_temp = float(match.group("temp"))
            sample_index = float(match.group("idx"))
    return np.asarray(
        [
            float(bool(selected)),
            float(is_dp),
            float(is_mean),
            float(is_scale),
            float(scale_value),
            float(is_sample),
            float(sample_temp),
            float(sample_index / 8.0 if sample_index >= 0.0 else 0.0),
            float(short_steps / 24.0),
        ],
        dtype=np.float32,
    )


def load_action(bank_path: Path, candidate_index: int, max_steps: int, action_dim: int) -> np.ndarray:
    bank = np.load(str(bank_path), allow_pickle=False)
    actions = np.asarray(bank["candidate_full_actions"], dtype=np.float32)
    action = actions[int(candidate_index), : int(max_steps), : int(action_dim)]
    if action.shape[0] < int(max_steps):
        pad = np.zeros((int(max_steps) - action.shape[0], int(action_dim)), dtype=np.float32)
        action = np.concatenate([action, pad], axis=0)
    return action.astype(np.float32)


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in paths:
        path = Path(text).resolve()
        for row in read_jsonl(path):
            if row.get("schema") != "cosmos3_live_snapshot_action_bank_outcome_label_v1":
                continue
            copied = dict(row)
            copied["_live_outcome_jsonl"] = str(path)
            rows.append(copied)
    if not rows:
        raise RuntimeError("no live action-bank replay labels found")
    return rows


def build_dataset(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    features: list[np.ndarray] = []
    error_targets: list[float] = []
    state_targets: list[np.ndarray] = []
    delta_targets: list[float] = []
    binary_targets: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    action_cache: dict[tuple[str, int], np.ndarray] = {}
    for row in rows:
        bank_path = Path(str(row["candidate_action_bank_npz"])).resolve()
        candidate_index = int(row["candidate_index"])
        cache_key = (str(bank_path), candidate_index)
        if cache_key not in action_cache:
            action_cache[cache_key] = load_action(
                bank_path,
                candidate_index,
                int(args.max_action_steps),
                int(args.robot_action_dim),
            )
        action = action_cache[cache_key]
        before_rel = as_vec(row.get("before_peg_head_at_hole"), 3)
        before_metrics = metric_values(row.get("before_rel_metrics"))
        before_gate = gate_values(row.get("before_continuability_gate"))
        desc = candidate_descriptor(
            str(row.get("candidate_name") or ""),
            bool(row.get("candidate_selected_by_live_scorer", False)),
        )
        action_mean = action.mean(axis=0)
        action_std = action.std(axis=0)
        feature = np.concatenate(
            [
                np.asarray(
                    [
                        float(row.get("prefix_frame_index", 0) or 0) / 300.0,
                        float(row.get("execute_steps_requested", 0) or 0) / max(1.0, float(args.max_action_steps)),
                    ],
                    dtype=np.float32,
                ),
                before_rel,
                before_metrics,
                before_gate,
                desc,
                action.reshape(-1),
                action[0],
                action[-1],
                action_mean,
                action_std,
            ]
        ).astype(np.float32)
        after_metrics = row.get("after_rel_metrics") or {}
        error = float(after_metrics.get("weighted_abs_xyz", 0.0) or 0.0)
        state = as_vec(row.get("after_peg_head_at_hole"), 3)
        delta = float(row.get("delta_abs_yz_sum", 0.0) or 0.0)
        gate_ok = bool((row.get("after_continuability_gate") or {}).get("ok", False))
        binary = np.asarray(
            [
                float(gate_ok),
                float(bool(row.get("after_success", False))),
                float(bool(row.get("after_grasped", False))),
                float(bool(row.get("after_inserted_live_pose", False))),
                float(bool(row.get("after_contact_stable_proxy", False))),
            ],
            dtype=np.float32,
        )
        features.append(feature)
        error_targets.append(error)
        state_targets.append(state)
        delta_targets.append(delta)
        binary_targets.append(binary)
        metas.append(
            {
                "group_id": str(row.get("iter_dir") or row.get("live_state_snapshot_before_controller") or ""),
                "sample_name": row.get("sample_name"),
                "scenario": row.get("scenario"),
                "iteration": row.get("iteration"),
                "prefix_frame_index": row.get("prefix_frame_index"),
                "candidate_name": row.get("candidate_name"),
                "candidate_index": candidate_index,
                "candidate_selected_by_live_scorer": bool(row.get("candidate_selected_by_live_scorer", False)),
                "true_after_gate_ok": gate_ok,
                "true_after_success": bool(row.get("after_success", False)),
                "true_after_grasped": bool(row.get("after_grasped", False)),
                "true_after_contact_stable_proxy": bool(row.get("after_contact_stable_proxy", False)),
                "true_after_weighted_error": error,
                "true_delta_abs_yz_sum": delta,
                "true_after_peg_head_at_hole": state.astype(float).tolist(),
                "live_outcome_jsonl": row.get("_live_outcome_jsonl"),
            }
        )
    widths = {item.shape[0] for item in features}
    if len(widths) != 1:
        raise RuntimeError(f"nonuniform feature widths: {sorted(widths)}")
    return {
        "x": np.stack(features).astype(np.float32),
        "error": np.asarray(error_targets, dtype=np.float32).reshape(-1, 1),
        "state": np.stack(state_targets).astype(np.float32),
        "delta": np.asarray(delta_targets, dtype=np.float32).reshape(-1, 1),
        "binary": np.stack(binary_targets).astype(np.float32),
        "meta": metas,
    }


def split_by_group(meta: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({str(item.get("group_id") or "") for item in meta})
    rng = random.Random(int(seed))
    rng.shuffle(groups)
    val_count = int(round(len(groups) * float(val_fraction)))
    if len(groups) > 1:
        val_count = max(1, min(len(groups) - 1, val_count))
    else:
        val_count = 0
    val_groups = set(groups[:val_count])
    train = [idx for idx, item in enumerate(meta) if str(item.get("group_id") or "") not in val_groups]
    val = [idx for idx, item in enumerate(meta) if str(item.get("group_id") or "") in val_groups]
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


class LiveConsequenceNet:
    def __new__(cls, feature_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Any:
        import torch

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[torch.nn.Module] = []
                in_dim = int(feature_dim)
                for _ in range(int(num_layers)):
                    layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
                    layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                    layers.append(torch.nn.GELU())
                    if dropout > 0:
                        layers.append(torch.nn.Dropout(float(dropout)))
                    in_dim = int(hidden_dim)
                self.encoder = torch.nn.Sequential(*layers)
                self.error_head = torch.nn.Linear(int(hidden_dim), 1)
                self.state_head = torch.nn.Linear(int(hidden_dim), 3)
                self.delta_head = torch.nn.Linear(int(hidden_dim), 1)
                self.binary_head = torch.nn.Linear(int(hidden_dim), 5)

            def forward(self, x: Any) -> tuple[Any, Any, Any, Any]:
                h = self.encoder(x)
                return self.error_head(h), self.state_head(h), self.delta_head(h), self.binary_head(h)

        return _Net()


def true_score(error: np.ndarray, delta: np.ndarray, binary: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return (
        -error.reshape(-1)
        - float(args.score_delta_weight) * delta.reshape(-1)
        + float(args.score_gate_weight) * binary[:, 0]
        + float(args.score_success_weight) * binary[:, 1]
        + float(args.score_grasped_weight) * binary[:, 2]
        + float(args.score_contact_stable_weight) * binary[:, 4]
    )


def predicted_score(pred_error: np.ndarray, pred_delta: np.ndarray, pred_binary: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return true_score(pred_error, pred_delta, pred_binary, args)


def is_dp_prior(name: str) -> bool:
    base = re.sub(r"^short\d+_", "", str(name or ""))
    return base == "dp_prior"


def evaluate(
    *,
    model: Any,
    x_norm: np.ndarray,
    error: np.ndarray,
    state: np.ndarray,
    delta: np.ndarray,
    binary: np.ndarray,
    meta: list[dict[str, Any]],
    indices: np.ndarray,
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if len(indices) == 0:
        return {"num_rows": 0, "num_groups": 0}
    module = model.module if hasattr(model, "module") else model
    module.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_norm[indices]).to(device)
        pred_error_t, pred_state_t, pred_delta_t, pred_logits = module(x)
        target_binary_t = torch.from_numpy(binary[indices]).to(device)
        pred_error = pred_error_t.detach().cpu().numpy()
        pred_state = pred_state_t.detach().cpu().numpy()
        pred_delta = pred_delta_t.detach().cpu().numpy()
        pred_binary = torch.sigmoid(pred_logits).detach().cpu().numpy()
        error_mse = float(np.mean((pred_error - error[indices]) ** 2))
        state_mse = float(np.mean((pred_state - state[indices]) ** 2))
        delta_mse = float(np.mean((pred_delta - delta[indices]) ** 2))
        binary_bce = float(F.binary_cross_entropy_with_logits(pred_logits, target_binary_t).item())

    groups: dict[str, list[int]] = defaultdict(list)
    for local_i, global_i in enumerate(indices):
        groups[str(meta[int(global_i)].get("group_id") or "")].append(local_i)

    scores = predicted_score(pred_error, pred_delta, pred_binary, args)
    target_scores = true_score(error[indices], delta[indices], binary[indices], args)
    selected_scores: list[float] = []
    live_selected_scores: list[float] = []
    dp_scores: list[float] = []
    oracle_scores: list[float] = []
    selected_gate: list[float] = []
    live_selected_gate: list[float] = []
    dp_gate: list[float] = []
    oracle_gate: list[float] = []
    selected_names: list[str] = []
    oracle_names: list[str] = []
    group_count = 0
    top1_oracle_match = 0
    for local_list in groups.values():
        if not local_list:
            continue
        dp_local = None
        live_local = None
        for local_i in local_list:
            m = meta[int(indices[local_i])]
            if dp_local is None and is_dp_prior(str(m.get("candidate_name") or "")):
                dp_local = local_i
            if live_local is None and bool(m.get("candidate_selected_by_live_scorer", False)):
                live_local = local_i
        if dp_local is None:
            continue
        if live_local is None:
            live_local = dp_local
        group_count += 1
        best_local = local_list[int(np.argmax(scores[local_list]))]
        oracle_local = local_list[int(np.argmax(target_scores[local_list]))]
        selected_scores.append(float(target_scores[best_local]))
        live_selected_scores.append(float(target_scores[live_local]))
        dp_scores.append(float(target_scores[dp_local]))
        oracle_scores.append(float(target_scores[oracle_local]))
        selected_gate.append(float(binary[indices[best_local], 0]))
        live_selected_gate.append(float(binary[indices[live_local], 0]))
        dp_gate.append(float(binary[indices[dp_local], 0]))
        oracle_gate.append(float(binary[indices[oracle_local], 0]))
        selected_names.append(str(meta[int(indices[best_local])].get("candidate_name") or ""))
        oracle_names.append(str(meta[int(indices[oracle_local])].get("candidate_name") or ""))
        if best_local == oracle_local:
            top1_oracle_match += 1
    selected_arr = np.asarray(selected_scores, dtype=np.float32)
    live_arr = np.asarray(live_selected_scores, dtype=np.float32)
    dp_arr = np.asarray(dp_scores, dtype=np.float32)
    oracle_arr = np.asarray(oracle_scores, dtype=np.float32)
    out = {
        "num_rows": int(len(indices)),
        "num_groups": int(group_count),
        "error_mse": error_mse,
        "state_mse": state_mse,
        "delta_mse": delta_mse,
        "binary_bce": binary_bce,
        "selected_true_score_mean": float(selected_arr.mean()) if selected_arr.size else None,
        "live_selected_true_score_mean": float(live_arr.mean()) if live_arr.size else None,
        "dp_prior_true_score_mean": float(dp_arr.mean()) if dp_arr.size else None,
        "oracle_true_score_mean": float(oracle_arr.mean()) if oracle_arr.size else None,
        "selected_minus_dp_true_score_mean": float((selected_arr - dp_arr).mean()) if selected_arr.size else None,
        "selected_minus_live_selected_true_score_mean": float((selected_arr - live_arr).mean()) if selected_arr.size else None,
        "oracle_minus_dp_true_score_mean": float((oracle_arr - dp_arr).mean()) if oracle_arr.size else None,
        "selected_gate_ok_count": int(sum(selected_gate)),
        "live_selected_gate_ok_count": int(sum(live_selected_gate)),
        "dp_prior_gate_ok_count": int(sum(dp_gate)),
        "oracle_gate_ok_count": int(sum(oracle_gate)),
        "top1_oracle_match_fraction": float(top1_oracle_match / max(1, group_count)),
        "selected_candidate_counts": dict(sorted(Counter(selected_names).items())),
        "oracle_candidate_counts": dict(sorted(Counter(oracle_names).items())),
    }
    module.train()
    return out


def main() -> int:
    args = parse_args()
    require_compute_step()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if bool(args.require_cuda) and not torch.cuda.is_available():
        raise SystemExit("require_cuda=true but torch.cuda.is_available() is false")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rows = load_rows(args.live_outcome_jsonl)
    data = build_dataset(rows, args)
    train_idx, val_idx = split_by_group(data["meta"], float(args.val_fraction), int(args.seed))
    x = data["x"]
    x_mean = x[train_idx].mean(axis=0, keepdims=True)
    x_std = x[train_idx].std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x_norm = ((x - x_mean) / x_std).astype(np.float32)

    model = LiveConsequenceNet(
        feature_dim=x_norm.shape[1],
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_ds = TensorDataset(
        torch.from_numpy(x_norm[train_idx]),
        torch.from_numpy(data["error"][train_idx]),
        torch.from_numpy(data["state"][train_idx]),
        torch.from_numpy(data["delta"][train_idx]),
        torch.from_numpy(data["binary"][train_idx]),
    )
    loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_payload: dict[str, Any] | None = None
    iterator = iter(loader)
    for step in range(1, int(args.max_steps) + 1):
        try:
            xb, eb, sb, db, bb = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, eb, sb, db, bb = next(iterator)
        xb = xb.to(device)
        eb = eb.to(device)
        sb = sb.to(device)
        db = db.to(device)
        bb = bb.to(device)
        pred_error, pred_state, pred_delta, pred_logits = model(xb)
        loss = (
            float(args.error_loss_weight) * F.mse_loss(pred_error, eb)
            + float(args.state_loss_weight) * F.mse_loss(pred_state, sb)
            + float(args.delta_loss_weight) * F.mse_loss(pred_delta, db)
            + float(args.binary_loss_weight) * F.binary_cross_entropy_with_logits(pred_logits, bb)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % int(args.eval_every_steps) == 0 or step == int(args.max_steps):
            train_metrics = evaluate(
                model=model,
                x_norm=x_norm,
                error=data["error"],
                state=data["state"],
                delta=data["delta"],
                binary=data["binary"],
                meta=data["meta"],
                indices=train_idx,
                args=args,
                device=device,
            )
            val_metrics = evaluate(
                model=model,
                x_norm=x_norm,
                error=data["error"],
                state=data["state"],
                delta=data["delta"],
                binary=data["binary"],
                meta=data["meta"],
                indices=val_idx,
                args=args,
                device=device,
            )
            item = {
                "step": int(step),
                "loss": float(loss.detach().cpu().item()),
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(item)
            val_score = float(val_metrics.get("selected_minus_dp_true_score_mean") or float("-inf"))
            if val_metrics.get("num_groups", 0) and val_score > best_val_score:
                best_val_score = val_score
                best_payload = {
                    "schema": "cosmos3_live_action_consequence_scorer_checkpoint_v1",
                    "model_state_dict": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "args": vars(args),
                    "feature_dim": int(x_norm.shape[1]),
                    "target_binary_names": [
                        "after_gate_ok",
                        "after_success",
                        "after_grasped",
                        "after_inserted_live_pose",
                        "after_contact_stable_proxy",
                    ],
                    "best_step": int(step),
                    "best_val_metrics": val_metrics,
                    "boundary": (
                        "Diagnostic live action-consequence scorer. It is trained "
                        "from replay labels and is not closed-loop method evidence."
                    ),
                }
                torch.save(best_payload, output_root / "checkpoint_best.pt")
            write_json(output_root / "training_history.json", history)

    if best_payload is None:
        best_payload = {
            "schema": "cosmos3_live_action_consequence_scorer_checkpoint_v1",
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "args": vars(args),
            "feature_dim": int(x_norm.shape[1]),
            "best_step": int(args.max_steps),
            "best_val_metrics": {},
            "boundary": "Diagnostic checkpoint only; no validation groups were available.",
        }
        torch.save(best_payload, output_root / "checkpoint_best.pt")
    torch.save(best_payload, output_root / "checkpoint_final.pt")
    summary = {
        "schema": "cosmos3_live_action_consequence_scorer_training_summary_v1",
        "output_root": str(output_root),
        "live_outcome_jsonl": args.live_outcome_jsonl,
        "num_rows": int(x.shape[0]),
        "num_groups": int(len({str(item.get("group_id") or "") for item in data["meta"]})),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "feature_dim": int(x_norm.shape[1]),
        "max_steps": int(args.max_steps),
        "device": str(device),
        "best_val_score": best_val_score if np.isfinite(best_val_score) else None,
        "latest": history[-1] if history else None,
        "boundary": (
            "This is a live action-consequence calibration sanity run. It is "
            "not closed-loop method evidence and should not be reported as task success."
        ),
    }
    write_json(output_root / "training_summary.json", summary)
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
