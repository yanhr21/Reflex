#!/usr/bin/env python3
"""Two-sample supervised overfit for the Cosmos3 executor interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--dp-prior-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_training_arrays(
    executor_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    prior_by_uuid = {str(row["uuid"]): row for row in prior_rows}
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    used_rows: list[dict[str, Any]] = []
    for row in executor_rows:
        if max_samples > 0 and len(features) >= max_samples:
            break
        uuid = str(row["uuid"])
        prior_row = prior_by_uuid.get(uuid)
        if prior_row is None:
            continue
        executor_npz = np.load(str(row["sample_npz"]), allow_pickle=False)
        prior_npz = np.load(str(prior_row["dp_prior_npz"]), allow_pickle=False)
        current = executor_npz["current_state"].astype(np.float32).reshape(-1)
        task_path = executor_npz["task_path"].astype(np.float32)
        teacher = executor_npz["teacher_robot_actions"].astype(np.float32)
        prior = prior_npz["dp_prior_actions"].astype(np.float32)
        horizon = min(int(task_path.shape[0]), int(teacher.shape[0]), int(prior.shape[0]))
        if horizon <= 0:
            continue
        task_path = task_path[:horizon]
        teacher = teacher[:horizon]
        prior = prior[:horizon]
        feat = np.concatenate([current, task_path.reshape(-1), prior.reshape(-1)]).astype(np.float32)
        features.append(feat)
        targets.append(teacher.reshape(-1).astype(np.float32))
        priors.append(prior.reshape(-1).astype(np.float32))
        used_rows.append(
            {
                "uuid": uuid,
                "scenario": row.get("scenario"),
                "prefix_role": row.get("prefix_role"),
                "horizon": horizon,
                "task_path_source": row.get("task_path_source"),
                "executor_sample_npz": row.get("sample_npz"),
                "dp_prior_npz": prior_row.get("dp_prior_npz"),
            }
        )
    if not features:
        raise RuntimeError("no matched executor/DP-prior samples")
    feature_widths = {item.shape[0] for item in features}
    target_widths = {item.shape[0] for item in targets}
    if len(feature_widths) != 1 or len(target_widths) != 1:
        raise RuntimeError(f"nonuniform feature/target widths: {feature_widths} {target_widths}")
    return np.stack(features), np.stack(targets), np.stack(priors), used_rows


def main() -> int:
    args = parse_args()
    require_compute_step()

    import torch

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    executor_rows = read_jsonl(Path(args.executor_jsonl).resolve())
    prior_rows = read_jsonl(Path(args.dp_prior_jsonl).resolve())
    x_np, y_np, prior_np, used_rows = load_training_arrays(executor_rows, prior_rows, int(args.max_samples))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    prior = torch.as_tensor(prior_np, dtype=torch.float32, device=device)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], int(args.hidden_dim)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(args.hidden_dim), int(args.hidden_dim)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(args.hidden_dim), y.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-6)

    with torch.no_grad():
        baseline_mse = torch.mean((prior - y) ** 2).item()
        initial_mse = torch.mean((prior + model(x) - y) ** 2).item()
    history: list[dict[str, float | int]] = []
    for step in range(1, int(args.max_steps) + 1):
        pred = prior + model(x)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 10 == 0 or step == int(args.max_steps):
            history.append({"step": int(step), "mse": float(loss.detach().cpu().item())})

    with torch.no_grad():
        final_pred = prior + model(x)
        final_mse = torch.mean((final_pred - y) ** 2).item()
        max_abs_err = torch.max(torch.abs(final_pred - y)).item()

    task_path_sources = sorted({str(row.get("task_path_source") or "unknown") for row in used_rows})
    uses_gt_debug_path = any("gt" in source and "debug" in source for source in task_path_sources)
    formal_blocker = (
        "This overfit uses GT task-path debug conditioning. Formal training "
        "must use causal Cosmos-predicted task paths and then pass closed-loop video/final-state eval."
        if uses_gt_debug_path
        else (
            "This overfit uses causal Cosmos-predicted task paths, but it is still "
            "a short debug gate. Formal training needs DP-prior export at "
            "training scale plus closed-loop video/final-state eval."
        )
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": int(x.shape[1]),
            "target_dim": int(y.shape[1]),
            "used_rows": used_rows,
        },
        output_root / "executor_overfit_model.pt",
    )
    write_json(output_root / "loss_history.json", history)
    summary = {
        "schema": "cosmos3_executor_overfit_v1",
        "executor_jsonl": str(Path(args.executor_jsonl).resolve()),
        "dp_prior_jsonl": str(Path(args.dp_prior_jsonl).resolve()),
        "output_root": str(output_root),
        "num_samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "device": str(device),
        "max_steps": int(args.max_steps),
        "baseline_dp_prior_mse": float(baseline_mse),
        "initial_residual_mse": float(initial_mse),
        "final_mse": float(final_mse),
        "max_abs_error": float(max_abs_err),
        "task_path_sources": task_path_sources,
        "loss_history": history,
        "used_rows": used_rows,
        "ready_for_debug_gate": bool(final_mse < baseline_mse and final_mse < initial_mse),
        "ready_for_formal_executor_training": False,
        "formal_training_blocker": formal_blocker,
        "training_rule": "short overfit exception: 1-2 GPUs, about 50-100 steps, no 3-hour minimum",
    }
    write_json(output_root / "executor_overfit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ready_for_debug_gate"] else 66


if __name__ == "__main__":
    raise SystemExit(main())
