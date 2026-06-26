#!/usr/bin/env python3
"""Receding Cosmos3 live-loop scaffold for PegInsertionSide.

This script is compute-node only. In dry-run mode it restores a source prefix,
builds the causal prefix-only video and live WAM history needed for one Cosmos
reobservation call, and stops. With ``--run-cosmos-inference`` it can call the
single-prefix wrapper, execute the returned short robot-action chunk, append
real simulator observations to the history, and repeat.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_cosmos3_maniskill_full_episode_wam_conditions import (  # noqa: E402
    FULL_EPISODE_VECTOR_NAMES,
    _load_episode_arrays,
    _raw_vectors_from_arrays,
)
from build_cosmos3_executor_predicted_task_path_dataset import WAM_TO_EXECUTOR_TASK_PATH  # noqa: E402
from build_cosmos3_executor_training_dataset import CURRENT_STATE_NAMES, TASK_PATH_NAMES  # noqa: E402
from run_cosmos3_receding_closed_loop import (  # noqa: E402
    _action_space_bounds,
    _get_base_env,
    _import_live_control_stack,
    _load_dp_agent,
    _live_eval,
    _make_live_env,
    _parse_seed_from_text,
    _prepare_step_action,
    _render_frame,
    jsonable,
)
from video_contract_utils import inspect_video_file, video_inspections_match_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--initial-video", default=None)
    parser.add_argument(
        "--prefix-frame-source",
        choices=("render_env_states", "initial_video"),
        default="render_env_states",
        help=(
            "How to build the initial observed prefix. render_env_states replays "
            "source env_states through the live renderer; initial_video reads a "
            "provided prefix-only/reference video."
        ),
    )
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dp-manifest", required=True)
    parser.add_argument("--dp-checkpoint", default="")
    parser.add_argument(
        "--dp-state-key",
        choices=("ema_agent", "agent"),
        default="ema_agent",
        help="Frozen DP state key used only after the live continuability gate passes.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--prefix-frame-index",
        type=int,
        default=-1,
        help=(
            "Manual diagnostic start frame. Active controller eval should use "
            "--prefix-start-mode=target_motion_onset instead of hand-picking this."
        ),
    )
    parser.add_argument(
        "--prefix-start-mode",
        choices=("manual", "target_motion_onset"),
        default="manual",
        help=(
            "How to choose the first Cosmos prefix. target_motion_onset scans "
            "observed target poses causally and starts only when target motion "
            "is detected; manual is diagnostic/backward-compatible."
        ),
    )
    parser.add_argument("--min-dynamic-prefix-frame", type=int, default=8)
    parser.add_argument("--target-motion-consecutive-frames", type=int, default=2)
    parser.add_argument("--scenario", default="live_dynamic")
    parser.add_argument(
        "--prefix-role",
        default="auto",
        help=(
            "Observed-prefix role. Use auto for live receding evaluation so "
            "the role is recomputed from real history at each reobservation; "
            "fixed values are diagnostic overrides."
        ),
    )
    parser.add_argument("--sample-name", default="live_receding")
    parser.add_argument("--max-receding-iterations", type=int, default=1)
    parser.add_argument("--action-exec-horizon", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--expected-video-frames", type=int, default=301)
    parser.add_argument("--expected-action-steps", type=int, default=300)
    parser.add_argument("--expected-action-dim", type=int, default=32)
    parser.add_argument("--robot-action-dim", type=int, default=7)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--full-episode-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true, keep executing the unified detector/controller loop "
            "until the 300-action episode horizon unless the simulator "
            "terminates/truncates. Success is recorded but does not shorten "
            "the demo video."
        ),
    )
    parser.add_argument(
        "--annotate-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write an annotated rollout video with per-frame controller mode "
            "and target-motion detector state. Raw video is always written."
        ),
    )
    parser.add_argument(
        "--pretrigger-control-mode",
        choices=("frozen_dp_until_target_motion", "source_restore"),
        default="frozen_dp_until_target_motion",
        help=(
            "How to build the observed prefix before the first Cosmos call. "
            "frozen_dp_until_target_motion runs the frozen static DP in the "
            "live env from source frame 0 while replaying only external target "
            "motion, then triggers Cosmos from observed target motion. "
            "source_restore directly restores the source prefix and is "
            "diagnostic only."
        ),
    )
    parser.add_argument(
        "--external-target-mode",
        choices=("source_env_state", "none"),
        default="source_env_state",
        help=(
            "How the exogenous moving target is advanced during live eval. "
            "source_env_state replays only the source H5 target actor pose after "
            "each live robot action, preserving the dynamic task while leaving "
            "robot/peg state live."
        ),
    )
    parser.add_argument("--run-cosmos-inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cosmos-wrapper", default="scripts/slurm/run_cosmos3_live_prefix_policy_inference_in_allocation.sh")
    parser.add_argument(
        "--controller-action-source",
        choices=("cosmos_robot_action", "residual_executor", "contact_executor", "candidate_executor"),
        default="cosmos_robot_action",
        help=(
            "cosmos_robot_action executes the raw robot-action columns predicted "
            "by Cosmos. residual_executor uses Cosmos only as a task-path world "
            "model, then runs the learned DP-prior residual executor for robot actions. "
            "contact_executor uses the contact/progress-conditioned action head "
            "with causal live contact context. candidate_executor samples/scales "
            "multiple residual chunks and selects one with an action-conditioned "
            "progress/contact/value scorer."
        ),
    )
    parser.add_argument(
        "--executor-checkpoint",
        default="",
        help="Executor checkpoint used with residual_executor, contact_executor, or candidate_executor.",
    )
    parser.add_argument(
        "--candidate-outcome-scorer-checkpoint",
        default="",
        help=(
            "Optional real-outcome progress/contact/value scorer used to rank "
            "candidate_executor action chunks. If omitted, the candidate "
            "executor's built-in scorer selects the chunk."
        ),
    )
    parser.add_argument(
        "--candidate-outcome-scorer-dp-margin",
        type=float,
        default=0.0,
        help=(
            "When the optional outcome scorer is active, keep the DP-prior "
            "candidate unless a non-DP candidate beats it by at least this "
            "composite score margin."
        ),
    )
    parser.add_argument(
        "--candidate-outcome-scorer-min-progress-delta",
        type=float,
        default=-1.0e9,
        help=(
            "Optional live safety filter for the outcome-scorer selector. "
            "When active, non-DP candidates whose predicted contact-progress "
            "delta is below this value are rejected before selecting an action "
            "chunk. Defaults to disabled to preserve old results."
        ),
    )
    parser.add_argument(
        "--candidate-outcome-scorer-min-continuable-prob",
        type=float,
        default=0.0,
        help=(
            "Optional live safety filter for non-DP candidates. Reject a "
            "non-DP candidate unless the outcome scorer predicts at least this "
            "DP-continuable probability. Defaults to zero, which is disabled "
            "for sigmoid probabilities."
        ),
    )
    parser.add_argument(
        "--candidate-outcome-scorer-min-inserted-prob",
        type=float,
        default=0.0,
        help=(
            "Optional live safety filter for non-DP candidates. Reject a "
            "non-DP candidate unless the outcome scorer predicts at least this "
            "inserted probability. Defaults to zero, which is disabled."
        ),
    )
    parser.add_argument(
        "--candidate-outcome-scorer-score-state-abs-axis-weights",
        default="",
        help=(
            "Optional override for the outcome scorer's predicted final "
            "peg-head-at-hole absolute state penalty weights. Empty uses the "
            "checkpoint value."
        ),
    )
    parser.add_argument(
        "--candidate-outcome-scorer-score-state-target",
        default="",
        help=(
            "Optional override for the outcome scorer's predicted final "
            "peg-head-at-hole target. Empty uses the checkpoint value."
        ),
    )
    parser.add_argument(
        "--candidate-executor-short-prefix-steps",
        default="",
        help=(
            "Optional comma-separated step counts for short-prefix candidate "
            "execution, for example 8,12,16. A short candidate is scored as "
            "candidate-prefix plus DP prior suffix over the checkpoint horizon, "
            "but only the prefix is executed before reobserving. Empty preserves "
            "the original full-chunk behavior."
        ),
    )
    parser.add_argument(
        "--source-insertion-suffix-bank",
        default="",
        help=(
            "Optional diagnostic NPZ built by build_cosmos3_source_insertion_suffix_bank.py. "
            "When used with candidate_executor, nearest successful insertion suffixes are "
            "added as retrieval-residual-style candidates from the current live task state."
        ),
    )
    parser.add_argument("--source-suffix-k", type=int, default=0)
    parser.add_argument("--source-suffix-blends", default="1.0")
    parser.add_argument("--source-suffix-execute-steps", type=int, default=32)
    parser.add_argument("--source-suffix-offsets", default="")
    parser.add_argument(
        "--source-suffix-scenario-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic-only exact scenario filter. Default stays off to avoid scenario-label control flow.",
    )
    parser.add_argument("--source-suffix-query-x-weight", type=float, default=1.0)
    parser.add_argument("--source-suffix-query-y-weight", type=float, default=2.0)
    parser.add_argument("--source-suffix-query-z-weight", type=float, default=4.0)
    parser.add_argument(
        "--source-suffix-max-distance",
        type=float,
        default=-1.0,
        help=(
            "Optional weighted peg-head-at-hole distance gate for source-suffix "
            "retrieval. Values <0 disable the gate. This prevents executing a "
            "successful suffix whose recorded start state is not close to the "
            "current live state."
        ),
    )
    parser.add_argument(
        "--source-suffix-ignore-residual-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic switch allowing source-suffix candidates to bypass the learned "
            "model-residual L2 cap. Use only with explicit replay/live notes."
        ),
    )
    parser.add_argument(
        "--executor-residual-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the learned residual before adding it to the DP prior. "
            "Use 1.0 for the raw trained executor; smaller values require a recorded "
            "validation calibration and must not be used to hide a failed model."
        ),
    )
    parser.add_argument("--clip-live-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dp-handoff-horizon",
        type=int,
        default=0,
        help=(
            "Optional frozen-DP continuation horizon after a real live state "
            "passes the conservative continuability gate. Zero disables DP."
        ),
    )
    parser.add_argument(
        "--dp-handoff-chunk-horizon",
        type=int,
        default=0,
        help=(
            "Maximum DP steps to execute in one reobserved handoff chunk. "
            "Zero preserves the old behavior of using action_exec_horizon. "
            "This separates high-frequency executor chunk length from the DP "
            "handoff length used to match DP-rollout continuability labels."
        ),
    )
    parser.add_argument(
        "--cosmos-step-handoff-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After every executed Cosmos action step, evaluate the real-state "
            "C_pi gate. If it passes, stop the current Cosmos chunk early so "
            "the next observe-decide iteration can immediately hand off a "
            "short chunk to frozen DP. This does not relax C_pi and does not "
            "use generated state as authority."
        ),
    )
    parser.add_argument(
        "--save-live-state-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write per-iteration live simulator state snapshots before and "
            "after controller execution. This is for failed-state recovery "
            "data construction only and is not a controller condition."
        ),
    )
    parser.add_argument(
        "--save-candidate-action-bank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For candidate-executor iterations, save every generated candidate "
            "action/residual array to an NPZ sidecar. This is for later "
            "live-state replay labeling only and is not a controller input."
        ),
    )
    parser.add_argument(
        "--live-progress-interval",
        type=int,
        default=0,
        help="When positive, print live-loop progress every N pretrigger steps.",
    )
    parser.add_argument("--continuability-min-rel-x", type=float, default=-0.08)
    parser.add_argument("--continuability-max-rel-x", type=float, default=0.04)
    parser.add_argument("--continuability-max-abs-y", type=float, default=0.025)
    parser.add_argument("--continuability-max-abs-z", type=float, default=0.025)
    parser.add_argument("--continuability-max-hole-speed", type=float, default=0.01)
    parser.add_argument(
        "--continuability-stats-json",
        default="",
        help=(
            "Optional static-DP success-manifold statistics JSON. When set, "
            "the live DP handoff gate derives distance thresholds from the "
            "specified within-N-steps profile instead of ad-hoc CLI defaults."
        ),
    )
    parser.add_argument(
        "--continuability-stats-horizon",
        type=int,
        default=32,
        help="Use within_<horizon>_steps_to_first_success from the stats JSON.",
    )
    parser.add_argument(
        "--continuability-stats-x-lower-quantile",
        type=float,
        default=0.01,
        help="Lower x quantile used as the far-before-hole DP handoff bound.",
    )
    parser.add_argument(
        "--continuability-stats-abs-quantile",
        type=float,
        default=0.95,
        help="Absolute y/z quantile used as lateral/vertical DP handoff bounds.",
    )
    parser.add_argument(
        "--continuability-stats-set-x-upper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, also derive max_rel_x from the static-DP x quantile. "
            "By default the CLI safety cap is preserved because closer-than-demo "
            "states can still be safe handoff candidates."
        ),
    )
    parser.add_argument(
        "--continuability-stats-x-upper-quantile",
        type=float,
        default=1.0,
        help="Upper x quantile used only with --continuability-stats-set-x-upper.",
    )
    parser.add_argument("--target-motion-delta-threshold", type=float, default=0.002)
    parser.add_argument("--target-motion-speed-threshold", type=float, default=0.001)
    return parser.parse_args()


def require_compute_step() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id or not step_id or step_id == "extern":
        raise SystemExit(
            "refusing_login_node_execution=true\n"
            "reason=Run live receding loop only inside a compute-node srun step."
        )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n")


def write_action_json(path: Path, raw: np.ndarray) -> None:
    write_json(path, {"action": raw.astype(float).tolist()})


def build_residual_executor_model(feature_dim: int, target_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Any:
    import torch

    layers: list[torch.nn.Module] = []
    width = int(hidden_dim)
    in_dim = int(feature_dim)
    for _ in range(int(num_layers)):
        layers.append(torch.nn.Linear(in_dim, width))
        layers.append(torch.nn.LayerNorm(width))
        layers.append(torch.nn.GELU())
        if float(dropout) > 0:
            layers.append(torch.nn.Dropout(float(dropout)))
        in_dim = width
    layers.append(torch.nn.Linear(width, int(target_dim)))
    return torch.nn.Sequential(*layers)


def build_contact_executor_model(feature_dim: int, target_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Any:
    import torch

    class _ContactExecutorNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            in_dim = int(feature_dim)
            for _ in range(int(num_layers)):
                layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
                layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                layers.append(torch.nn.GELU())
                if float(dropout) > 0:
                    layers.append(torch.nn.Dropout(float(dropout)))
                in_dim = int(hidden_dim)
            self.trunk = torch.nn.Sequential(*layers)
            self.action_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
            self.progress_head = torch.nn.Linear(int(hidden_dim), 2)
            self.binary_head = torch.nn.Linear(int(hidden_dim), 2)

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            z = self.trunk(x)
            return self.action_head(z), self.progress_head(z), self.binary_head(z)

    return _ContactExecutorNet()


def build_candidate_executor_model(
    feature_dim: int,
    target_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    logstd_min: float,
    logstd_max: float,
    next_state_dim: int = 0,
) -> Any:
    import torch

    class _CandidateExecutorNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            in_dim = int(feature_dim)
            for _ in range(int(num_layers)):
                layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
                layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                layers.append(torch.nn.GELU())
                if float(dropout) > 0:
                    layers.append(torch.nn.Dropout(float(dropout)))
                in_dim = int(hidden_dim)
            self.encoder = torch.nn.Sequential(*layers)
            self.mean_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
            self.logstd_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
            self.scorer = torch.nn.Sequential(
                torch.nn.Linear(int(hidden_dim) + int(target_dim), int(hidden_dim)),
                torch.nn.LayerNorm(int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(hidden_dim), max(1, int(hidden_dim) // 2)),
                torch.nn.GELU(),
            )
            score_dim = max(1, int(hidden_dim) // 2)
            self.progress_head = torch.nn.Linear(score_dim, 2)
            self.binary_head = torch.nn.Linear(score_dim, 2)
            self.value_head = torch.nn.Linear(score_dim, 1)
            self.next_state_head = (
                torch.nn.Linear(score_dim, int(next_state_dim))
                if int(next_state_dim) > 0
                else None
            )
            self.logstd_min = float(logstd_min)
            self.logstd_max = float(logstd_max)

        def encode(self, x: Any) -> Any:
            return self.encoder(x)

        def distribution(self, z: Any) -> tuple[Any, Any]:
            mean = self.mean_head(z)
            raw = self.logstd_head(z)
            logstd = self.logstd_min + (self.logstd_max - self.logstd_min) * torch.sigmoid(raw)
            return mean, logstd

        def score_candidate(self, z: Any, candidate_resid_norm: Any) -> tuple[Any, Any, Any, Any]:
            h = self.scorer(torch.cat([z, candidate_resid_norm], dim=-1))
            next_state = self.next_state_head(h) if self.next_state_head is not None else None
            return self.progress_head(h), self.binary_head(h), self.value_head(h), next_state

        def forward(self, x: Any, candidate_resid_norm: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
            z = self.encode(x)
            mean, logstd = self.distribution(z)
            progress, binary, value, next_state = self.score_candidate(z, candidate_resid_norm)
            return mean, logstd, progress, binary, value, next_state

    return _CandidateExecutorNet()


def build_diffusion_candidate_executor_model(
    feature_dim: int,
    target_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    logstd_min: float,
    logstd_max: float,
    next_state_dim: int = 0,
) -> Any:
    import torch

    class _DiffusionCandidateExecutorNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            in_dim = int(feature_dim)
            for _ in range(int(num_layers)):
                layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
                layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                layers.append(torch.nn.GELU())
                if float(dropout) > 0:
                    layers.append(torch.nn.Dropout(float(dropout)))
                in_dim = int(hidden_dim)
            self.encoder = torch.nn.Sequential(*layers)
            self.mean_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
            self.logstd_head = torch.nn.Linear(int(hidden_dim), int(target_dim))
            self.denoiser = torch.nn.Sequential(
                torch.nn.Linear(int(hidden_dim) + int(target_dim) + 1, int(hidden_dim)),
                torch.nn.LayerNorm(int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(hidden_dim), int(target_dim)),
            )
            self.scorer = torch.nn.Sequential(
                torch.nn.Linear(int(hidden_dim) + int(target_dim), int(hidden_dim)),
                torch.nn.LayerNorm(int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(hidden_dim), max(1, int(hidden_dim) // 2)),
                torch.nn.GELU(),
            )
            score_dim = max(1, int(hidden_dim) // 2)
            self.progress_head = torch.nn.Linear(score_dim, 2)
            self.binary_head = torch.nn.Linear(score_dim, 2)
            self.value_head = torch.nn.Linear(score_dim, 1)
            self.next_state_head = (
                torch.nn.Linear(score_dim, int(next_state_dim))
                if int(next_state_dim) > 0
                else None
            )
            self.logstd_min = float(logstd_min)
            self.logstd_max = float(logstd_max)

        def encode(self, x: Any) -> Any:
            return self.encoder(x)

        def distribution(self, z: Any) -> tuple[Any, Any]:
            mean = self.mean_head(z)
            raw = self.logstd_head(z)
            logstd = self.logstd_min + (self.logstd_max - self.logstd_min) * torch.sigmoid(raw)
            return mean, logstd

        def denoise(self, z: Any, noisy_resid_norm: Any, t_norm: Any) -> Any:
            if t_norm.ndim == 1:
                t_norm = t_norm[:, None]
            return self.denoiser(torch.cat([z, noisy_resid_norm, t_norm], dim=-1))

        def score_candidate(self, z: Any, candidate_resid_norm: Any) -> tuple[Any, Any, Any, Any]:
            h = self.scorer(torch.cat([z, candidate_resid_norm], dim=-1))
            next_state = self.next_state_head(h) if self.next_state_head is not None else None
            return self.progress_head(h), self.binary_head(h), self.value_head(h), next_state

        def forward(self, x: Any, candidate_resid_norm: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
            z = self.encode(x)
            mean, logstd = self.distribution(z)
            progress, binary, value, next_state = self.score_candidate(z, candidate_resid_norm)
            return mean, logstd, progress, binary, value, next_state

    return _DiffusionCandidateExecutorNet()


def load_residual_executor_checkpoint(path: Path, robot_action_dim: int) -> dict[str, Any]:
    import torch

    checkpoint_path = path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing residual executor checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = payload.get("args") or {}
    feature_dim = int(payload["feature_dim"])
    target_dim = int(payload["target_dim"])
    if target_dim % int(robot_action_dim) != 0:
        raise ValueError(f"executor target_dim {target_dim} is not divisible by robot_action_dim {robot_action_dim}")
    model = build_residual_executor_model(
        feature_dim=feature_dim,
        target_dim=target_dim,
        hidden_dim=int(args.get("hidden_dim", 1024)),
        num_layers=int(args.get("num_layers", 4)),
        dropout=float(args.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "model": model,
        "feature_dim": feature_dim,
        "target_dim": target_dim,
        "horizon": target_dim // int(robot_action_dim),
        "x_mean": np.asarray(payload["x_mean"], dtype=np.float32).reshape(-1),
        "x_std": np.asarray(payload["x_std"], dtype=np.float32).reshape(-1),
        "residual_mean": np.asarray(payload["residual_mean"], dtype=np.float32).reshape(-1),
        "residual_std": np.asarray(payload["residual_std"], dtype=np.float32).reshape(-1),
        "checkpoint_metrics": payload.get("metrics") or {},
        "checkpoint_args": args,
    }


def load_contact_executor_checkpoint(path: Path, robot_action_dim: int) -> dict[str, Any]:
    import torch

    checkpoint_path = path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing contact executor checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = payload.get("args") or {}
    feature_dim = int(payload["feature_dim"])
    target_dim = int(payload["target_dim"])
    if target_dim % int(robot_action_dim) != 0:
        raise ValueError(f"executor target_dim {target_dim} is not divisible by robot_action_dim {robot_action_dim}")
    model = build_contact_executor_model(
        feature_dim=feature_dim,
        target_dim=target_dim,
        hidden_dim=int(args.get("hidden_dim", 2048)),
        num_layers=int(args.get("num_layers", 5)),
        dropout=float(args.get("dropout", 0.10)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "model": model,
        "feature_dim": feature_dim,
        "target_dim": target_dim,
        "horizon": target_dim // int(robot_action_dim),
        "x_mean": np.asarray(payload["x_mean"], dtype=np.float32).reshape(-1),
        "x_std": np.asarray(payload["x_std"], dtype=np.float32).reshape(-1),
        "residual_mean": np.asarray(payload["residual_mean"], dtype=np.float32).reshape(-1),
        "residual_std": np.asarray(payload["residual_std"], dtype=np.float32).reshape(-1),
        "phase_residual_l2_caps": {
            str(key): float(value)
            for key, value in dict(payload.get("phase_residual_l2_caps") or {}).items()
        },
        "checkpoint_metrics": payload.get("latest_metrics") or payload.get("metrics") or {},
        "checkpoint_args": args,
        "executor_type": "contact_executor",
    }


def _parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    return values or [1.0]


def _parse_scale_list(text: str) -> list[float]:
    values = []
    for value in _parse_float_list(text):
        if value < 0:
            raise ValueError(f"candidate scales must be non-negative, got {value}")
        if value == 0.0:
            continue
        values.append(float(value))
    return values or [0.05, 0.1, 0.2]


def _parse_blend_list(text: str) -> list[float]:
    values: list[float] = []
    for value in _parse_float_list(text):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"source suffix blend must be in [0, 1], got {value}")
        if value not in values:
            values.append(float(value))
    return values or [1.0]


def _parse_positive_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"step counts must be positive, got {value}")
        if value not in out:
            out.append(value)
    return out


def _parse_nonnegative_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(f"step counts must be non-negative, got {value}")
        if value not in out:
            out.append(value)
    return out


def _short_prefix_steps_from_name(name: str) -> int | None:
    prefix, sep, _rest = str(name).partition("_")
    if sep and prefix.startswith("short") and prefix[5:].isdigit():
        return int(prefix[5:])
    return None


def _candidate_descriptor_base_name(name: str) -> str:
    if _short_prefix_steps_from_name(name) is None:
        return str(name)
    return str(name).split("_", 1)[1]


def _fixed_width_vector(text: str, width: int, default: float = 0.0) -> np.ndarray:
    values = _parse_float_list(str(text)) if str(text).strip() else []
    if not values:
        values = [float(default)] * int(width)
    if len(values) < int(width):
        values.extend([float(values[-1])] * (int(width) - len(values)))
    return np.asarray(values[: int(width)], dtype=np.float32)


def load_source_insertion_suffix_bank(path_text: str) -> dict[str, np.ndarray] | None:
    if not str(path_text).strip():
        return None
    path = Path(path_text).resolve()
    payload = np.load(str(path), allow_pickle=False)
    bank = {key: payload[key] for key in payload.files}
    if "actions" not in bank or "start_peg_head_at_hole" not in bank:
        raise ValueError(f"source insertion suffix bank missing required arrays: {path}")
    bank["_bank_path"] = np.asarray([str(path)], dtype="<U512")
    return bank


def source_insertion_suffix_residual_candidates(
    *,
    source_bank: dict[str, np.ndarray] | None,
    live: dict[str, Any],
    scenario: str,
    dp_prior: np.ndarray,
    horizon: int,
    robot_action_dim: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if source_bank is None or int(args.source_suffix_k) <= 0:
        return []
    actions_all = np.asarray(source_bank["actions"], dtype=np.float32)
    start_rel = np.asarray(source_bank["start_peg_head_at_hole"], dtype=np.float32)
    scenarios = [str(item) for item in np.asarray(source_bank.get("scenario", np.asarray([], dtype="<U1"))).tolist()]
    offsets = np.asarray(source_bank.get("offset_before_insert", np.full((actions_all.shape[0],), -1)), dtype=np.int32)
    valid_steps = np.asarray(source_bank.get("valid_steps", np.full((actions_all.shape[0],), actions_all.shape[1])), dtype=np.int32)
    allowed_offsets = set(_parse_nonnegative_int_list(str(args.source_suffix_offsets))) if str(args.source_suffix_offsets).strip() else set()
    query = np.asarray(live.get("peg_head_at_hole", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)[:3]
    if query.size < 3:
        query = np.pad(query, (0, 3 - query.size), mode="constant")
    weights = np.asarray(
        [
            float(args.source_suffix_query_x_weight),
            float(args.source_suffix_query_y_weight),
            float(args.source_suffix_query_z_weight),
        ],
        dtype=np.float32,
    )
    scenario_available = bool(scenarios) and any(item == str(scenario) for item in scenarios)
    scored: list[tuple[float, int]] = []
    for idx in range(int(actions_all.shape[0])):
        if allowed_offsets and int(offsets[idx]) not in allowed_offsets:
            continue
        if bool(args.source_suffix_scenario_match) and scenario_available and scenarios[idx] != str(scenario):
            continue
        rel = np.asarray(start_rel[idx, :3], dtype=np.float32)
        dist = float(np.linalg.norm((rel - query) * weights))
        if float(args.source_suffix_max_distance) >= 0.0 and dist > float(args.source_suffix_max_distance):
            continue
        scored.append((dist, idx))
    scored.sort(key=lambda item: item[0])
    if not scored:
        return []

    prior = np.asarray(dp_prior, dtype=np.float32)[:, : int(robot_action_dim)]
    out: list[dict[str, Any]] = []
    for rank, (dist, source_idx) in enumerate(scored[: int(args.source_suffix_k)]):
        source_actions = np.asarray(actions_all[source_idx], dtype=np.float32)[:, : int(robot_action_dim)]
        local_horizon = min(int(horizon), int(prior.shape[0]), int(source_actions.shape[0]))
        if local_horizon <= 0:
            continue
        for blend in _parse_blend_list(str(args.source_suffix_blends)):
            full_actions = prior[:local_horizon] * (1.0 - float(blend)) + source_actions[:local_horizon] * float(blend)
            if local_horizon < int(horizon):
                pad = np.repeat(full_actions[-1:, :], int(horizon) - local_horizon, axis=0)
                full_actions = np.concatenate([full_actions, pad], axis=0)
            execute_steps = int(args.source_suffix_execute_steps)
            if execute_steps <= 0:
                execute_steps = min(int(horizon), int(valid_steps[source_idx]))
            execute_steps = max(1, min(int(execute_steps), int(horizon), int(valid_steps[source_idx])))
            blend_tag = f"{float(blend):g}".replace(".", "p")
            name = f"retrieval_resid_srcsuffix_r{rank}_s{blend_tag}_o{int(offsets[source_idx])}"
            out.append(
                {
                    "name": name,
                    "actions": full_actions.astype(np.float32),
                    "execute_steps": int(execute_steps),
                    "source_idx": int(source_idx),
                    "rank": int(rank),
                    "distance": float(dist),
                    "blend": float(blend),
                    "offset": int(offsets[source_idx]),
                    "scenario": scenarios[source_idx] if scenarios else None,
                    "start_rel": start_rel[source_idx, :3].astype(float).tolist(),
                    "query_rel": query.astype(float).tolist(),
                }
            )
    return out


def _diffusion_schedule(
    *,
    steps: int,
    beta_start: float,
    beta_end: float,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch

    step_count = max(2, int(steps))
    betas = torch.linspace(float(beta_start), float(beta_end), step_count, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def _diffusion_candidate_samples(
    *,
    model: Any,
    z: Any,
    sample_count: int,
    steps: int,
    beta_start: float,
    beta_end: float,
    generator: Any,
    device: Any,
) -> list[tuple[str, Any]]:
    import torch

    if int(sample_count) <= 0:
        return []
    step_count = max(2, int(steps))
    betas, alphas, alpha_bars = _diffusion_schedule(
        steps=step_count,
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        device=device,
    )
    batch = int(z.shape[0])
    target_dim = int(model.mean_head.out_features)
    samples: list[tuple[str, Any]] = []
    for sample_idx in range(int(sample_count)):
        x_t = torch.randn((batch, target_dim), generator=generator, device=device)
        for step_idx in reversed(range(step_count)):
            t_norm = torch.full((batch, 1), float(step_idx) / max(1, step_count - 1), device=device)
            pred_noise = model.denoise(z, x_t, t_norm)
            beta_t = betas[step_idx]
            alpha_t = alphas[step_idx]
            alpha_bar_t = alpha_bars[step_idx]
            coef = beta_t / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-6))
            mean = (x_t - coef * pred_noise) / torch.sqrt(torch.clamp(alpha_t, min=1e-6))
            if step_idx > 0:
                noise = torch.randn(x_t.shape, generator=generator, device=device)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean
        samples.append((f"diffusion_{sample_idx}", x_t))
    return samples


def load_candidate_executor_checkpoint(path: Path, robot_action_dim: int) -> dict[str, Any]:
    import torch

    checkpoint_path = path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing candidate executor checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = payload.get("args") or {}
    feature_dim = int(payload["feature_dim"])
    target_dim = int(payload["target_dim"])
    if target_dim % int(robot_action_dim) != 0:
        raise ValueError(f"candidate executor target_dim {target_dim} is not divisible by robot_action_dim {robot_action_dim}")
    generator_type = str(args.get("generator_type", "gaussian"))
    state_dict = payload["model_state_dict"]
    next_state_dim = int(payload.get("next_state_dim", 0) or 0)
    if next_state_dim <= 0 and any(str(key).startswith("next_state_head.") for key in state_dict):
        next_state_dim = 3
    model_builder = build_diffusion_candidate_executor_model if generator_type == "diffusion" else build_candidate_executor_model
    model = model_builder(
        feature_dim=feature_dim,
        target_dim=target_dim,
        hidden_dim=int(args.get("hidden_dim", 1024)),
        num_layers=int(args.get("num_layers", 4)),
        dropout=float(args.get("dropout", 0.05)),
        logstd_min=float(args.get("logstd_min", -4.5)),
        logstd_max=float(args.get("logstd_max", 1.0)),
        next_state_dim=next_state_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "model": model,
        "feature_dim": feature_dim,
        "target_dim": target_dim,
        "next_state_dim": int(next_state_dim),
        "horizon": target_dim // int(robot_action_dim),
        "x_mean": np.asarray(payload["x_mean"], dtype=np.float32).reshape(-1),
        "x_std": np.asarray(payload["x_std"], dtype=np.float32).reshape(-1),
        "residual_mean": np.asarray(payload["residual_mean"], dtype=np.float32).reshape(-1),
        "residual_std": np.asarray(payload["residual_std"], dtype=np.float32).reshape(-1),
        "phase_residual_l2_caps": {
            str(key): float(value)
            for key, value in dict(payload.get("phase_residual_l2_caps") or {}).items()
        },
        "checkpoint_metrics": payload.get("latest_metrics") or payload.get("metrics") or {},
        "checkpoint_args": args,
        "generator_type": generator_type,
        "diffusion_steps": int(args.get("diffusion_steps", 16)),
        "diffusion_beta_start": float(args.get("diffusion_beta_start", 1e-4)),
        "diffusion_beta_end": float(args.get("diffusion_beta_end", 2e-2)),
        "candidate_temps": _parse_float_list(str(args.get("candidate_temps", "0.5,1.0,1.5"))),
        "candidate_scales": _parse_scale_list(str(args.get("candidate_scales", "0.05,0.1,0.2,0.5,1.0"))),
        "candidate_samples": int(args.get("candidate_samples", 24)),
        "candidate_rank_loss_weight": float(args.get("candidate_rank_loss_weight", 0.0)),
        "candidate_rank_random_count": int(args.get("candidate_rank_random_count", 0)),
        "candidate_rank_diffusion_count": int(args.get("candidate_rank_diffusion_count", 0)),
        "candidate_rank_temperature": float(args.get("candidate_rank_temperature", 1.0)),
        "score_inserted_weight": float(args.get("score_inserted_weight", 0.6)),
        "score_dp_continuable_weight": float(args.get("score_dp_continuable_weight", 0.3)),
        "score_value_weight": float(args.get("score_value_weight", 0.4)),
        "score_next_state_weight": float(args.get("score_next_state_weight", 0.0 if next_state_dim <= 0 else 0.8)),
        "score_next_state_axis_weights": _fixed_width_vector(
            str(args.get("score_next_state_axis_weights", "1.0,2.0,4.0")), 3, 1.0
        ),
        "score_next_state_target": _fixed_width_vector(
            str(args.get("score_next_state_target", "0.0,0.0,0.0")), 3, 0.0
        ),
        "score_logprob_weight": float(args.get("score_logprob_weight", 0.05)),
        "score_residual_l2_penalty": float(args.get("score_residual_l2_penalty", 0.02)),
        "score_mean_source_penalty": float(args.get("score_mean_source_penalty", 0.0)),
        "score_scale_source_penalty": float(args.get("score_scale_source_penalty", 0.0)),
        "score_large_scale_source_penalty": float(args.get("score_large_scale_source_penalty", 0.0)),
        "score_stochastic_source_penalty": float(args.get("score_stochastic_source_penalty", 0.25)),
        "selector_residual_l2_cap_max": float(args.get("selector_residual_l2_cap_max", 0.02)),
        "dp_fallback_phases": {
            item.strip()
            for item in str(args.get("dp_fallback_phases", "dp_continuable,preinsert_aligned")).split(",")
            if item.strip()
        },
        "dp_fallback_score_margin": float(args.get("dp_fallback_score_margin", 0.25)),
        "seed": int(args.get("seed", 20260615)),
        "executor_type": "candidate_executor",
    }


def build_candidate_outcome_scorer_model(
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    binary_dim: int = 4,
) -> Any:
    import torch

    class _OutcomeScorerNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            in_dim = int(feature_dim)
            for _ in range(int(num_layers)):
                layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
                layers.append(torch.nn.LayerNorm(int(hidden_dim)))
                layers.append(torch.nn.GELU())
                if float(dropout) > 0:
                    layers.append(torch.nn.Dropout(float(dropout)))
                in_dim = int(hidden_dim)
            self.encoder = torch.nn.Sequential(*layers)
            self.error_head = torch.nn.Linear(int(hidden_dim), 1)
            self.state_head = torch.nn.Linear(int(hidden_dim), 3)
            self.progress_head = torch.nn.Linear(int(hidden_dim), 2)
            self.binary_head = torch.nn.Linear(int(hidden_dim), int(binary_dim))

        def forward(self, x: Any) -> tuple[Any, Any, Any, Any]:
            h = self.encoder(x)
            return self.error_head(h), self.state_head(h), self.progress_head(h), self.binary_head(h)

    return _OutcomeScorerNet()


CANDIDATE_OUTCOME_DESCRIPTOR_NAMES = (
    "is_dp_prior",
    "is_teacher",
    "is_teacher_scale",
    "is_model_mean",
    "is_model_scale",
    "model_scale_value",
    "is_model_sample",
    "model_sample_temperature",
    "model_sample_index_norm64",
    "is_model_diffusion",
    "model_diffusion_index_norm15",
    "is_retrieval_success_residual",
    "retrieval_rank_norm8",
    "retrieval_residual_scale",
    "source_legacy_teacher_scale",
    "source_checkpoint_model_or_model_generated",
    "source_retrieval_success_residual",
)


def _candidate_outcome_descriptor(name: str, source: str) -> np.ndarray:
    virtual_name = _candidate_descriptor_base_name(str(name))
    if source == "checkpoint_model":
        if virtual_name == "dp_prior":
            virtual_name = "model_dp_prior"
        elif virtual_name == "mean":
            virtual_name = "model_mean"
        elif virtual_name.startswith("scale_"):
            virtual_name = "model_scale_" + virtual_name.split("_", 1)[1]
        elif virtual_name.startswith("sample_"):
            virtual_name = "model_" + virtual_name
        elif virtual_name.startswith("diffusion_"):
            virtual_name = "model_diffusion_" + virtual_name.rsplit("_", 1)[-1]
    is_dp = virtual_name in {"dp_prior", "model_dp_prior"}
    is_model_scale = virtual_name.startswith("model_scale_")
    is_model_sample = virtual_name.startswith("model_sample_")
    is_model_diffusion = virtual_name.startswith("model_diffusion_")
    is_retrieval = virtual_name.startswith("retrieval_resid_")
    model_scale_value = 0.0
    model_sample_temp = 0.0
    model_sample_index = -1.0
    diffusion_index = -1.0
    retrieval_rank = -1.0
    retrieval_scale = 0.0
    teacher_scale_value = 0.0
    if is_model_scale:
        try:
            model_scale_value = float(virtual_name.rsplit("_", 1)[-1])
        except ValueError:
            model_scale_value = 0.0
    if is_model_sample:
        parts = virtual_name.split("_")
        if len(parts) >= 4 and parts[2].startswith("t"):
            try:
                model_sample_temp = float(parts[2][1:])
            except ValueError:
                model_sample_temp = 0.0
            try:
                model_sample_index = float(parts[3])
            except ValueError:
                model_sample_index = -1.0
    if is_model_diffusion:
        try:
            diffusion_index = float(virtual_name.rsplit("_", 1)[-1])
        except ValueError:
            diffusion_index = -1.0
    if is_retrieval:
        for part in virtual_name.split("_"):
            if part.startswith("r") and part[1:].isdigit():
                retrieval_rank = float(part[1:])
            elif part.startswith("s"):
                try:
                    retrieval_scale = float(part[1:].replace("p", "."))
                except ValueError:
                    retrieval_scale = 1.0
    is_teacher_scale = virtual_name.startswith("scale_") and source == "legacy_teacher_scale"
    if is_teacher_scale:
        try:
            teacher_scale_value = float(virtual_name.rsplit("_", 1)[-1])
        except ValueError:
            teacher_scale_value = 0.0
    values = [
        float(is_dp),
        float(virtual_name == "teacher"),
        float(is_teacher_scale),
        float(virtual_name == "model_mean"),
        float(is_model_scale),
        float(model_scale_value),
        float(is_model_sample),
        float(model_sample_temp),
        float(model_sample_index / 64.0 if model_sample_index >= 0.0 else 0.0),
        float(is_model_diffusion),
        float(diffusion_index / 15.0 if diffusion_index >= 0.0 else 0.0),
        float(is_retrieval),
        float(retrieval_rank / 8.0 if retrieval_rank >= 0.0 else 0.0),
        float(retrieval_scale),
        float(source == "legacy_teacher_scale"),
        float(source in {"checkpoint_model", "model_generated"}),
        float(source == "retrieval_success_residual"),
    ]
    if len(values) != len(CANDIDATE_OUTCOME_DESCRIPTOR_NAMES):
        raise AssertionError("candidate outcome descriptor schema length mismatch")
    return np.asarray(values, dtype=np.float32)


def _check_candidate_outcome_descriptor_schema(payload: dict[str, Any]) -> list[str]:
    names = list(payload.get("candidate_descriptor_names") or [])
    if names and names != list(CANDIDATE_OUTCOME_DESCRIPTOR_NAMES):
        raise RuntimeError(
            "candidate outcome scorer descriptor schema mismatch: "
            f"checkpoint={names} live={list(CANDIDATE_OUTCOME_DESCRIPTOR_NAMES)}"
        )
    return names


def load_candidate_outcome_scorer_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    checkpoint_path = path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing candidate outcome scorer checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    descriptor_names = _check_candidate_outcome_descriptor_schema(payload)
    args = payload.get("args") or {}
    feature_dim = int(payload["feature_dim"])
    state_dict = payload["model_state_dict"]
    binary_dim = int(payload.get("binary_dim") or state_dict.get("binary_head.weight").shape[0])
    model = build_candidate_outcome_scorer_model(
        feature_dim=feature_dim,
        hidden_dim=int(args.get("hidden_dim", 512)),
        num_layers=int(args.get("num_layers", 3)),
        dropout=float(args.get("dropout", 0.05)),
        binary_dim=binary_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "model": model,
        "feature_dim": feature_dim,
        "binary_dim": binary_dim,
        "x_mean": np.asarray(payload["x_mean"], dtype=np.float32).reshape(-1),
        "x_std": np.asarray(payload["x_std"], dtype=np.float32).reshape(-1),
        "error_mean": np.asarray(payload["error_mean"], dtype=np.float32).reshape(-1),
        "error_std": np.asarray(payload["error_std"], dtype=np.float32).reshape(-1),
        "candidate_descriptor_names": descriptor_names,
        "score_success_weight": float(args.get("score_success_weight", 0.5)),
        "score_handoff_success_weight": float(args.get("score_handoff_success_weight", 0.0)),
        "score_inserted_weight": float(args.get("score_inserted_weight", 0.25)),
        "score_grasped_weight": float(args.get("score_grasped_weight", 0.1)),
        "score_progress_weight": float(args.get("score_progress_weight", 0.25)),
        "score_progress_delta_weight": float(args.get("score_progress_delta_weight", 0.15)),
        "score_continuable_weight": float(args.get("score_continuable_weight", 0.25)),
        "score_state_abs_axis_weights": _fixed_width_vector(
            str(args.get("score_state_abs_axis_weights", "0,0,0")), 3, 0.0
        ),
        "score_state_target": _fixed_width_vector(str(args.get("score_state_target", "0,0,0")), 3, 0.0),
        "checkpoint_metrics": payload.get("latest_metrics") or {},
        "checkpoint_args": args,
        "selector_type": "candidate_outcome_scorer",
    }


def _pad_rows(arr: np.ndarray, rows: int) -> tuple[np.ndarray, bool]:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"expected 2D array, got {out.shape}")
    if out.shape[0] >= rows:
        return out[:rows].astype(np.float32), False
    if out.shape[0] == 0:
        raise ValueError("cannot pad empty array")
    pad = np.repeat(out[-1:, :], rows - out.shape[0], axis=0)
    return np.concatenate([out, pad], axis=0).astype(np.float32), True


def _cosmos_action_array_from_sample_output(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    outputs = payload.get("outputs") or []
    content = outputs[0].get("content") if outputs else None
    action = (content or {}).get("action")
    arr = np.asarray(action, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (300, len(FULL_EPISODE_VECTOR_NAMES)):
        raise ValueError(f"{path} action shape {arr.shape} != (300, {len(FULL_EPISODE_VECTOR_NAMES)})")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} contains non-finite action values")
    return arr


def _denormalize_cosmos_action(pred: np.ndarray, condition_root: Path) -> tuple[np.ndarray, list[str]]:
    stats_path = condition_root / "normalization_stats.json"
    stats = json.loads(stats_path.read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, -1)
    vector_names = [str(name) for name in stats.get("vector_names") or []]
    if mean.shape != (1, pred.shape[1]) or std.shape != (1, pred.shape[1]):
        raise ValueError(f"normalization stats mismatch: mean={mean.shape} std={std.shape} pred={pred.shape}")
    return (pred * std + mean).astype(np.float32), vector_names


def task_path_from_cosmos_prediction(
    *,
    sample_output_json: Path,
    condition_root: Path,
    prefix_frame: int,
    horizon: int,
) -> dict[str, Any]:
    pred_norm = _cosmos_action_array_from_sample_output(sample_output_json)
    pred_raw, vector_names = _denormalize_cosmos_action(pred_norm, condition_root)
    by_name = {name: idx for idx, name in enumerate(vector_names)}
    cols: list[int] = []
    missing: list[str] = []
    for exec_name in TASK_PATH_NAMES:
        wam_name = WAM_TO_EXECUTOR_TASK_PATH[exec_name]
        if wam_name not in by_name:
            missing.append(wam_name)
        else:
            cols.append(by_name[wam_name])
    if missing:
        raise ValueError(f"Cosmos normalization stats missing executor task columns: {missing}")
    start = max(0, min(int(prefix_frame), pred_raw.shape[0]))
    raw_path = pred_raw[start : start + int(horizon), cols].astype(np.float32)
    task_path, padded = _pad_rows(raw_path, int(horizon))
    return {
        "task_path": task_path,
        "task_path_names": list(TASK_PATH_NAMES),
        "sample_output_json": str(sample_output_json),
        "chunk_start": int(start),
        "chunk_end_exclusive": int(min(pred_raw.shape[0], start + int(horizon))),
        "requested_horizon": int(horizon),
        "padded_to_horizon": bool(padded),
        "source": "cosmos_predicted_action_sidecar",
    }


def current_executor_state_from_live(base_env: Any, stack: dict[str, Any], live: dict[str, Any]) -> np.ndarray:
    common = stack["common"]
    qpos = common.to_numpy(base_env.agent.robot.get_qpos())[0].astype(np.float32)
    qvel = common.to_numpy(base_env.agent.robot.get_qvel())[0].astype(np.float32)
    qpos = np.pad(qpos[:9], (0, max(0, 9 - qpos[:9].shape[0])), constant_values=0.0)
    qvel = np.pad(qvel[:9], (0, max(0, 9 - qvel[:9].shape[0])), constant_values=0.0)
    row = np.concatenate(
        [
            np.asarray(live["tcp_pose"], dtype=np.float32)[:3],
            np.asarray(live["peg_pose"], dtype=np.float32)[:3],
            np.asarray(live["hole_pose"], dtype=np.float32)[:3],
            qpos.astype(np.float32),
            qvel.astype(np.float32),
            np.asarray(live["peg_head_at_hole"], dtype=np.float32)[:3],
            np.asarray(live["hole_velocity"], dtype=np.float32)[:3],
            np.asarray([float(bool(live["grasped"])), float(bool(live["inserted"]))], dtype=np.float32),
        ]
    ).astype(np.float32)
    if row.shape != (len(CURRENT_STATE_NAMES),):
        raise RuntimeError(f"executor current_state shape {row.shape} != {len(CURRENT_STATE_NAMES)}")
    return row


def dp_prior_chunk_from_agent(
    *,
    dp_agent: Any,
    dp_args: Any,
    dp_obs_history: list[np.ndarray],
    stack: dict[str, Any],
    horizon: int,
) -> np.ndarray:
    torch = stack["torch"]
    dp_device = next(dp_agent.parameters()).device
    obs_seq = np.stack(dp_obs_history[-2:], axis=0)[None].astype(np.float32)
    obs_tensor = torch.as_tensor(obs_seq, device=dp_device, dtype=torch.float32)
    with torch.no_grad():
        action_seq = dp_agent.get_action(obs_tensor)
    if getattr(dp_args, "sim_backend", "physx_cpu") == "physx_cpu" or hasattr(action_seq, "detach"):
        action_seq_np = action_seq.detach().cpu().numpy()
    else:
        action_seq_np = np.asarray(action_seq)
    prior = np.asarray(action_seq_np[0], dtype=np.float32)
    if prior.ndim != 2 or prior.shape[1] < 7:
        raise RuntimeError(f"DP prior action sequence has invalid shape {prior.shape}")
    prior = prior[:, :7]
    prior, _ = _pad_rows(prior, int(horizon))
    return prior.astype(np.float32)


def residual_executor_action_chunk(
    *,
    bundle: dict[str, Any],
    current_state: np.ndarray,
    task_path: np.ndarray,
    dp_prior: np.ndarray,
    residual_scale: float,
) -> dict[str, Any]:
    import torch

    horizon = int(bundle["horizon"])
    task_path, task_padded = _pad_rows(task_path, horizon)
    dp_prior, prior_padded = _pad_rows(dp_prior, horizon)
    feature = np.concatenate(
        [
            np.asarray(current_state, dtype=np.float32).reshape(-1),
            task_path.reshape(-1),
            dp_prior.reshape(-1),
        ]
    ).astype(np.float32)
    if feature.shape[0] != int(bundle["feature_dim"]):
        raise RuntimeError(f"executor feature width {feature.shape[0]} != checkpoint feature_dim {bundle['feature_dim']}")
    x = ((feature - bundle["x_mean"]) / bundle["x_std"]).astype(np.float32)
    device = bundle["device"]
    with torch.no_grad():
        pred_norm = bundle["model"](torch.as_tensor(x[None], device=device, dtype=torch.float32)).detach().cpu().numpy()[0]
    pred_resid = pred_norm * bundle["residual_std"] + bundle["residual_mean"]
    scale = float(residual_scale)
    pred_resid_scaled = pred_resid.astype(np.float32) * scale
    pred_action = dp_prior.reshape(-1) + pred_resid_scaled
    pred_action = pred_action.reshape(horizon, 7).astype(np.float32)
    return {
        "ok": bool(np.isfinite(pred_action).all()),
        "controller_action_source": "residual_executor",
        "executor_checkpoint": bundle["checkpoint_path"],
        "horizon": horizon,
        "feature_dim": int(bundle["feature_dim"]),
        "target_dim": int(bundle["target_dim"]),
        "current_state_names": list(CURRENT_STATE_NAMES),
        "task_path_names": list(TASK_PATH_NAMES),
        "residual_scale": scale,
        "task_path_padded_to_horizon": bool(task_padded),
        "dp_prior_padded_to_horizon": bool(prior_padded),
        "dp_prior_action_stats": array_stats(dp_prior),
        "residual_action_stats": array_stats(pred_resid.reshape(horizon, 7)),
        "scaled_residual_action_stats": array_stats(pred_resid_scaled.reshape(horizon, 7)),
        "executor_action_stats": array_stats(pred_action),
        "denormalized_robot_action_chunk": pred_action.astype(float).tolist(),
        "boundary": (
            "Robot actions come from the learned residual executor conditioned "
            "on causal Cosmos-predicted task path plus frozen-DP action prior. "
            "Cosmos raw robot-action columns are not executed in this branch."
        ),
    }


def _phase_one_hot(phase_id: int, width: int = 6) -> np.ndarray:
    out = np.zeros((width,), dtype=np.float32)
    if 0 <= int(phase_id) < width:
        out[int(phase_id)] = 1.0
    return out


def _phase_name(phase_id: int) -> str:
    names = {
        0: "no_grasp",
        1: "far",
        2: "lateral_align",
        3: "preinsert_aligned",
        4: "dp_continuable",
        5: "inserted",
    }
    return names.get(int(phase_id), f"phase_{int(phase_id)}")


def contact_context_from_live(
    *,
    live: dict[str, Any],
    history: np.ndarray,
    prefix_frame: int,
    horizon: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rel = np.asarray(live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    x, y, z = [float(v) for v in rel]
    lateral = float(np.sqrt(y * y + z * z))
    grasped = bool(live.get("grasped", False))
    inserted = bool(live.get("inserted", False))
    gate = continuability_gate(live, history, prefix_frame, args)
    dp_continuable = bool(gate.get("ok", False))
    progress_far_x = -0.25
    insert_x_min = -0.015
    lateral_align_radius = 0.05
    preinsert_yz_multiplier = 2.0
    preinsert = (
        grasped
        and x >= progress_far_x
        and x <= float(args.continuability_max_rel_x)
        and abs(y) <= preinsert_yz_multiplier * float(args.continuability_max_abs_y)
        and abs(z) <= preinsert_yz_multiplier * float(args.continuability_max_abs_z)
    )
    lateral_align = grasped and lateral <= lateral_align_radius and not preinsert
    phase_id = 1
    if lateral_align:
        phase_id = 2
    if preinsert:
        phase_id = 3
    if dp_continuable:
        phase_id = 4
    if inserted:
        phase_id = 5
    if not grasped:
        phase_id = 0
    denom = max(insert_x_min - progress_far_x, 1e-6)
    insertion_progress = float(np.clip((x - progress_far_x) / denom, 0.0, 1.0))
    if inserted:
        insertion_progress = 1.0
    lateral_progress = float(np.clip(1.0 - lateral / max(lateral_align_radius, 1e-6), 0.0, 1.0))
    hold_progress = float(grasped)
    contact_progress = float(0.45 * insertion_progress + 0.45 * lateral_progress + 0.10 * hold_progress)
    if inserted:
        contact_progress = 1.0
    end_frame = min(300, int(prefix_frame) + int(horizon))
    context = np.concatenate(
        [
            _phase_one_hot(phase_id),
            np.asarray(
                [
                    contact_progress,
                    float(grasped),
                    float(inserted),
                    float(dp_continuable),
                    float(prefix_frame) / 300.0,
                    float(end_frame) / 300.0,
                ],
                dtype=np.float32,
            ),
            rel.astype(np.float32),
        ]
    ).astype(np.float32)
    return {
        "context": context,
        "phase_id": int(phase_id),
        "phase_name": _phase_name(int(phase_id)),
        "contact_progress": contact_progress,
        "insertion_progress": insertion_progress,
        "lateral_progress": lateral_progress,
        "grasped": grasped,
        "inserted": inserted,
        "dp_continuable": dp_continuable,
        "peg_head_at_hole": rel.astype(float).tolist(),
        "lateral_error": lateral,
        "continuability_gate": gate,
        "boundary": (
            "Live contact context is computed from the current real simulator "
            "state and causal handoff thresholds only. It does not use future "
            "ground-truth contact labels."
        ),
    }


def contact_executor_action_chunk(
    *,
    bundle: dict[str, Any],
    current_state: np.ndarray,
    task_path: np.ndarray,
    dp_prior: np.ndarray,
    live: dict[str, Any],
    history: np.ndarray,
    prefix_frame: int,
    args: argparse.Namespace,
    residual_scale: float,
) -> dict[str, Any]:
    import torch

    horizon = int(bundle["horizon"])
    task_path, task_padded = _pad_rows(task_path, horizon)
    dp_prior, prior_padded = _pad_rows(dp_prior, horizon)
    context_result = contact_context_from_live(
        live=live,
        history=history,
        prefix_frame=prefix_frame,
        horizon=horizon,
        args=args,
    )
    feature = np.concatenate(
        [
            np.asarray(current_state, dtype=np.float32).reshape(-1),
            task_path.reshape(-1),
            dp_prior.reshape(-1),
            np.asarray(context_result["context"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)
    if feature.shape[0] != int(bundle["feature_dim"]):
        raise RuntimeError(f"contact executor feature width {feature.shape[0]} != checkpoint feature_dim {bundle['feature_dim']}")
    x = ((feature - bundle["x_mean"]) / bundle["x_std"]).astype(np.float32)
    device = bundle["device"]
    with torch.no_grad():
        pred_norm_t, pred_progress_t, pred_logits_t = bundle["model"](
            torch.as_tensor(x[None], device=device, dtype=torch.float32)
        )
    pred_norm = pred_norm_t.detach().cpu().numpy()[0]
    pred_progress = pred_progress_t.detach().cpu().numpy()[0].astype(np.float32)
    pred_logits = pred_logits_t.detach().cpu().numpy()[0].astype(np.float32)
    pred_binary = 1.0 / (1.0 + np.exp(-pred_logits))
    pred_resid = pred_norm * bundle["residual_std"] + bundle["residual_mean"]
    scale = float(residual_scale)
    pred_resid_scaled = pred_resid.astype(np.float32) * scale
    pred_action = dp_prior.reshape(-1) + pred_resid_scaled
    pred_action = pred_action.reshape(horizon, 7).astype(np.float32)
    return {
        "ok": bool(np.isfinite(pred_action).all()),
        "controller_action_source": "contact_executor",
        "executor_checkpoint": bundle["checkpoint_path"],
        "horizon": horizon,
        "feature_dim": int(bundle["feature_dim"]),
        "target_dim": int(bundle["target_dim"]),
        "current_state_names": list(CURRENT_STATE_NAMES),
        "task_path_names": list(TASK_PATH_NAMES),
        "residual_scale": scale,
        "task_path_padded_to_horizon": bool(task_padded),
        "dp_prior_padded_to_horizon": bool(prior_padded),
        "live_contact_context": {key: value for key, value in context_result.items() if key != "context"},
        "predicted_progress_readout": pred_progress.astype(float).tolist(),
        "predicted_inserted_probability": float(pred_binary[0]),
        "predicted_dp_continuable_probability": float(pred_binary[1]),
        "dp_prior_action_stats": array_stats(dp_prior),
        "residual_action_stats": array_stats(pred_resid.reshape(horizon, 7)),
        "scaled_residual_action_stats": array_stats(pred_resid_scaled.reshape(horizon, 7)),
        "executor_action_stats": array_stats(pred_action),
        "denormalized_robot_action_chunk": pred_action.astype(float).tolist(),
        "boundary": (
            "Robot actions come from the contact/progress-conditioned executor "
            "using causal Cosmos-predicted task path, live contact context, and "
            "frozen-DP action prior. Future contact labels are not controller inputs."
        ),
    }


def candidate_executor_action_chunk(
    *,
    bundle: dict[str, Any],
    outcome_scorer: dict[str, Any] | None,
    current_state: np.ndarray,
    task_path: np.ndarray,
    dp_prior: np.ndarray,
    live: dict[str, Any],
    history: np.ndarray,
    prefix_frame: int,
    iteration: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    horizon = int(bundle["horizon"])
    task_path, task_padded = _pad_rows(task_path, horizon)
    dp_prior, prior_padded = _pad_rows(dp_prior, horizon)
    context_result = contact_context_from_live(
        live=live,
        history=history,
        prefix_frame=prefix_frame,
        horizon=horizon,
        args=args,
    )
    feature = np.concatenate(
        [
            np.asarray(current_state, dtype=np.float32).reshape(-1),
            task_path.reshape(-1),
            dp_prior.reshape(-1),
            np.asarray(context_result["context"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)
    if feature.shape[0] != int(bundle["feature_dim"]):
        raise RuntimeError(f"candidate executor feature width {feature.shape[0]} != checkpoint feature_dim {bundle['feature_dim']}")

    x = ((feature - bundle["x_mean"]) / bundle["x_std"]).astype(np.float32)
    device = bundle["device"]
    model = bundle["model"]
    raw_mean = torch.as_tensor(bundle["residual_mean"], device=device, dtype=torch.float32).reshape(1, -1)
    raw_std = torch.as_tensor(bundle["residual_std"], device=device, dtype=torch.float32).reshape(1, -1)

    with torch.no_grad():
        z = model.encode(torch.as_tensor(x[None], device=device, dtype=torch.float32))
        mean_norm, logstd = model.distribution(z)
        std = torch.exp(logstd)
        zero_resid_norm = (torch.zeros_like(mean_norm) - raw_mean) / raw_std
        candidates: list[tuple[str, Any]] = [("dp_prior", zero_resid_norm), ("mean", mean_norm)]
        for scale in list(bundle.get("candidate_scales") or [0.05, 0.1, 0.2, 0.5, 1.0]):
            raw = float(scale) * (mean_norm * raw_std + raw_mean)
            candidates.append((f"scale_{scale:g}", (raw - raw_mean) / raw_std))

        temps = list(bundle.get("candidate_temps") or [1.0])
        generator = torch.Generator(device=device)
        generator.manual_seed(int(bundle.get("seed", 20260615)) + int(prefix_frame) * 1009 + int(iteration) * 9173)
        if str(bundle.get("generator_type", "gaussian")) == "diffusion":
            candidates.extend(
                _diffusion_candidate_samples(
                    model=model,
                    z=z,
                    sample_count=int(bundle.get("candidate_samples", 24)),
                    steps=int(bundle.get("diffusion_steps", 16)),
                    beta_start=float(bundle.get("diffusion_beta_start", 1e-4)),
                    beta_end=float(bundle.get("diffusion_beta_end", 2e-2)),
                    generator=generator,
                    device=device,
                )
            )
        elif int(bundle.get("candidate_samples", 24)) > 0:
            samples_per_temp = max(1, int(bundle.get("candidate_samples", 24)) // max(1, len(temps)))
            for temp in temps:
                for sample_idx in range(samples_per_temp):
                    noise = torch.randn(mean_norm.shape, generator=generator, device=device)
                    candidates.append((f"sample_t{float(temp):g}_{sample_idx}", mean_norm + float(temp) * std * noise))

        short_prefix_steps = [
            value
            for value in _parse_positive_int_list(
                str(getattr(args, "candidate_executor_short_prefix_steps", ""))
            )
            if value < horizon
        ]
        if short_prefix_steps:
            robot_dim = int(dp_prior.shape[1])
            base_candidates = list(candidates)
            for steps in short_prefix_steps:
                keep_values = int(steps) * robot_dim
                for name, cand_norm in base_candidates:
                    base_name = _candidate_descriptor_base_name(name)
                    if base_name == "dp_prior":
                        continue
                    raw_resid = cand_norm * raw_std + raw_mean
                    short_resid = torch.zeros_like(raw_resid)
                    short_resid[:, :keep_values] = raw_resid[:, :keep_values]
                    short_norm = (short_resid - raw_mean) / raw_std
                    candidates.append((f"short{int(steps)}_{name}", short_norm))

        candidate_execute_step_overrides: dict[str, int] = {}
        source_suffix_items = source_insertion_suffix_residual_candidates(
            source_bank=bundle.get("source_suffix_bank"),
            live=live,
            scenario=str(getattr(args, "scenario", "")),
            dp_prior=dp_prior,
            horizon=horizon,
            robot_action_dim=int(dp_prior.shape[1]),
            args=args,
        )
        for item in source_suffix_items:
            actions_np = np.asarray(item["actions"], dtype=np.float32)[:horizon, : int(dp_prior.shape[1])]
            raw_resid_np = (actions_np - dp_prior[:horizon]).reshape(1, -1).astype(np.float32)
            raw_resid = torch.as_tensor(raw_resid_np, device=device, dtype=torch.float32)
            cand_norm = (raw_resid - raw_mean) / raw_std
            name = str(item["name"])
            candidates.append((name, cand_norm))
            candidate_execute_step_overrides[name] = int(item["execute_steps"])

        candidate_records: list[dict[str, Any]] = []
        scores: list[Any] = []
        raw_resids: list[Any] = []
        progress_preds: list[Any] = []
        binary_preds: list[Any] = []
        value_preds: list[Any] = []
        logprobs: list[Any] = []
        penalties: list[Any] = []
        next_state_preds: list[Any] = []
        next_state_penalties: list[Any] = []
        over_cap_flags: list[bool] = []
        dp_prior_live_gate_block_flags: list[bool] = []
        source_suffix_meta_by_name = {str(item["name"]): item for item in source_suffix_items}
        phase = str(context_result.get("phase_name") or "unknown")
        dp_prior_executable_by_live_gate = bool(
            (context_result.get("continuability_gate") or {}).get("ok", False)
        )
        phase_caps = dict(bundle.get("phase_residual_l2_caps") or {})
        cap_value = float(
            phase_caps.get(
                phase,
                phase_caps.get("__global__", float(bundle.get("selector_residual_l2_cap_max", 0.02))),
            )
        )
        next_state_target = torch.as_tensor(
            np.asarray(bundle.get("score_next_state_target", np.zeros(3, dtype=np.float32)), dtype=np.float32),
            device=device,
            dtype=torch.float32,
        ).reshape(1, -1)
        next_state_axis = torch.as_tensor(
            np.asarray(bundle.get("score_next_state_axis_weights", np.ones(3, dtype=np.float32)), dtype=np.float32),
            device=device,
            dtype=torch.float32,
        ).reshape(1, -1)
        for name, cand_norm in candidates:
            base_name = _candidate_descriptor_base_name(name)
            pred_progress, pred_logits, pred_value, pred_next_state = model.score_candidate(z, cand_norm)
            probs = torch.sigmoid(pred_logits)
            raw_resid = cand_norm * raw_std + raw_mean
            penalty = torch.mean(raw_resid**2, dim=1, keepdim=True)
            if pred_next_state is None:
                next_state_penalty = torch.zeros_like(penalty)
            else:
                next_state_penalty = torch.mean(
                    ((pred_next_state - next_state_target) * next_state_axis) ** 2,
                    dim=1,
                    keepdim=True,
                )
            logprob = -0.5 * torch.mean(((cand_norm - mean_norm) / std) ** 2 + 2.0 * logstd, dim=1, keepdim=True)
            score = (
                pred_progress[:, 1:2]
                + float(bundle["score_inserted_weight"]) * probs[:, 0:1]
                + float(bundle["score_dp_continuable_weight"]) * probs[:, 1:2]
                + float(bundle["score_value_weight"]) * pred_value
                + float(bundle["score_logprob_weight"]) * logprob
                - float(bundle["score_residual_l2_penalty"]) * penalty
                - float(bundle.get("score_next_state_weight", 0.0)) * next_state_penalty
            )
            if base_name == "mean":
                score = score - float(bundle["score_mean_source_penalty"])
            if base_name.startswith("scale_"):
                score = score - float(bundle["score_scale_source_penalty"])
                try:
                    scale_value = float(base_name.split("_", 1)[1])
                except ValueError:
                    scale_value = 0.0
                if scale_value >= 0.5:
                    score = score - float(bundle["score_large_scale_source_penalty"])
            if base_name.startswith("sample_") or base_name.startswith("diffusion_"):
                score = score - float(bundle["score_stochastic_source_penalty"])
            is_source_suffix = base_name.startswith("retrieval_resid_srcsuffix_")
            over_cap = bool(
                base_name != "dp_prior"
                and not (is_source_suffix and bool(getattr(args, "source_suffix_ignore_residual_cap", False)))
                and float(penalty.detach().cpu().numpy()[0, 0]) > cap_value
            )
            if over_cap:
                score = torch.full_like(score, -1.0e9)
            dp_prior_blocked_by_live_gate = bool(base_name == "dp_prior" and not dp_prior_executable_by_live_gate)
            if dp_prior_blocked_by_live_gate:
                score = torch.full_like(score, -1.0e9)
            scores.append(score)
            raw_resids.append(raw_resid)
            progress_preds.append(pred_progress)
            binary_preds.append(probs)
            value_preds.append(pred_value)
            next_state_preds.append(pred_next_state)
            logprobs.append(logprob)
            penalties.append(penalty)
            next_state_penalties.append(next_state_penalty)
            over_cap_flags.append(over_cap)
            dp_prior_live_gate_block_flags.append(dp_prior_blocked_by_live_gate)
        score_t = torch.cat(scores, dim=1)
        outcome_scores_np: np.ndarray | None = None
        outcome_pred_errors_np: np.ndarray | None = None
        outcome_pred_state_np: np.ndarray | None = None
        outcome_pred_progress_np: np.ndarray | None = None
        outcome_pred_binary_np: np.ndarray | None = None
        outcome_validity_reasons: list[list[str]] = [[] for _ in candidates]
        selector_mode = "candidate_executor_internal"
        if outcome_scorer is not None:
            outcome_features: list[np.ndarray] = []
            for idx, (name, _cand_norm) in enumerate(candidates):
                raw_resid_np = raw_resids[idx].detach().cpu().numpy()[0].astype(np.float32)
                candidate_action_flat = dp_prior.reshape(-1).astype(np.float32) + raw_resid_np
                descriptor_source = (
                    "retrieval_success_residual"
                    if _candidate_descriptor_base_name(name).startswith("retrieval_resid_")
                    else "checkpoint_model"
                )
                descriptor = _candidate_outcome_descriptor(name, descriptor_source)
                outcome_features.append(
                    np.concatenate([feature, candidate_action_flat, raw_resid_np, descriptor]).astype(np.float32)
                )
            outcome_x = np.stack(outcome_features).astype(np.float32)
            if outcome_x.shape[1] != int(outcome_scorer["feature_dim"]):
                raise RuntimeError(
                    f"candidate outcome scorer feature width {outcome_x.shape[1]} "
                    f"!= checkpoint feature_dim {outcome_scorer['feature_dim']}"
                )
            outcome_device = outcome_scorer["device"]
            outcome_model = outcome_scorer["model"]
            x_norm = ((outcome_x - outcome_scorer["x_mean"]) / outcome_scorer["x_std"]).astype(np.float32)
            pred_error_norm, pred_state, pred_progress, pred_logits = outcome_model(
                torch.as_tensor(x_norm, device=outcome_device, dtype=torch.float32)
            )
            error_mean = torch.as_tensor(outcome_scorer["error_mean"], device=outcome_device, dtype=torch.float32).reshape(1, -1)
            error_std = torch.as_tensor(outcome_scorer["error_std"], device=outcome_device, dtype=torch.float32).reshape(1, -1)
            pred_error = pred_error_norm * error_std + error_mean
            pred_binary = torch.sigmoid(pred_logits)
            handoff_success = (
                pred_binary[:, 1]
                if pred_binary.shape[1] > 4
                else torch.zeros_like(pred_binary[:, 0])
            )
            inserted_idx = 2 if pred_binary.shape[1] > 4 else 1
            grasped_idx = 3 if pred_binary.shape[1] > 4 else 2
            continuable_idx = 4 if pred_binary.shape[1] > 4 else 3
            state_weights = torch.as_tensor(
                outcome_scorer["score_state_abs_axis_weights"],
                device=outcome_device,
                dtype=torch.float32,
            ).reshape(1, 3)
            state_target = torch.as_tensor(
                outcome_scorer["score_state_target"],
                device=outcome_device,
                dtype=torch.float32,
            ).reshape(1, 3)
            pred_state_penalty = torch.sum(torch.abs(pred_state[:, :3] - state_target) * state_weights, dim=1)
            outcome_score = (
                -pred_error.reshape(-1)
                + float(outcome_scorer["score_success_weight"]) * pred_binary[:, 0]
                + float(outcome_scorer["score_handoff_success_weight"]) * handoff_success
                + float(outcome_scorer["score_inserted_weight"]) * pred_binary[:, inserted_idx]
                + float(outcome_scorer["score_grasped_weight"]) * pred_binary[:, grasped_idx]
                + float(outcome_scorer["score_progress_weight"]) * pred_progress[:, 0]
                + float(outcome_scorer["score_progress_delta_weight"]) * pred_progress[:, 1]
                + float(outcome_scorer["score_continuable_weight"]) * pred_binary[:, continuable_idx]
                - pred_state_penalty
            )
            over_cap_mask = torch.as_tensor(over_cap_flags, device=outcome_device, dtype=torch.bool)
            outcome_score = torch.where(over_cap_mask, torch.full_like(outcome_score, -1.0e9), outcome_score)
            dp_prior_block_mask = torch.as_tensor(
                dp_prior_live_gate_block_flags,
                device=outcome_device,
                dtype=torch.bool,
            )
            outcome_score = torch.where(dp_prior_block_mask, torch.full_like(outcome_score, -1.0e9), outcome_score)
            outcome_scores_np = outcome_score.detach().cpu().numpy().astype(np.float32)
            outcome_pred_errors_np = pred_error.detach().cpu().numpy().reshape(-1).astype(np.float32)
            outcome_pred_state_np = pred_state.detach().cpu().numpy().astype(np.float32)
            outcome_pred_progress_np = pred_progress.detach().cpu().numpy().astype(np.float32)
            outcome_pred_binary_np = pred_binary.detach().cpu().numpy().astype(np.float32)
            validity_mask_np = np.ones((len(candidates),), dtype=bool)
            min_progress_delta = float(args.candidate_outcome_scorer_min_progress_delta)
            min_continuable = float(args.candidate_outcome_scorer_min_continuable_prob)
            min_inserted = float(args.candidate_outcome_scorer_min_inserted_prob)
            for idx, (name, _cand_norm) in enumerate(candidates):
                if name == "dp_prior":
                    if dp_prior_live_gate_block_flags[idx]:
                        validity_mask_np[idx] = False
                        outcome_validity_reasons[idx].append("dp_prior_live_gate_false")
                    continue
                if idx == 0:
                    continue
                if outcome_pred_progress_np[idx, 1] < min_progress_delta:
                    validity_mask_np[idx] = False
                    outcome_validity_reasons[idx].append(
                        f"progress_delta<{min_progress_delta:g}"
                    )
                if outcome_pred_binary_np[idx, continuable_idx] < min_continuable:
                    validity_mask_np[idx] = False
                    outcome_validity_reasons[idx].append(
                        f"continuable<{min_continuable:g}"
                    )
                if outcome_pred_binary_np[idx, inserted_idx] < min_inserted:
                    validity_mask_np[idx] = False
                    outcome_validity_reasons[idx].append(
                        f"inserted<{min_inserted:g}"
                    )
            validity_mask = torch.as_tensor(validity_mask_np, device=outcome_device, dtype=torch.bool)
            outcome_score = torch.where(validity_mask, outcome_score, torch.full_like(outcome_score, -1.0e9))
            outcome_scores_np = outcome_score.detach().cpu().numpy().astype(np.float32)
            best_idx = int(np.argmax(outcome_scores_np))
            selector_mode = "candidate_outcome_scorer"
        else:
            best_idx = int(torch.argmax(score_t, dim=1).detach().cpu().numpy()[0])
        fallback_applied = False
        if outcome_scores_np is not None and best_idx != 0:
            dp_score = float(outcome_scores_np[0])
            chosen_score = float(outcome_scores_np[best_idx])
            if chosen_score < dp_score + float(args.candidate_outcome_scorer_dp_margin):
                best_idx = 0
                fallback_applied = True
        elif outcome_scores_np is None:
            fallback_phases = set(bundle.get("dp_fallback_phases") or set())
            fallback_all = "all" in fallback_phases or "*" in fallback_phases
            if (fallback_all or phase in fallback_phases) and best_idx != 0:
                dp_score = float(score_t[0, 0].detach().cpu().item())
                chosen_score = float(score_t[0, best_idx].detach().cpu().item())
                if chosen_score < dp_score + float(bundle["dp_fallback_score_margin"]):
                    best_idx = 0
                    fallback_applied = True

        default_execute_steps = max(1, min(int(horizon), int(getattr(args, "action_exec_horizon", horizon))))
        for idx, (name, _cand_norm) in enumerate(candidates):
            execute_steps = candidate_execute_step_overrides.get(name)
            if execute_steps is None:
                execute_steps = _short_prefix_steps_from_name(name) or default_execute_steps
            execute_steps = max(1, min(int(execute_steps), int(horizon)))
            probs = binary_preds[idx].detach().cpu().numpy()[0].astype(np.float32)
            progress = progress_preds[idx].detach().cpu().numpy()[0].astype(np.float32)
            raw_resid_np = raw_resids[idx].detach().cpu().numpy()[0].astype(np.float32)
            source_suffix_meta = source_suffix_meta_by_name.get(str(name))
            candidate_records.append(
                {
                    "index": int(idx),
                    "name": name,
                    "base_name": _candidate_descriptor_base_name(name),
                    "candidate_source": (
                        "source_insertion_suffix" if source_suffix_meta is not None else "checkpoint_model"
                    ),
                    "source_suffix_meta": (
                        {
                            key: value
                            for key, value in source_suffix_meta.items()
                            if key not in {"actions"}
                        }
                        if source_suffix_meta is not None
                        else None
                    ),
                    "short_prefix_steps": _short_prefix_steps_from_name(name),
                    "execute_steps": execute_steps,
                    "score": float(score_t[0, idx].detach().cpu().item()),
                    "predicted_progress": progress.astype(float).tolist(),
                    "predicted_inserted_probability": float(probs[0]),
                    "predicted_dp_continuable_probability": float(probs[1]),
                    "predicted_value": float(value_preds[idx].detach().cpu().numpy()[0, 0]),
                    "predicted_next_state_peg_head_at_hole": (
                        next_state_preds[idx].detach().cpu().numpy()[0].astype(float).tolist()
                        if next_state_preds[idx] is not None
                        else None
                    ),
                    "predicted_next_state_penalty": float(
                        next_state_penalties[idx].detach().cpu().numpy()[0, 0]
                    ),
                    "logprob": float(logprobs[idx].detach().cpu().numpy()[0, 0]),
                    "residual_l2_penalty_input": float(penalties[idx].detach().cpu().numpy()[0, 0]),
                    "selector_residual_l2_cap": float(cap_value),
                    "selector_over_cap": bool(over_cap_flags[idx]),
                    "dp_prior_blocked_by_live_gate": bool(dp_prior_live_gate_block_flags[idx]),
                    "outcome_scorer_live_valid": (
                        not outcome_validity_reasons[idx] if outcome_scores_np is not None else None
                    ),
                    "outcome_scorer_live_invalid_reasons": list(outcome_validity_reasons[idx]),
                    "outcome_scorer_score": (
                        float(outcome_scores_np[idx]) if outcome_scores_np is not None else None
                    ),
                    "outcome_scorer_predicted_error": (
                        float(outcome_pred_errors_np[idx]) if outcome_pred_errors_np is not None else None
                    ),
                    "outcome_scorer_predicted_state": (
                        outcome_pred_state_np[idx].astype(float).tolist()
                        if outcome_pred_state_np is not None
                        else None
                    ),
                    "outcome_scorer_predicted_progress": (
                        outcome_pred_progress_np[idx].astype(float).tolist()
                        if outcome_pred_progress_np is not None
                        else None
                    ),
                    "outcome_scorer_predicted_binary": (
                        outcome_pred_binary_np[idx].astype(float).tolist()
                        if outcome_pred_binary_np is not None
                        else None
                    ),
                    "outcome_scorer_predicted_handoff_success_probability": (
                        float(outcome_pred_binary_np[idx, 1])
                        if outcome_pred_binary_np is not None and outcome_pred_binary_np.shape[1] > 4
                        else None
                    ),
                    "outcome_scorer_predicted_inserted_probability": (
                        float(outcome_pred_binary_np[idx, 2 if outcome_pred_binary_np.shape[1] > 4 else 1])
                        if outcome_pred_binary_np is not None
                        else None
                    ),
                    "outcome_scorer_predicted_continuable_probability": (
                        float(outcome_pred_binary_np[idx, 4 if outcome_pred_binary_np.shape[1] > 4 else 3])
                        if outcome_pred_binary_np is not None
                        else None
                    ),
                    "residual_stats": array_stats(raw_resid_np.reshape(horizon, 7)),
                }
            )

        selected_resid = raw_resids[best_idx].detach().cpu().numpy()[0].astype(np.float32)
        selected_action_full = dp_prior.reshape(-1) + selected_resid
        selected_action_full = selected_action_full.reshape(horizon, 7).astype(np.float32)
        selected_execute_steps = int(candidate_records[best_idx]["execute_steps"])
        selected_action = selected_action_full[:selected_execute_steps].astype(np.float32)
        selected_resid_executed = selected_resid.reshape(horizon, 7)[:selected_execute_steps].astype(np.float32)
        selected_binary = binary_preds[best_idx].detach().cpu().numpy()[0].astype(np.float32)
        selected_progress = progress_preds[best_idx].detach().cpu().numpy()[0].astype(np.float32)
        selected_value = float(value_preds[best_idx].detach().cpu().numpy()[0, 0])
        selected_next_state = (
            next_state_preds[best_idx].detach().cpu().numpy()[0].astype(np.float32)
            if next_state_preds[best_idx] is not None
            else None
        )

    candidate_action_bank_payload: dict[str, Any] | None = None
    if bool(getattr(args, "save_candidate_action_bank", False)):
        names: list[str] = []
        full_actions: list[np.ndarray] = []
        raw_residuals: list[np.ndarray] = []
        execute_steps_arr: list[int] = []
        short_steps_arr: list[int] = []
        selected_mask: list[bool] = []
        dp_flat = dp_prior.reshape(-1).astype(np.float32)
        for idx, (name, _cand_norm) in enumerate(candidates):
            raw_resid_np = raw_resids[idx].detach().cpu().numpy()[0].astype(np.float32)
            full_action = (dp_flat + raw_resid_np).reshape(horizon, 7).astype(np.float32)
            names.append(str(name))
            full_actions.append(full_action)
            raw_residuals.append(raw_resid_np.reshape(horizon, 7).astype(np.float32))
            execute_steps_arr.append(int(candidate_records[idx]["execute_steps"]))
            short_steps = candidate_records[idx].get("short_prefix_steps")
            short_steps_arr.append(int(short_steps) if short_steps is not None else -1)
            selected_mask.append(idx == int(best_idx))
        candidate_action_bank_payload = {
            "schema": np.asarray(["cosmos3_candidate_action_bank_npz_v1"]),
            "candidate_names": np.asarray(names, dtype="<U128"),
            "candidate_full_actions": np.stack(full_actions).astype(np.float32),
            "candidate_raw_residuals": np.stack(raw_residuals).astype(np.float32),
            "candidate_execute_steps": np.asarray(execute_steps_arr, dtype=np.int32),
            "candidate_short_prefix_steps": np.asarray(short_steps_arr, dtype=np.int32),
            "candidate_selected_mask": np.asarray(selected_mask, dtype=bool),
            "dp_prior_actions": dp_prior.astype(np.float32),
            "horizon": np.asarray([horizon], dtype=np.int32),
            "robot_action_dim": np.asarray([7], dtype=np.int32),
            "selected_candidate_index": np.asarray([int(best_idx)], dtype=np.int32),
            "prefix_frame_index": np.asarray([int(prefix_frame)], dtype=np.int32),
            "iteration": np.asarray([int(iteration)], dtype=np.int32),
        }

    return {
        "ok": bool(np.isfinite(selected_action).all()),
        "controller_action_source": "candidate_executor",
        "executor_checkpoint": bundle["checkpoint_path"],
        "horizon": horizon,
        "feature_dim": int(bundle["feature_dim"]),
        "target_dim": int(bundle["target_dim"]),
        "next_state_dim": int(bundle.get("next_state_dim", 0)),
        "current_state_names": list(CURRENT_STATE_NAMES),
        "task_path_names": list(TASK_PATH_NAMES),
        "task_path_padded_to_horizon": bool(task_padded),
        "dp_prior_padded_to_horizon": bool(prior_padded),
        "live_contact_context": {key: value for key, value in context_result.items() if key != "context"},
        "selected_candidate": candidate_records[best_idx],
        "selected_candidate_name": candidate_records[best_idx]["name"],
        "selected_candidate_index": int(best_idx),
        "candidate_selector_mode": selector_mode,
        "candidate_outcome_scorer_checkpoint": (
            outcome_scorer.get("checkpoint_path") if outcome_scorer is not None else None
        ),
        "candidate_outcome_scorer_dp_margin": (
            float(args.candidate_outcome_scorer_dp_margin) if outcome_scorer is not None else None
        ),
        "candidate_outcome_scorer_live_filters": (
            {
                "min_progress_delta": float(args.candidate_outcome_scorer_min_progress_delta),
                "min_continuable_prob": float(args.candidate_outcome_scorer_min_continuable_prob),
                "min_inserted_prob": float(args.candidate_outcome_scorer_min_inserted_prob),
                "score_state_abs_axis_weights": np.asarray(
                    outcome_scorer.get("score_state_abs_axis_weights", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                ).astype(float).tolist(),
                "score_state_target": np.asarray(
                    outcome_scorer.get("score_state_target", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                ).astype(float).tolist(),
                "boundary": (
                    "These filters reject low-confidence non-DP action chunks "
                    "before live execution. They are controller-selection "
                    "guards only; final success still comes from real simulator "
                    "state and video review."
                ),
            }
            if outcome_scorer is not None
            else None
        ),
        "dp_fallback_applied": bool(fallback_applied),
        "dp_prior_executable_by_live_gate": bool(dp_prior_executable_by_live_gate),
        "candidate_count": int(len(candidate_records)),
        "candidate_executor_short_prefix_steps": [
            int(item)
            for item in _parse_positive_int_list(
                str(getattr(args, "candidate_executor_short_prefix_steps", ""))
            )
            if int(item) < horizon
        ],
        "candidate_executor_default_execute_steps": int(default_execute_steps),
        "action_exec_horizon": int(getattr(args, "action_exec_horizon", 0)),
        "source_insertion_suffix_bank": (
            str(np.asarray(bundle.get("source_suffix_bank", {}).get("_bank_path")).reshape(-1)[0])
            if bundle.get("source_suffix_bank") is not None
            else None
        ),
        "source_suffix_candidate_count": int(len(source_suffix_items)),
        "source_suffix_k": int(getattr(args, "source_suffix_k", 0)),
        "source_suffix_blends": _parse_blend_list(str(getattr(args, "source_suffix_blends", "1.0")))
        if int(getattr(args, "source_suffix_k", 0)) > 0
        else [],
        "source_suffix_execute_steps": int(getattr(args, "source_suffix_execute_steps", 0)),
        "source_suffix_max_distance": float(getattr(args, "source_suffix_max_distance", -1.0)),
        "source_suffix_ignore_residual_cap": bool(getattr(args, "source_suffix_ignore_residual_cap", False)),
        "selected_execute_steps": int(selected_execute_steps),
        "generator_type": str(bundle.get("generator_type", "gaussian")),
        "diffusion_steps": int(bundle.get("diffusion_steps", 0)) if str(bundle.get("generator_type", "gaussian")) == "diffusion" else 0,
        "candidate_samples": int(bundle.get("candidate_samples", 0)),
        "candidate_scales": [float(item) for item in list(bundle.get("candidate_scales") or [])],
        "candidate_rank_loss_weight": float(bundle.get("candidate_rank_loss_weight", 0.0)),
        "candidate_rank_random_count": int(bundle.get("candidate_rank_random_count", 0)),
        "candidate_rank_diffusion_count": int(bundle.get("candidate_rank_diffusion_count", 0)),
        "candidate_rank_temperature": float(bundle.get("candidate_rank_temperature", 1.0)),
        "selector_phase_residual_l2_cap": float(cap_value),
        "selector_over_cap_candidate_count": int(sum(1 for item in over_cap_flags if item)),
        "candidate_action_bank_requested": bool(getattr(args, "save_candidate_action_bank", False)),
        "_candidate_action_bank_npz_payload": candidate_action_bank_payload,
        "candidate_records": candidate_records,
        "predicted_progress_readout": selected_progress.astype(float).tolist(),
        "predicted_inserted_probability": float(selected_binary[0]),
        "predicted_dp_continuable_probability": float(selected_binary[1]),
        "predicted_value": selected_value,
        "predicted_next_state_peg_head_at_hole": (
            selected_next_state.astype(float).tolist() if selected_next_state is not None else None
        ),
        "score_next_state_weight": float(bundle.get("score_next_state_weight", 0.0)),
        "score_next_state_axis_weights": np.asarray(
            bundle.get("score_next_state_axis_weights", np.ones(3, dtype=np.float32)),
            dtype=np.float32,
        ).astype(float).tolist(),
        "score_next_state_target": np.asarray(
            bundle.get("score_next_state_target", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ).astype(float).tolist(),
        "dp_prior_action_stats": array_stats(dp_prior),
        "selected_residual_action_stats": array_stats(selected_resid_executed),
        "selected_full_horizon_residual_action_stats": array_stats(selected_resid.reshape(horizon, 7)),
        "selected_full_horizon_action_stats": array_stats(selected_action_full),
        "executor_action_stats": array_stats(selected_action),
        "denormalized_robot_action_chunk": selected_action.astype(float).tolist(),
        "boundary": (
            "Robot actions come from a stochastic candidate executor conditioned "
            "on causal Cosmos-predicted task path, live contact context, and "
            "frozen-DP action prior. The selected chunk is chosen by an "
            "action-conditioned progress/contact/value scorer; future contact "
            "labels are not controller inputs."
        ),
    }


def array_stats(arr: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    return {
        "finite": bool(np.isfinite(flat).all()),
        "num_values": int(flat.size),
        "min": float(np.min(flat)) if flat.size else None,
        "max": float(np.max(flat)) if flat.size else None,
        "mean_abs": float(np.mean(np.abs(flat))) if flat.size else None,
        "max_abs": float(np.max(np.abs(flat))) if flat.size else None,
    }


def _write_h5_dict_group(group: Any, data: Any) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            child = group.create_group(str(key)) if isinstance(value, dict) else group
            if isinstance(value, dict):
                _write_h5_dict_group(child, value)
            else:
                group.create_dataset(str(key), data=np.asarray(value))
        return
    group.attrs["payload_type"] = type(data).__name__
    group.create_dataset("value", data=np.asarray(data))


def write_live_state_snapshot(
    *,
    path: Path,
    base_env: Any,
    stack: dict[str, Any],
    prefix_frame: int,
    iteration: int,
    label: str,
) -> str:
    import h5py

    common = stack["common"]
    path.parent.mkdir(parents=True, exist_ok=True)
    state = common.to_numpy(common.batch(base_env.get_state_dict()))
    obs = common.to_numpy(common.batch(base_env.get_obs()))
    with h5py.File(path, "w") as h5:
        h5.attrs["prefix_frame_index"] = int(prefix_frame)
        h5.attrs["iteration"] = int(iteration)
        h5.attrs["label"] = str(label)
        h5.attrs["boundary"] = (
            "Live simulator state snapshot for failed-state recovery. This is "
            "not used as a privileged controller condition."
        )
        state_group = h5.create_group("state")
        _write_h5_dict_group(state_group, state)
        obs_group = h5.create_group("observation")
        _write_h5_dict_group(obs_group, obs)
    return str(path)


def emit_live_progress(event: str, **payload: Any) -> None:
    record = {"event": event, **payload}
    print(json.dumps(jsonable(record), sort_keys=True), flush=True)


def _lookup_quantile(quantiles: dict[str, Any], requested: float) -> tuple[float, str]:
    if not quantiles:
        raise ValueError("empty quantile table")
    target = float(requested)
    best_key = min(quantiles.keys(), key=lambda key: abs(float(key) - target))
    if abs(float(best_key) - target) > 1e-9:
        raise ValueError(f"quantile {requested} not found; available={sorted(quantiles.keys(), key=float)}")
    value = float(quantiles[best_key])
    if not np.isfinite(value):
        raise ValueError(f"quantile {best_key} is not finite: {value}")
    return value, str(best_key)


def apply_continuability_stats(args: argparse.Namespace) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "source": "cli_thresholds",
        "thresholds": {
            "min_rel_x": float(args.continuability_min_rel_x),
            "max_rel_x": float(args.continuability_max_rel_x),
            "max_abs_y": float(args.continuability_max_abs_y),
            "max_abs_z": float(args.continuability_max_abs_z),
            "max_hole_speed": float(args.continuability_max_hole_speed),
        },
        "profile": getattr(args, "continuability_profile", None),
    }
    if not args.continuability_stats_json:
        return profile

    stats_path = Path(args.continuability_stats_json).resolve()
    data = json.loads(stats_path.read_text())
    horizon = int(args.continuability_stats_horizon)
    key = f"within_{horizon}_steps_to_first_success"
    if key not in data:
        available = sorted(k for k in data if k.startswith("within_"))
        raise ValueError(f"missing continuability stats profile {key}; available={available}")
    stats = data[key]
    min_rel_x, x_lower_key = _lookup_quantile(
        stats.get("x_quantiles", {}),
        float(args.continuability_stats_x_lower_quantile),
    )
    max_abs_y, y_abs_key = _lookup_quantile(
        stats.get("y_abs_quantiles", {}),
        float(args.continuability_stats_abs_quantile),
    )
    max_abs_z, z_abs_key = _lookup_quantile(
        stats.get("z_abs_quantiles", {}),
        float(args.continuability_stats_abs_quantile),
    )
    max_rel_x = float(args.continuability_max_rel_x)
    x_upper_key = None
    if bool(args.continuability_stats_set_x_upper):
        max_rel_x, x_upper_key = _lookup_quantile(
            stats.get("x_quantiles", {}),
            float(args.continuability_stats_x_upper_quantile),
        )

    args.continuability_min_rel_x = min_rel_x
    args.continuability_max_rel_x = max_rel_x
    args.continuability_max_abs_y = max_abs_y
    args.continuability_max_abs_z = max_abs_z

    profile = {
        "source": "static_dp_success_manifold_stats",
        "stats_json": str(stats_path),
        "source_h5": data.get("source_h5"),
        "num_trajectories": data.get("num_trajectories"),
        "success_trajectories": data.get("success_trajectories"),
        "profile_key": key,
        "profile_n": stats.get("n"),
        "quantiles": {
            "x_lower": x_lower_key,
            "abs_y": y_abs_key,
            "abs_z": z_abs_key,
            "x_upper": x_upper_key,
        },
        "x_upper_source": "stats" if x_upper_key is not None else "cli_safety_cap",
        "thresholds": {
            "min_rel_x": float(args.continuability_min_rel_x),
            "max_rel_x": float(args.continuability_max_rel_x),
            "max_abs_y": float(args.continuability_max_abs_y),
            "max_abs_z": float(args.continuability_max_abs_z),
            "max_hole_speed": float(args.continuability_max_hole_speed),
        },
        "boundary": (
            "This profile is derived from states on successful frozen-static-DP "
            "trajectories that are within the requested remaining step horizon "
            "of first simulator success. It is a data-calibrated handoff gate, "
            "not a learned C_pi model and not success evidence by itself."
        ),
    }
    return profile


def target_actor_position_from_env_state(env_state: dict[str, Any]) -> np.ndarray:
    actors = env_state.get("actors") if isinstance(env_state, dict) else None
    if not isinstance(actors, dict) or "box_with_hole" not in actors:
        raise KeyError("source env_state is missing actors/box_with_hole")
    actor_state = np.asarray(actors["box_with_hole"], dtype=np.float32).reshape(-1, 13)[0]
    return actor_state[:3].astype(np.float32)


def select_initial_prefix_frame(env_states: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    max_frame = min(len(env_states), int(args.expected_video_frames)) - 2
    if max_frame < 0:
        raise ValueError(f"not enough env states for prefix selection: {len(env_states)}")
    if args.prefix_start_mode == "manual":
        if int(args.prefix_frame_index) < 0:
            raise ValueError("--prefix-frame-index is required when --prefix-start-mode=manual")
        prefix_frame = max(0, min(int(args.prefix_frame_index), max_frame))
        return {
            "mode": "manual",
            "prefix_frame_index": prefix_frame,
            "requested_prefix_frame_index": int(args.prefix_frame_index),
            "method_evidence_allowed": False,
            "boundary": (
                "Manual prefix selection is diagnostic only. It must not be "
                "reported as dynamic trigger/controller evidence."
            ),
        }

    initial = target_actor_position_from_env_state(env_states[0])
    min_frame = max(1, int(args.min_dynamic_prefix_frame))
    consecutive_required = max(1, int(args.target_motion_consecutive_frames))
    consecutive = 0
    first_streak_frame: int | None = None
    last_pos = initial
    records: list[dict[str, Any]] = []
    for frame in range(1, max_frame + 1):
        pos = target_actor_position_from_env_state(env_states[frame])
        delta = float(np.linalg.norm(pos - initial))
        speed = float(np.linalg.norm(pos - last_pos))
        moving = (
            frame >= min_frame
            and (
                delta >= float(args.target_motion_delta_threshold)
                or speed >= float(args.target_motion_speed_threshold)
            )
        )
        if moving:
            if consecutive == 0:
                first_streak_frame = frame
            consecutive += 1
        else:
            consecutive = 0
            first_streak_frame = None
        if moving or frame in {1, min_frame, max_frame}:
            records.append(
                {
                    "frame": frame,
                    "target_delta": delta,
                    "target_speed": speed,
                    "moving": moving,
                    "consecutive_moving": consecutive,
                }
            )
        if consecutive >= consecutive_required:
            return {
                "mode": "target_motion_onset",
                "prefix_frame_index": frame,
                "detected_frame_index": frame,
                "first_streak_frame_index": first_streak_frame,
                "triggered": True,
                "thresholds": {
                    "min_dynamic_prefix_frame": min_frame,
                    "target_motion_delta": float(args.target_motion_delta_threshold),
                    "target_motion_speed": float(args.target_motion_speed_threshold),
                    "consecutive_frames": consecutive_required,
                },
                "causal_boundary": (
                    "The start frame is the frame where the consecutive-motion "
                    "rule becomes observable from past/current target poses. "
                    "It does not back-date the prefix to the first moving frame."
                ),
                "records_tail": records[-12:],
            }
        last_pos = pos
    raise ValueError(
        "target_motion_onset trigger never fired; do not silently fall back to "
        "a hand-picked prefix for method evidence"
    )


def target_motion_update(
    *,
    frame: int,
    pos: np.ndarray,
    initial_pos: np.ndarray,
    previous_pos: np.ndarray,
    consecutive: int,
    first_streak_frame: int | None,
    args: argparse.Namespace,
) -> tuple[bool, int, int | None, dict[str, Any]]:
    delta = float(np.linalg.norm(pos - initial_pos))
    speed = float(np.linalg.norm(pos - previous_pos))
    moving = (
        frame >= max(1, int(args.min_dynamic_prefix_frame))
        and (
            delta >= float(args.target_motion_delta_threshold)
            or speed >= float(args.target_motion_speed_threshold)
        )
    )
    if moving:
        if consecutive == 0:
            first_streak_frame = int(frame)
        consecutive += 1
    else:
        consecutive = 0
        first_streak_frame = None
    triggered = bool(consecutive >= max(1, int(args.target_motion_consecutive_frames)))
    record = {
        "frame": int(frame),
        "target_delta": delta,
        "target_speed": speed,
        "moving": bool(moving),
        "consecutive_moving": int(consecutive),
        "triggered": triggered,
        "first_streak_frame_index": first_streak_frame,
    }
    return triggered, consecutive, first_streak_frame, record


def future_target_motion_scan(
    *,
    env_states: list[Any],
    start_frame: int,
    initial_pos: np.ndarray,
    previous_pos: np.ndarray,
    consecutive: int,
    first_streak_frame: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    max_frame = min(int(args.expected_action_steps), len(env_states) - 1)
    scan_records: list[dict[str, Any]] = []
    prev = np.asarray(previous_pos, dtype=np.float32).copy()
    consecutive_i = int(consecutive)
    first_streak_i = first_streak_frame
    max_delta = 0.0
    max_speed = 0.0
    for frame in range(int(start_frame) + 1, max_frame + 1):
        pos = target_actor_position_from_env_state(env_states[frame])
        triggered, consecutive_i, first_streak_i, record = target_motion_update(
            frame=frame,
            pos=pos,
            initial_pos=initial_pos,
            previous_pos=prev,
            consecutive=consecutive_i,
            first_streak_frame=first_streak_i,
            args=args,
        )
        prev = pos
        max_delta = max(max_delta, float(record["target_delta"]))
        max_speed = max(max_speed, float(record["target_speed"]))
        if record["moving"] or triggered:
            scan_records.append(record)
        if triggered:
            return {
                "would_trigger": True,
                "trigger_frame": int(frame),
                "first_streak_frame_index": first_streak_i,
                "records_tail": scan_records[-12:],
                "max_delta": max_delta,
                "max_speed": max_speed,
            }
    return {
        "would_trigger": False,
        "trigger_frame": None,
        "first_streak_frame_index": None,
        "records_tail": scan_records[-12:],
        "max_delta": max_delta,
        "max_speed": max_speed,
    }


def read_initial_frames(video_path: Path, keep: int) -> list[Any]:
    import imageio.v2 as imageio

    frames: list[Any] = []
    reader = imageio.get_reader(video_path)
    try:
        for frame in reader:
            if len(frames) >= keep:
                break
            frames.append(np.asarray(frame))
    finally:
        reader.close()
    if len(frames) != keep:
        raise ValueError(f"initial video yielded {len(frames)} frames, expected {keep}")
    return frames


def render_prefix_from_env_states(env: Any, base_env: Any, env_states: list[Any], prefix_frame: int) -> list[Any]:
    frames: list[Any] = []
    for frame_idx in range(prefix_frame + 1):
        base_env.set_state_dict(env_states[frame_idx])
        frames.append(_render_frame(env))
    base_env.set_state_dict(env_states[prefix_frame])
    return frames


def write_video(path: Path, frames: list[Any], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [np.asarray(frame) for frame in frames], fps=max(1, int(fps)))


def controller_timeline_from_summary(summary: dict[str, Any], frame_count: int) -> list[dict[str, Any]]:
    prefix_selection = summary.get("prefix_selection") or {}
    trigger_frame = prefix_selection.get("detected_frame_index")
    try:
        trigger_frame_int = int(trigger_frame) if trigger_frame is not None else None
    except Exception:
        trigger_frame_int = None
    initial_prefix = summary.get("initial_prefix_frame_index")
    try:
        initial_prefix_int = int(initial_prefix) if initial_prefix is not None else 0
    except Exception:
        initial_prefix_int = 0
    triggered = bool(prefix_selection.get("triggered", False))
    timeline: list[dict[str, Any]] = []
    for frame_idx in range(frame_count):
        if frame_idx == 0:
            controller = "INIT_OBS"
        elif triggered and frame_idx <= initial_prefix_int:
            controller = "DP_SCAN_TARGET"
        elif not triggered:
            controller = "DP_SCAN_TARGET"
        else:
            controller = "UNASSIGNED"
        timeline.append(
            {
                "frame_index": int(frame_idx),
                "controller": controller,
                "target_motion_detected": bool(trigger_frame_int is not None and frame_idx >= trigger_frame_int),
                "target_motion_trigger_frame": trigger_frame_int,
                "wm_active": False,
                "dp_active": controller.startswith("DP"),
                "prefix_role": None,
                "iteration": None,
            }
        )

    for iteration in summary.get("iterations") or []:
        iter_idx = iteration.get("iteration")
        role = iteration.get("prefix_role")
        step_type = str(iteration.get("controller_step_type") or "")
        active_controller = (
            "EXECUTOR_ACTIVE"
            if step_type in {"residual_executor_short_chunk", "contact_executor_short_chunk", "candidate_executor_short_chunk"}
            else "WM_ACTIVE"
        )
        for step in iteration.get("executed_steps") or []:
            frame_idx = int(step.get("global_action_index", -1)) + 1
            if 0 <= frame_idx < frame_count:
                timeline[frame_idx].update(
                    {
                        "controller": active_controller,
                        "target_motion_detected": True if trigger_frame_int is not None else timeline[frame_idx]["target_motion_detected"],
                        "wm_active": True,
                        "executor_active": active_controller == "EXECUTOR_ACTIVE",
                        "dp_active": False,
                        "prefix_role": role,
                        "iteration": iter_idx,
                    }
                )
        for step in iteration.get("dp_handoff_steps") or []:
            frame_idx = int(step.get("global_action_index", -1)) + 1
            if 0 <= frame_idx < frame_count:
                timeline[frame_idx].update(
                    {
                        "controller": "DP_HANDOFF",
                        "target_motion_detected": True if trigger_frame_int is not None else timeline[frame_idx]["target_motion_detected"],
                        "wm_active": False,
                        "dp_active": True,
                        "prefix_role": role,
                        "iteration": iter_idx,
                    }
                )
    return timeline


def write_annotated_video(path: Path, frames: list[Any], fps: int, summary: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    timeline = controller_timeline_from_summary(summary, len(frames))
    annotated: list[np.ndarray] = []
    for frame, meta in zip(frames, timeline):
        img = Image.fromarray(np.asarray(frame)).convert("RGB")
        draw = ImageDraw.Draw(img)
        controller = str(meta.get("controller"))
        trigger = meta.get("target_motion_trigger_frame")
        role = meta.get("prefix_role") or "-"
        iteration = meta.get("iteration")
        lines = [
            f"frame {meta['frame_index']:03d}/300  controller={controller}",
            f"target_motion_detected={bool(meta.get('target_motion_detected'))}  trigger={trigger if trigger is not None else 'none'}",
            f"wm_active={bool(meta.get('wm_active'))}  dp_active={bool(meta.get('dp_active'))}  iter={iteration if iteration is not None else '-'}  role={role}",
        ]
        text_w = max(draw.textlength(line) for line in lines) + 12
        text_h = 18 * len(lines) + 8
        draw.rectangle((6, 6, 6 + int(text_w), 6 + text_h), fill=(255, 255, 255), outline=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((12, 11 + i * 18), line, fill=(0, 0, 0))
        annotated.append(np.asarray(img))
    write_video(path, annotated, fps)
    video_inspection = inspect_video_file(path)
    counts: dict[str, int] = {}
    for meta in timeline:
        controller = str(meta.get("controller"))
        counts[controller] = counts.get(controller, 0) + 1
    return {
        "annotated_video": str(path),
        "frame_count": len(frames),
        "video_inspection": video_inspection,
        "controller_frame_counts": counts,
        "wm_active_frame_count": sum(1 for meta in timeline if meta.get("wm_active")),
        "dp_active_frame_count": sum(1 for meta in timeline if meta.get("dp_active")),
        "target_motion_detected_frame_count": sum(1 for meta in timeline if meta.get("target_motion_detected")),
        "timeline": timeline,
    }


def initialize_history_from_source(source_h5: Path, prefix_frame: int, args: argparse.Namespace) -> np.ndarray:
    export_args = SimpleNamespace(
        total_video_frames=args.expected_video_frames,
        total_action_steps=args.expected_action_steps,
    )
    arrays = _load_episode_arrays(source_h5, export_args)
    raw = _raw_vectors_from_arrays(
        arrays,
        args.expected_video_frames,
        prefix_frame + 1,
        "future_aligned_state",
    )
    if raw.shape != (args.expected_action_steps, args.expected_action_dim):
        raise ValueError(f"raw source history shape {raw.shape} is invalid")
    history = np.zeros_like(raw)
    history[:prefix_frame] = raw[:prefix_frame]
    return history


def empty_history(args: argparse.Namespace) -> np.ndarray:
    history = np.zeros((args.expected_action_steps, args.expected_action_dim), dtype=np.float32)
    denom = max(1, args.expected_action_steps - 1)
    history[:, args.expected_action_dim - 2] = np.arange(args.expected_action_steps, dtype=np.float32) / float(denom)
    return history


def live_pose_row(base_env: Any, stack: dict[str, Any], previous_hole_xyz: np.ndarray | None) -> dict[str, Any]:
    common = stack["common"]
    tcp_pose = common.to_numpy(base_env.agent.tcp.pose.raw_pose)[0].astype(np.float32)
    peg_pose = common.to_numpy(base_env.peg.pose.raw_pose)[0].astype(np.float32)
    hole_pose = common.to_numpy(base_env.box_hole_pose.raw_pose)[0].astype(np.float32)
    peg_head_at_hole = common.to_numpy((base_env.box_hole_pose.inv() * base_env.peg_head_pose).p)[0].astype(np.float32)
    hole_xyz = hole_pose[:3].astype(np.float32)
    if previous_hole_xyz is None:
        hole_velocity = np.zeros((3,), dtype=np.float32)
    else:
        hole_velocity = (hole_xyz - previous_hole_xyz.astype(np.float32)).astype(np.float32)
    grasped = bool(common.to_numpy(base_env.agent.is_grasping(base_env.peg, max_angle=20))[0])
    inserted = bool(_live_eval(base_env)["success"])
    return {
        "tcp_pose": tcp_pose,
        "peg_pose": peg_pose,
        "hole_pose": hole_pose,
        "peg_head_at_hole": peg_head_at_hole,
        "hole_velocity": hole_velocity,
        "grasped": grasped,
        "inserted": inserted,
        "hole_xyz": hole_xyz,
    }


def read_state_obs(base_env: Any, stack: dict[str, Any]) -> np.ndarray:
    common = stack["common"]
    obs = common.to_numpy(base_env.get_obs())
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 2:
        return obs[0].astype(np.float32)
    if obs.ndim == 1:
        return obs.astype(np.float32)
    raise RuntimeError(f"unexpected state obs shape {obs.shape}")


def apply_external_target_pose(
    *,
    base_env: Any,
    stack: dict[str, Any],
    env_states: list[Any],
    source_frame: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.external_target_mode == "none":
        return {"applied": False, "mode": "none", "source_frame": int(source_frame)}

    frame = int(source_frame)
    if frame < 0 or frame >= len(env_states):
        raise IndexError(f"external target source frame {frame} outside env state length {len(env_states)}")
    actors = env_states[frame].get("actors") if isinstance(env_states[frame], dict) else None
    if not isinstance(actors, dict) or "box_with_hole" not in actors:
        raise KeyError("source env_state is missing actors/box_with_hole for external target replay")

    actor_state = np.asarray(actors["box_with_hole"], dtype=np.float32).reshape(-1, 13)[0]
    position = actor_state[:3]
    quat = actor_state[3:7]
    torch = stack["torch"]
    from mani_skill.utils.structs import Pose

    p_t = torch.as_tensor(position, device=base_env.device, dtype=base_env.box.pose.p.dtype).view(1, 3)
    q_t = torch.as_tensor(quat, device=base_env.device, dtype=base_env.box.pose.q.dtype).view(1, 4)
    base_env.box.set_pose(Pose.create_from_pq(p_t, q_t))
    return {
        "applied": True,
        "mode": "source_env_state",
        "source_frame": frame,
        "actor": "box_with_hole",
        "target_actor_state_pq": actor_state[:7].astype(float).tolist(),
    }


def fill_live_history_row(
    history: np.ndarray,
    step: int,
    robot_action: np.ndarray,
    live: dict[str, Any],
) -> None:
    denom = max(1, history.shape[0] - 1)
    row = np.zeros((history.shape[1],), dtype=np.float32)
    row[0:7] = np.asarray(robot_action, dtype=np.float32).reshape(-1)[:7]
    row[7:10] = np.asarray(live["tcp_pose"], dtype=np.float32)[:3]
    row[10:13] = np.asarray(live["peg_pose"], dtype=np.float32)[:3]
    row[13:16] = np.asarray(live["hole_pose"], dtype=np.float32)[:3]
    row[16:19] = np.asarray(live["peg_head_at_hole"], dtype=np.float32)[:3]
    row[19:22] = np.asarray(live["hole_velocity"], dtype=np.float32)[:3]
    row[22] = float(bool(live["grasped"]))
    row[23] = float(bool(live["inserted"]))
    row[30] = float(step) / float(denom)
    history[step] = row


def latest_hole_speed(history: np.ndarray, prefix_frame: int) -> float:
    if prefix_frame <= 0 or prefix_frame > history.shape[0]:
        return 0.0
    velocity = np.asarray(history[prefix_frame - 1, 19:22], dtype=np.float32)
    if not np.isfinite(velocity).all():
        return float("inf")
    return float(np.linalg.norm(velocity))


def infer_prefix_role(
    *,
    history: np.ndarray,
    prefix_frame: int,
    scenario: str,
    live: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    end = max(0, min(int(prefix_frame), history.shape[0]))
    role = str(args.prefix_role)
    if role != "auto":
        return {
            "role": role,
            "source": "explicit_diagnostic_override",
            "target_delta": None,
            "hole_speed": None,
            "grasped": bool(live.get("grasped", False)),
        }

    if end <= 0:
        role = "target_pre_motion"
        return {
            "role": role,
            "source": "empty_history_no_target_motion_observed",
            "target_delta": 0.0,
            "hole_speed": 0.0,
            "grasped": bool(live.get("grasped", False)),
        }

    grasp_history = np.asarray(history[:end, 22], dtype=np.float32) if end > 0 else np.asarray([], dtype=np.float32)
    ever_grasped = bool(grasp_history.size and float(np.max(grasp_history)) > 0.5)
    recent_grasped = bool(grasp_history.size and float(np.max(grasp_history[max(0, end - 8) : end])) > 0.5)
    if "peg_drop" in scenario or "peg_disturb" in scenario:
        return {
            "role": "peg_recovery",
            "source": "scenario_peg_perturbation",
            "target_delta": None,
            "hole_speed": latest_hole_speed(history, end),
            "grasped": bool(live.get("grasped", False)),
            "ever_grasped": ever_grasped,
        }
    if ever_grasped and not recent_grasped and not bool(live.get("grasped", False)):
        return {
            "role": "peg_recovery",
            "source": "stable_lost_grasp_after_prior_grasp",
            "target_delta": None,
            "hole_speed": latest_hole_speed(history, end),
            "grasped": False,
            "ever_grasped": True,
            "recent_grasped": False,
        }

    hole = np.asarray(history[:end, 13:16], dtype=np.float32)
    hole_delta = np.linalg.norm(hole - hole[0:1], axis=1) if hole.size else np.asarray([0.0], dtype=np.float32)
    target_delta = float(np.max(hole_delta)) if hole_delta.size else 0.0
    hole_speed = latest_hole_speed(history, end)
    moved = target_delta >= float(args.target_motion_delta_threshold)
    moving_now = hole_speed >= float(args.target_motion_speed_threshold)
    if moving_now:
        role = "target_motion_observed"
        source = "observed_target_velocity"
    elif moved:
        role = "target_post_motion"
        source = "observed_target_motion_settled"
    else:
        role = "target_pre_motion"
        source = "no_target_motion_observed_yet"
    return {
        "role": role,
        "source": source,
        "target_delta": target_delta,
        "hole_speed": hole_speed,
        "grasped": bool(live.get("grasped", False)),
        "ever_grasped": ever_grasped,
        "recent_grasped": recent_grasped,
        "thresholds": {
            "target_motion_delta": float(args.target_motion_delta_threshold),
            "target_motion_speed": float(args.target_motion_speed_threshold),
        },
    }


def continuability_gate(live: dict[str, Any], history: np.ndarray, prefix_frame: int, args: argparse.Namespace) -> dict[str, Any]:
    rel = np.asarray(live["peg_head_at_hole"], dtype=np.float32).reshape(-1)[:3]
    hole_speed = latest_hole_speed(history, prefix_frame)
    checks = {
        "grasped": bool(live.get("grasped", False)),
        "rel_x_min": bool(rel[0] >= float(args.continuability_min_rel_x)),
        "rel_x_max": bool(rel[0] <= float(args.continuability_max_rel_x)),
        "rel_y_abs": bool(abs(float(rel[1])) <= float(args.continuability_max_abs_y)),
        "rel_z_abs": bool(abs(float(rel[2])) <= float(args.continuability_max_abs_z)),
        "hole_speed": bool(hole_speed <= float(args.continuability_max_hole_speed)),
    }
    return {
        "boundary": (
            "Conservative diagnostic C_pi gate from real live state only. "
            "It permits frozen-DP handoff only when the peg is still grasped, "
            "the peg head is close to the current hole frame, and recent "
            "target motion is slow. Passing this gate is not method success "
            "by itself; final live simulator success plus video review remain "
            "the authority."
        ),
        "ok": bool(all(checks.values())),
        "checks": checks,
        "peg_head_at_hole": rel.astype(float).tolist(),
        "hole_speed": hole_speed,
        "thresholds": {
            "min_rel_x": float(args.continuability_min_rel_x),
            "max_rel_x": float(args.continuability_max_rel_x),
            "max_abs_y": float(args.continuability_max_abs_y),
            "max_abs_z": float(args.continuability_max_abs_z),
            "max_hole_speed": float(args.continuability_max_hole_speed),
        },
    }


def write_partial_summary(output_root: Path, summary: dict[str, Any], base_env: Any, prefix_frame: int, observed_frames: list[Any]) -> None:
    partial = dict(summary)
    partial["partial"] = True
    partial["partial_eval"] = _live_eval(base_env)
    partial["partial_prefix_frame_index"] = int(prefix_frame)
    partial["partial_observed_frames"] = len(observed_frames)
    write_json(output_root / "live_receding_loop_partial_summary.json", partial)


def run_prefix_inference(
    *,
    args: argparse.Namespace,
    iter_dir: Path,
    prefix_video: Path,
    history_path: Path,
    prefix_frame: int,
    prefix_role: str,
    iteration: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "CONDITION_ROOT": str(Path(args.condition_root).resolve()),
            "CHECKPOINT_PATH": str(Path(args.checkpoint_path).resolve()),
            "CONFIG_FILE": str(Path(args.config_file).resolve()),
            "OUTPUT_ROOT": str(iter_dir / "cosmos_live_prefix"),
            "PREFIX_VIDEO": str(prefix_video),
            "PREFIX_FRAME_INDEX": str(prefix_frame),
            "HISTORY_ACTION_PATH": str(history_path),
            "SAMPLE_NAME": f"{args.sample_name}_iter{iteration:02d}",
            "SCENARIO": args.scenario,
            "PREFIX_ROLE": prefix_role,
            "ACTION_EXEC_HORIZON": str(args.action_exec_horizon),
            "RUN_INFERENCE": "true",
            # The outer shell may carry SOURCE_H5 for the live-loop source
            # trajectory. Per-reobservation Cosmos calls must consume the live
            # history file instead of silently falling back to source rows.
            "SOURCE_H5": "",
        }
    )
    cmd = ["bash", str((ROOT / args.cosmos_wrapper).resolve())]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    chunk_path = iter_dir / "cosmos_live_prefix" / "live_prefix_action_chunk.json"
    return json.loads(chunk_path.read_text())


def build_source_restore_prefix(
    *,
    env: Any,
    base_env: Any,
    stack: dict[str, Any],
    source_h5: Path,
    env_states: list[Any],
    args: argparse.Namespace,
) -> tuple[int, list[Any], np.ndarray, list[np.ndarray], np.ndarray, dict[str, Any], Any | None, Any | None]:
    if int(args.live_progress_interval) > 0:
        emit_live_progress(
            "live_source_restore_before_select_prefix",
            sample_name=args.sample_name,
            scenario=args.scenario,
        )
    prefix_selection = select_initial_prefix_frame(env_states, args)
    prefix_frame = int(prefix_selection["prefix_frame_index"])
    if prefix_frame >= len(env_states):
        raise ValueError(f"prefix_frame {prefix_frame} outside env state length {len(env_states)}")
    if int(args.live_progress_interval) > 0:
        emit_live_progress(
            "live_source_restore_before_set_state",
            sample_name=args.sample_name,
            scenario=args.scenario,
            prefix_frame=prefix_frame,
        )
    base_env.set_state_dict(env_states[prefix_frame])
    if args.prefix_frame_source == "render_env_states":
        if int(args.live_progress_interval) > 0:
            emit_live_progress(
                "live_source_restore_before_render_prefix",
                sample_name=args.sample_name,
                scenario=args.scenario,
                prefix_frame=prefix_frame,
            )
        observed_frames = render_prefix_from_env_states(env, base_env, env_states, prefix_frame)
        if int(args.live_progress_interval) > 0:
            emit_live_progress(
                "live_source_restore_after_render_prefix",
                sample_name=args.sample_name,
                scenario=args.scenario,
                prefix_frame=prefix_frame,
                observed_frames=len(observed_frames),
            )
    else:
        initial_video = Path(args.initial_video).resolve() if args.initial_video else None
        if initial_video is None:
            raise SystemExit("--initial-video is required when --prefix-frame-source=initial_video")
        observed_frames = read_initial_frames(initial_video, prefix_frame + 1)
    history = initialize_history_from_source(source_h5, prefix_frame, args)
    current_state_obs = read_state_obs(base_env, stack)
    dp_obs_history: list[np.ndarray] = [current_state_obs.copy(), current_state_obs.copy()]
    previous_hole_xyz = live_pose_row(base_env, stack, None)["hole_xyz"]
    prefix_selection = {
        **prefix_selection,
        "pretrigger_control_mode": "source_restore",
        "method_evidence_allowed": False,
        "boundary": (
            "This prefix was restored from source env states rather than "
            "generated by live DP execution. It is diagnostic only."
        ),
    }
    return prefix_frame, observed_frames, history, dp_obs_history, previous_hole_xyz, prefix_selection, None, None


def run_live_dp_until_trigger(
    *,
    env: Any,
    base_env: Any,
    stack: dict[str, Any],
    env_states: list[Any],
    args: argparse.Namespace,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[int, list[Any], np.ndarray, list[np.ndarray], np.ndarray, dict[str, Any], Any, Any]:
    if not args.dp_checkpoint:
        raise SystemExit("--dp-checkpoint is required for frozen_dp_until_target_motion pretrigger control")
    if args.prefix_frame_source != "render_env_states":
        raise SystemExit("frozen_dp_until_target_motion requires --prefix-frame-source=render_env_states")
    if args.prefix_start_mode != "target_motion_onset":
        raise SystemExit("frozen_dp_until_target_motion currently requires --prefix-start-mode=target_motion_onset")

    torch = stack["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
    dp_device = next(dp_agent.parameters()).device
    if int(args.live_progress_interval) > 0:
        emit_live_progress(
            "live_pretrigger_dp_loaded",
            sample_name=args.sample_name,
            scenario=args.scenario,
            device=str(dp_device),
        )

    base_env.set_state_dict(env_states[0])
    observed_frames: list[Any] = [_render_frame(env)]
    history = empty_history(args)
    current_state_obs = read_state_obs(base_env, stack)
    dp_obs_history: list[np.ndarray] = [current_state_obs.copy(), current_state_obs.copy()]
    initial_live = live_pose_row(base_env, stack, None)
    initial_hole_xyz = initial_live["hole_xyz"].copy()
    previous_hole_xyz = initial_hole_xyz.copy()
    consecutive = 0
    first_streak_frame: int | None = None
    trigger_records: list[dict[str, Any]] = []
    pretrigger_steps: list[dict[str, Any]] = []
    prefix_frame = 0
    dp_call_index = 0

    while prefix_frame < min(args.expected_action_steps, len(env_states) - 1):
        obs_seq = np.stack(dp_obs_history[-2:], axis=0)[None].astype(np.float32)
        obs_tensor = torch.as_tensor(obs_seq, device=dp_device, dtype=torch.float32)
        with torch.no_grad():
            action_seq = dp_agent.get_action(obs_tensor)
        if getattr(dp_args, "sim_backend", "physx_cpu") == "physx_cpu":
            action_seq_np = action_seq.detach().cpu().numpy()
        else:
            action_seq_np = action_seq
        act_horizon = int(action_seq_np.shape[1])
        if act_horizon <= 0:
            raise RuntimeError("frozen DP returned empty pretrigger action sequence")
        for chunk_local_i in range(act_horizon):
            if prefix_frame >= min(args.expected_action_steps, len(env_states) - 1):
                break
            step_action, action_record = _prepare_step_action(
                action_seq_np[:, chunk_local_i],
                low,
                high,
                bool(args.clip_live_actions),
            )
            obs, reward, terminated, truncated, info = env.step(step_action)
            external_target = apply_external_target_pose(
                base_env=base_env,
                stack=stack,
                env_states=env_states,
                source_frame=prefix_frame + 1,
                args=args,
            )
            current_state_obs = read_state_obs(base_env, stack)
            dp_obs_history.append(current_state_obs.copy())
            live = live_pose_row(base_env, stack, previous_hole_xyz)
            fill_live_history_row(history, prefix_frame, np.asarray(action_record["executed"]), live)
            observed_frames.append(_render_frame(env))
            prefix_frame += 1
            triggered, consecutive, first_streak_frame, trigger_record = target_motion_update(
                frame=prefix_frame,
                pos=live["hole_xyz"],
                initial_pos=initial_hole_xyz,
                previous_pos=previous_hole_xyz,
                consecutive=consecutive,
                first_streak_frame=first_streak_frame,
                args=args,
            )
            previous_hole_xyz = live["hole_xyz"]
            if trigger_record["moving"] or prefix_frame in {1, int(args.min_dynamic_prefix_frame)}:
                trigger_records.append(trigger_record)
            if int(args.live_progress_interval) > 0 and (
                prefix_frame == 1 or prefix_frame % int(args.live_progress_interval) == 0 or triggered
            ):
                emit_live_progress(
                    "live_pretrigger_step",
                    sample_name=args.sample_name,
                    scenario=args.scenario,
                    prefix_frame=int(prefix_frame),
                    triggered=bool(triggered),
                    moving=bool(trigger_record.get("moving")),
                    hole_delta=float(trigger_record.get("target_delta", 0.0)),
                    hole_speed=float(trigger_record.get("target_speed", 0.0)),
                    live_eval=_live_eval(base_env),
                )
            pretrigger_steps.append(
                {
                    "global_action_index": int(prefix_frame - 1),
                    "dp_call_index": int(dp_call_index),
                    "chunk_local_step": int(chunk_local_i),
                    "action": action_record,
                    "external_target": external_target,
                    "reward": jsonable(reward),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                    "live_eval": _live_eval(base_env),
                    "target_motion": trigger_record,
                }
            )
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                terminal_record = {
                    "frame_index": int(prefix_frame),
                    "global_action_index": int(prefix_frame - 1),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                    "live_eval": _live_eval(base_env),
                    "target_motion_at_terminal": trigger_record,
                }
                future_motion = future_target_motion_scan(
                    env_states=env_states,
                    start_frame=prefix_frame,
                    initial_pos=initial_hole_xyz,
                    previous_pos=previous_hole_xyz,
                    consecutive=consecutive,
                    first_streak_frame=first_streak_frame,
                    args=args,
                )
                if (not bool(args.full_episode_rollout)) or bool(future_motion["would_trigger"]):
                    raise RuntimeError(
                        "pretrigger frozen DP rollout terminated before target-motion trigger: "
                        + json.dumps(
                            {
                                "terminal_record": terminal_record,
                                "future_target_motion_scan": future_motion,
                                "full_episode_rollout": bool(args.full_episode_rollout),
                            },
                            sort_keys=True,
                        )
                    )

                zero_action = np.zeros((7,), dtype=np.float32)
                terminal_padding_steps: list[dict[str, Any]] = []
                while prefix_frame < min(args.expected_action_steps, len(env_states) - 1):
                    pad_global_action_index = int(prefix_frame)
                    external_target = apply_external_target_pose(
                        base_env=base_env,
                        stack=stack,
                        env_states=env_states,
                        source_frame=prefix_frame + 1,
                        args=args,
                    )
                    current_state_obs = read_state_obs(base_env, stack)
                    dp_obs_history.append(current_state_obs.copy())
                    live = live_pose_row(base_env, stack, previous_hole_xyz)
                    fill_live_history_row(history, prefix_frame, zero_action, live)
                    observed_frames.append(_render_frame(env))
                    prefix_frame += 1
                    triggered_during_padding, consecutive, first_streak_frame, pad_motion_record = target_motion_update(
                        frame=prefix_frame,
                        pos=live["hole_xyz"],
                        initial_pos=initial_hole_xyz,
                        previous_pos=previous_hole_xyz,
                        consecutive=consecutive,
                        first_streak_frame=first_streak_frame,
                        args=args,
                    )
                    previous_hole_xyz = live["hole_xyz"]
                    if pad_motion_record["moving"] or prefix_frame in {1, int(args.min_dynamic_prefix_frame)}:
                        trigger_records.append(pad_motion_record)
                    terminal_padding_steps.append(
                        {
                            "global_action_index": pad_global_action_index,
                            "dp_call_index": int(dp_call_index),
                            "chunk_local_step": None,
                            "terminal_padding": True,
                            "action": {
                                "raw": zero_action.astype(float).tolist(),
                                "executed": zero_action.astype(float).tolist(),
                                "clipped": False,
                                "within_action_space": True,
                                "max_action_space_violation": 0.0,
                            },
                            "external_target": external_target,
                            "reward": None,
                            "terminated": terminal_record["terminated"],
                            "truncated": terminal_record["truncated"],
                            "live_eval": _live_eval(base_env),
                            "target_motion": pad_motion_record,
                        }
                    )
                    if triggered_during_padding:
                        raise RuntimeError(
                            "target motion triggered during terminal padding despite future no-motion scan"
                        )

                pretrigger_steps.extend(terminal_padding_steps)
                return (
                    int(prefix_frame),
                    observed_frames,
                    history,
                    dp_obs_history,
                    previous_hole_xyz,
                    {
                        "mode": "target_motion_detector_never_triggered_after_terminal_completion",
                        "pretrigger_control_mode": "frozen_dp_until_target_motion",
                        "prefix_frame_index": int(prefix_frame),
                        "detected_frame_index": None,
                        "first_streak_frame_index": None,
                        "triggered": False,
                        "wm_triggered": False,
                        "thresholds": {
                            "min_dynamic_prefix_frame": int(args.min_dynamic_prefix_frame),
                            "target_motion_delta": float(args.target_motion_delta_threshold),
                            "target_motion_speed": float(args.target_motion_speed_threshold),
                            "consecutive_frames": int(args.target_motion_consecutive_frames),
                        },
                        "causal_boundary": (
                            "The same causal target-motion detector was used for the "
                            "entire rollout. Frozen DP reached a terminal state before "
                            "any target motion was observed, and a scan of the remaining "
                            "source target poses showed the detector would still never "
                            "fire. The final live state was therefore held to the "
                            "300-action/301-frame evidence contract without entering a "
                            "separate static-sample branch or invoking Cosmos."
                        ),
                        "pretrigger_dp_steps": len(pretrigger_steps),
                        "terminal_before_target_motion": terminal_record,
                        "terminal_padding_steps": len(terminal_padding_steps),
                        "future_target_motion_scan_after_terminal": future_motion,
                        "records_tail": trigger_records[-12:],
                    },
                    dp_agent,
                    dp_args,
                )
            if triggered:
                prefix_selection = {
                    "mode": "target_motion_onset",
                    "pretrigger_control_mode": "frozen_dp_until_target_motion",
                    "prefix_frame_index": int(prefix_frame),
                    "detected_frame_index": int(prefix_frame),
                    "first_streak_frame_index": first_streak_frame,
                    "triggered": True,
                    "wm_triggered": True,
                    "thresholds": {
                        "min_dynamic_prefix_frame": int(args.min_dynamic_prefix_frame),
                        "target_motion_delta": float(args.target_motion_delta_threshold),
                        "target_motion_speed": float(args.target_motion_speed_threshold),
                        "consecutive_frames": int(args.target_motion_consecutive_frames),
                    },
                    "causal_boundary": (
                        "The prefix was produced by live frozen-DP execution "
                        "from source frame 0 while only the target actor pose "
                        "was externally replayed from the source trajectory. "
                        "Cosmos starts at the first frame where target motion "
                        "is observable under the consecutive-frame rule."
                    ),
                    "pretrigger_dp_steps": len(pretrigger_steps),
                    "records_tail": trigger_records[-12:],
                }
                return (
                    int(prefix_frame),
                    observed_frames,
                    history,
                    dp_obs_history,
                    previous_hole_xyz,
                    prefix_selection,
                    dp_agent,
                    dp_args,
                )
        dp_call_index += 1
    return (
        int(prefix_frame),
        observed_frames,
        history,
        dp_obs_history,
        previous_hole_xyz,
        {
            "mode": "target_motion_detector_never_triggered",
            "pretrigger_control_mode": "frozen_dp_until_target_motion",
            "prefix_frame_index": int(prefix_frame),
            "detected_frame_index": None,
            "first_streak_frame_index": None,
            "triggered": False,
            "wm_triggered": False,
            "thresholds": {
                "min_dynamic_prefix_frame": int(args.min_dynamic_prefix_frame),
                "target_motion_delta": float(args.target_motion_delta_threshold),
                "target_motion_speed": float(args.target_motion_speed_threshold),
                "consecutive_frames": int(args.target_motion_consecutive_frames),
            },
            "causal_boundary": (
                "The same causal target-motion detector was used for the "
                "entire rollout. It never fired before the 300-action horizon, "
                "so the unified controller never entered WM-active mode and "
                "frozen DP produced the full episode. This is not a separate "
                "static-sample branch."
            ),
            "pretrigger_dp_steps": len(pretrigger_steps),
            "records_tail": trigger_records[-12:],
        },
        dp_agent,
        dp_args,
    )


def main() -> int:
    args = parse_args()
    require_compute_step()
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    if args.expected_video_frames != 301 or args.expected_action_steps != 300:
        raise SystemExit("active contract requires 301 video frames and 300 action steps")
    if args.expected_action_dim != len(FULL_EPISODE_VECTOR_NAMES):
        raise SystemExit("expected action dim does not match WAM vector names")
    if args.controller_action_source in {"residual_executor", "contact_executor", "candidate_executor"} and not args.executor_checkpoint:
        raise SystemExit(f"--executor-checkpoint is required for --controller-action-source={args.controller_action_source}")
    if args.candidate_outcome_scorer_checkpoint and args.controller_action_source != "candidate_executor":
        raise SystemExit("--candidate-outcome-scorer-checkpoint requires --controller-action-source=candidate_executor")
    try:
        args.continuability_profile = apply_continuability_stats(args)
    except Exception as exc:
        raise SystemExit(f"invalid continuability stats profile: {exc}") from exc

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_h5 = Path(args.source_h5).resolve()
    initial_video = Path(args.initial_video).resolve() if args.initial_video else None
    if args.prefix_frame_source == "initial_video" and initial_video is None:
        raise SystemExit("--initial-video is required when --prefix-frame-source=initial_video")
    source_uuid_text = " ".join([source_h5.name, args.sample_name, args.scenario])
    reset_seed = _parse_seed_from_text(source_uuid_text) or 0

    if int(args.live_progress_interval) > 0:
        emit_live_progress(
            "live_main_before_stack_import",
            sample_name=args.sample_name,
            scenario=args.scenario,
            output_root=str(output_root),
        )
    stack = _import_live_control_stack(ROOT)
    if int(args.live_progress_interval) > 0:
        emit_live_progress("live_main_after_stack_import", sample_name=args.sample_name, scenario=args.scenario)
    trajectory_utils = stack["trajectory_utils"]
    if int(args.live_progress_interval) > 0:
        emit_live_progress("live_main_before_make_env", sample_name=args.sample_name, scenario=args.scenario)
    env = _make_live_env(stack, Path(args.dp_manifest).resolve(), args)
    if int(args.live_progress_interval) > 0:
        emit_live_progress("live_main_after_make_env", sample_name=args.sample_name, scenario=args.scenario)
    summary: dict[str, Any] = {
        "boundary": (
            "Live receding loop scaffold. Dry-run builds causal prefix video "
            "and WAM history only. With Cosmos inference enabled, every short "
            "chunk must be followed by real re-observation before the next call."
        ),
        "evidence_boundary": (
            "Single-sample or short-run outputs are interface diagnostics. "
            "Method evidence still requires scenario-diverse live receding "
            "rollouts, continuability-gated DP handoff, real final-state "
            "success metrics, and direct video review."
        ),
        "source_h5": str(source_h5),
        "initial_video": str(initial_video) if initial_video else None,
        "condition_root": str(Path(args.condition_root).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "config_file": str(Path(args.config_file).resolve()),
        "dp_manifest": str(Path(args.dp_manifest).resolve()),
        "sample_name": args.sample_name,
        "scenario": args.scenario,
        "prefix_role_request": args.prefix_role,
        "prefix_start_mode": args.prefix_start_mode,
        "pretrigger_control_mode": args.pretrigger_control_mode,
        "manual_prefix_frame_index": int(args.prefix_frame_index),
        "prefix_frame_source": args.prefix_frame_source,
        "initial_prefix_frame_index": None,
        "max_receding_iterations": int(args.max_receding_iterations),
        "action_exec_horizon": int(args.action_exec_horizon),
        "cosmos_step_handoff_gate": bool(args.cosmos_step_handoff_gate),
        "save_live_state_snapshots": bool(args.save_live_state_snapshots),
        "live_state_snapshot_boundary": (
            "Snapshots, when enabled, record real simulator states for later "
            "failed-state recovery data construction. They are not generated "
            "model outputs and are not controller-facing conditions."
        ),
        "full_episode_rollout": bool(args.full_episode_rollout),
        "annotate_video": bool(args.annotate_video),
        "max_episode_steps": int(args.max_episode_steps),
        "expected_video_frames": int(args.expected_video_frames),
        "expected_action_steps": int(args.expected_action_steps),
        "expected_action_dim": int(args.expected_action_dim),
        "robot_action_dim": int(args.robot_action_dim),
        "clip_live_actions": bool(args.clip_live_actions),
        "external_target_mode": args.external_target_mode,
        "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()) if args.dp_checkpoint else None,
        "dp_state_key": args.dp_state_key,
        "dp_handoff_horizon": int(args.dp_handoff_horizon),
        "dp_handoff_chunk_horizon": int(args.dp_handoff_chunk_horizon),
        "continuability_thresholds": {
            "min_rel_x": float(args.continuability_min_rel_x),
            "max_rel_x": float(args.continuability_max_rel_x),
            "max_abs_y": float(args.continuability_max_abs_y),
            "max_abs_z": float(args.continuability_max_abs_z),
            "max_hole_speed": float(args.continuability_max_hole_speed),
        },
        "continuability_profile": args.continuability_profile,
        "target_motion_role_thresholds": {
            "delta": float(args.target_motion_delta_threshold),
            "speed": float(args.target_motion_speed_threshold),
        },
        "run_cosmos_inference": bool(args.run_cosmos_inference),
        "controller_action_source": args.controller_action_source,
        "executor_checkpoint": str(Path(args.executor_checkpoint).resolve()) if args.executor_checkpoint else None,
        "candidate_outcome_scorer_checkpoint": (
            str(Path(args.candidate_outcome_scorer_checkpoint).resolve())
            if args.candidate_outcome_scorer_checkpoint
            else None
        ),
        "candidate_outcome_scorer_dp_margin": float(args.candidate_outcome_scorer_dp_margin),
        "candidate_outcome_scorer_min_progress_delta": float(args.candidate_outcome_scorer_min_progress_delta),
        "candidate_outcome_scorer_min_continuable_prob": float(args.candidate_outcome_scorer_min_continuable_prob),
        "candidate_outcome_scorer_min_inserted_prob": float(args.candidate_outcome_scorer_min_inserted_prob),
        "candidate_outcome_scorer_score_state_abs_axis_weights_override": (
            str(args.candidate_outcome_scorer_score_state_abs_axis_weights) or None
        ),
        "candidate_outcome_scorer_score_state_target_override": (
            str(args.candidate_outcome_scorer_score_state_target) or None
        ),
        "candidate_executor_short_prefix_steps": str(args.candidate_executor_short_prefix_steps) or None,
        "source_insertion_suffix_bank": (
            str(Path(args.source_insertion_suffix_bank).resolve())
            if str(args.source_insertion_suffix_bank).strip()
            else None
        ),
        "source_suffix_k": int(args.source_suffix_k),
        "source_suffix_blends": _parse_blend_list(str(args.source_suffix_blends))
        if int(args.source_suffix_k) > 0
        else [],
        "source_suffix_execute_steps": int(args.source_suffix_execute_steps),
        "source_suffix_offsets": str(args.source_suffix_offsets) or None,
        "source_suffix_scenario_match": bool(args.source_suffix_scenario_match),
        "source_suffix_max_distance": float(args.source_suffix_max_distance),
        "source_suffix_ignore_residual_cap": bool(args.source_suffix_ignore_residual_cap),
        "executor_residual_scale": float(args.executor_residual_scale),
        "iterations": [],
    }
    if int(args.dp_handoff_horizon) > 0 and not args.dp_checkpoint:
        raise SystemExit("--dp-checkpoint is required when --dp-handoff-horizon > 0")
    try:
        import h5py

        with h5py.File(source_h5, "r") as h5:
            traj_name = "traj_0" if "traj_0" in h5 else next(iter(h5.keys()))
            env_states = trajectory_utils.dict_to_list_of_dicts(h5[traj_name]["env_states"])
        if int(args.live_progress_interval) > 0:
            emit_live_progress(
                "live_main_after_h5_load",
                sample_name=args.sample_name,
                scenario=args.scenario,
                traj_name=traj_name,
                env_state_count=len(env_states),
            )
        if int(args.live_progress_interval) > 0:
            emit_live_progress("live_main_before_env_reset", sample_name=args.sample_name, scenario=args.scenario)
        obs, _ = env.reset(seed=reset_seed)
        if int(args.live_progress_interval) > 0:
            emit_live_progress("live_main_after_env_reset", sample_name=args.sample_name, scenario=args.scenario)
        base_env = _get_base_env(env)
        low, high = _action_space_bounds(env, args.robot_action_dim)
        torch = stack["torch"]
        residual_executor = None
        contact_executor = None
        candidate_executor = None
        candidate_outcome_scorer = None
        if args.controller_action_source == "residual_executor":
            residual_executor = load_residual_executor_checkpoint(Path(args.executor_checkpoint), int(args.robot_action_dim))
        elif args.controller_action_source == "contact_executor":
            contact_executor = load_contact_executor_checkpoint(Path(args.executor_checkpoint), int(args.robot_action_dim))
        elif args.controller_action_source == "candidate_executor":
            candidate_executor = load_candidate_executor_checkpoint(Path(args.executor_checkpoint), int(args.robot_action_dim))
            if str(args.source_insertion_suffix_bank).strip() and int(args.source_suffix_k) > 0:
                candidate_executor["source_suffix_bank"] = load_source_insertion_suffix_bank(
                    str(args.source_insertion_suffix_bank)
                )
            if args.candidate_outcome_scorer_checkpoint:
                candidate_outcome_scorer = load_candidate_outcome_scorer_checkpoint(
                    Path(args.candidate_outcome_scorer_checkpoint)
                )
                if str(args.candidate_outcome_scorer_score_state_abs_axis_weights).strip():
                    candidate_outcome_scorer["score_state_abs_axis_weights"] = _fixed_width_vector(
                        str(args.candidate_outcome_scorer_score_state_abs_axis_weights), 3, 0.0
                    )
                if str(args.candidate_outcome_scorer_score_state_target).strip():
                    candidate_outcome_scorer["score_state_target"] = _fixed_width_vector(
                        str(args.candidate_outcome_scorer_score_state_target), 3, 0.0
                    )
        if residual_executor is not None:
            summary["residual_executor_checkpoint_metrics"] = residual_executor.get("checkpoint_metrics")
            summary["residual_executor_horizon"] = int(residual_executor["horizon"])
        if contact_executor is not None:
            summary["contact_executor_checkpoint_metrics"] = contact_executor.get("checkpoint_metrics")
            summary["contact_executor_horizon"] = int(contact_executor["horizon"])
        if candidate_executor is not None:
            summary["candidate_executor_checkpoint_metrics"] = candidate_executor.get("checkpoint_metrics")
            summary["candidate_executor_horizon"] = int(candidate_executor["horizon"])
            summary["candidate_executor_source_suffix_bank_loaded"] = bool(
                candidate_executor.get("source_suffix_bank") is not None
            )
        if candidate_outcome_scorer is not None:
            summary["candidate_outcome_scorer_checkpoint_metrics"] = candidate_outcome_scorer.get("checkpoint_metrics")
            summary["candidate_outcome_scorer_feature_dim"] = int(candidate_outcome_scorer["feature_dim"])
            summary["candidate_outcome_scorer_descriptor_names"] = candidate_outcome_scorer.get(
                "candidate_descriptor_names"
            )
            summary["candidate_outcome_scorer_score_state_abs_axis_weights"] = np.asarray(
                candidate_outcome_scorer.get("score_state_abs_axis_weights", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).astype(float).tolist()
            summary["candidate_outcome_scorer_score_state_target"] = np.asarray(
                candidate_outcome_scorer.get("score_state_target", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).astype(float).tolist()
        if args.pretrigger_control_mode == "frozen_dp_until_target_motion":
            (
                prefix_frame,
                observed_frames,
                history,
                dp_obs_history,
                previous_hole_xyz,
                prefix_selection,
                dp_agent,
                dp_args,
            ) = run_live_dp_until_trigger(
                env=env,
                base_env=base_env,
                stack=stack,
                env_states=env_states,
                args=args,
                low=low,
                high=high,
            )
        else:
            (
                prefix_frame,
                observed_frames,
                history,
                dp_obs_history,
                previous_hole_xyz,
                prefix_selection,
                dp_agent,
                dp_args,
            ) = build_source_restore_prefix(
                env=env,
                base_env=base_env,
                stack=stack,
                source_h5=source_h5,
                env_states=env_states,
                args=args,
            )
        summary["prefix_selection"] = prefix_selection
        summary["initial_prefix_frame_index"] = prefix_frame
        if prefix_frame >= len(env_states):
            raise ValueError(f"prefix_frame {prefix_frame} outside env state length {len(env_states)}")

        iteration = 0
        while prefix_frame < args.expected_action_steps and iteration < max(1, int(args.max_receding_iterations)):
            iter_dir = output_root / f"iter_{iteration:02d}_prefix_f{prefix_frame:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            prefix_video = iter_dir / "observed_prefix.mp4"
            history_path = iter_dir / "live_history_raw_action_state.json"
            last_live = live_pose_row(base_env, stack, previous_hole_xyz)
            prefix_role_info = infer_prefix_role(
                history=history,
                prefix_frame=prefix_frame,
                scenario=args.scenario,
                live=last_live,
                args=args,
            )
            write_video(prefix_video, observed_frames, args.video_fps)
            write_action_json(history_path, history)

            iter_record: dict[str, Any] = {
                "iteration": iteration,
                "prefix_frame_index": prefix_frame,
                "prefix_role": prefix_role_info["role"],
                "prefix_role_info": prefix_role_info,
                "prefix_video": str(prefix_video),
                "history_action_state": str(history_path),
                "observed_prefix_frames": len(observed_frames),
                "prefix_frame_source": args.prefix_frame_source,
                "external_target_mode": args.external_target_mode,
                "before_eval": _live_eval(base_env),
            }
            if bool(args.save_live_state_snapshots):
                iter_record["live_state_snapshot_before_controller"] = write_live_state_snapshot(
                    path=iter_dir / "live_state_before_controller.h5",
                    base_env=base_env,
                    stack=stack,
                    prefix_frame=prefix_frame,
                    iteration=iteration,
                    label="before_controller",
                )
            if not args.run_cosmos_inference:
                iter_record["dry_run_stop"] = "cosmos_inference_disabled"
                summary["iterations"].append(iter_record)
                write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                break

            pre_gate = continuability_gate(last_live, history, prefix_frame, args)
            iter_record["pre_controller_continuability_gate"] = pre_gate
            if bool(pre_gate["ok"]) and int(args.dp_handoff_horizon) > 0:
                iter_record["controller_step_type"] = "frozen_dp_short_chunk"
                iter_record["dp_handoff_steps"] = []
                dp_chunk_limit = (
                    int(args.dp_handoff_chunk_horizon)
                    if int(args.dp_handoff_chunk_horizon) > 0
                    else int(args.action_exec_horizon)
                )
                dp_steps_this_iteration = min(
                    max(1, int(dp_chunk_limit)),
                    max(1, int(args.dp_handoff_horizon)),
                )
                iter_record["dp_handoff_requested_horizon"] = int(args.dp_handoff_horizon)
                iter_record["dp_handoff_chunk_horizon_config"] = int(args.dp_handoff_chunk_horizon)
                iter_record["dp_handoff_chunk_horizon"] = int(dp_steps_this_iteration)
                iter_record["dp_boundary"] = (
                    "Frozen DP is executed only as a short reobserved chunk "
                    "after the real live state passes C_pi. This is not a "
                    "blind long takeover; after this chunk the loop either "
                    "stops on real success/termination or refreshes the prefix "
                    "and chooses DP/Cosmos again."
                )
                if dp_agent is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
                assert dp_args is not None
                dp_device = next(dp_agent.parameters()).device
                dp_call_index = 0
                while len(iter_record["dp_handoff_steps"]) < dp_steps_this_iteration and prefix_frame < args.expected_action_steps:
                    obs_seq = np.stack(dp_obs_history[-2:], axis=0)[None].astype(np.float32)
                    obs_tensor = torch.as_tensor(obs_seq, device=dp_device, dtype=torch.float32)
                    with torch.no_grad():
                        action_seq = dp_agent.get_action(obs_tensor)
                    if getattr(dp_args, "sim_backend", "physx_cpu") == "physx_cpu":
                        action_seq_np = action_seq.detach().cpu().numpy()
                    else:
                        action_seq_np = action_seq
                    act_horizon = int(action_seq_np.shape[1])
                    if act_horizon <= 0:
                        iter_record["dp_handoff_stop_reason"] = "empty_dp_action_sequence"
                        break
                    for chunk_local_i in range(min(dp_steps_this_iteration - len(iter_record["dp_handoff_steps"]), act_horizon)):
                        if prefix_frame >= args.expected_action_steps:
                            break
                        step_action, action_record = _prepare_step_action(
                            action_seq_np[:, chunk_local_i],
                            low,
                            high,
                            bool(args.clip_live_actions),
                        )
                        obs, reward, terminated, truncated, info = env.step(step_action)
                        external_target = apply_external_target_pose(
                            base_env=base_env,
                            stack=stack,
                            env_states=env_states,
                            source_frame=prefix_frame + 1,
                            args=args,
                        )
                        current_state_obs = read_state_obs(base_env, stack)
                        dp_obs_history.append(current_state_obs.copy())
                        live = live_pose_row(base_env, stack, previous_hole_xyz)
                        previous_hole_xyz = live["hole_xyz"]
                        last_live = live
                        fill_live_history_row(history, prefix_frame, np.asarray(action_record["executed"]), live)
                        observed_frames.append(_render_frame(env))
                        prefix_frame += 1
                        iter_record["dp_handoff_steps"].append(
                            {
                                "local_step": len(iter_record["dp_handoff_steps"]),
                                "global_action_index": prefix_frame - 1,
                                "dp_call_index": dp_call_index,
                                "chunk_local_step": chunk_local_i,
                                "action": action_record,
                                "external_target": external_target,
                                "reward": jsonable(reward),
                                "terminated": jsonable(terminated),
                                "truncated": jsonable(truncated),
                                "live_eval": _live_eval(base_env),
                            }
                        )
                        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                            iter_record["terminated_or_truncated"] = True
                            iter_record["stop_reason"] = "terminated_or_truncated_after_dp_short_chunk"
                            break
                        if _live_eval(base_env).get("success"):
                            iter_record["live_success_observed_during_dp_short_chunk"] = True
                            if bool(args.full_episode_rollout):
                                continue
                            iter_record["dp_handoff_stop_reason"] = "live_success_after_dp_short_chunk"
                            iter_record["stop_reason"] = "live_success_after_dp_short_chunk"
                            break
                    if iter_record.get("terminated_or_truncated") or iter_record.get("dp_handoff_stop_reason"):
                        break
                    dp_call_index += 1
                iter_record["after_dp_handoff_eval"] = _live_eval(base_env)
                iter_record["dp_handoff_executed"] = bool(iter_record["dp_handoff_steps"])
                iter_record["after_dp_handoff_continuability_gate"] = continuability_gate(last_live, history, prefix_frame, args)
                if bool(args.save_live_state_snapshots):
                    iter_record["live_state_snapshot_after_controller"] = write_live_state_snapshot(
                        path=iter_dir / "live_state_after_controller.h5",
                        base_env=base_env,
                        stack=stack,
                        prefix_frame=prefix_frame,
                        iteration=iteration,
                        label="after_dp_handoff",
                    )
                summary["iterations"].append(iter_record)
                write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                if (
                    iter_record.get("terminated_or_truncated")
                    or prefix_frame >= args.expected_action_steps
                ):
                    break
                iteration += 1
                continue

            iter_record["controller_step_type"] = (
                "residual_executor_short_chunk"
                if args.controller_action_source == "residual_executor"
                else "contact_executor_short_chunk"
                if args.controller_action_source == "contact_executor"
                else "candidate_executor_short_chunk"
                if args.controller_action_source == "candidate_executor"
                else "cosmos_rebind_short_chunk"
            )
            chunk = run_prefix_inference(
                args=args,
                iter_dir=iter_dir,
                prefix_video=prefix_video,
                history_path=history_path,
                prefix_frame=prefix_frame,
                prefix_role=str(prefix_role_info["role"]),
                iteration=iteration,
            )
            if not bool(chunk.get("ok", False)):
                raise RuntimeError(f"Cosmos action chunk extraction failed: {chunk}")
            iter_record["action_chunk_json"] = str(iter_dir / "cosmos_live_prefix" / "live_prefix_action_chunk.json")
            iter_record["sample_output_json"] = chunk.get("sample_output_json")
            iter_record["chunk_start"] = chunk["chunk_start"]
            iter_record["chunk_end_exclusive"] = chunk["chunk_end_exclusive"]
            iter_record["chunk_steps"] = chunk.get("chunk_steps")
            iter_record["normalized_robot_action_stats"] = chunk.get("normalized_robot_action_stats")
            iter_record["denormalized_robot_action_stats"] = chunk.get("denormalized_robot_action_stats")
            if args.controller_action_source == "residual_executor":
                if residual_executor is None:
                    raise RuntimeError("residual executor was not loaded")
                if dp_agent is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
                assert dp_args is not None
                executor_horizon = int(residual_executor["horizon"])
                task_result = task_path_from_cosmos_prediction(
                    sample_output_json=Path(str(chunk.get("sample_output_json"))),
                    condition_root=Path(args.condition_root).resolve(),
                    prefix_frame=prefix_frame,
                    horizon=executor_horizon,
                )
                dp_prior = dp_prior_chunk_from_agent(
                    dp_agent=dp_agent,
                    dp_args=dp_args,
                    dp_obs_history=dp_obs_history,
                    stack=stack,
                    horizon=executor_horizon,
                )
                current_state = current_executor_state_from_live(base_env, stack, last_live)
                executor_chunk = residual_executor_action_chunk(
                    bundle=residual_executor,
                    current_state=current_state,
                    task_path=task_result["task_path"],
                    dp_prior=dp_prior,
                    residual_scale=float(args.executor_residual_scale),
                )
                if not bool(executor_chunk.get("ok", False)):
                    raise RuntimeError(f"residual executor produced invalid action chunk: {executor_chunk}")
                executor_json = iter_dir / "residual_executor_action_chunk.json"
                write_json(
                    executor_json,
                    {
                        **executor_chunk,
                        "cosmos_task_path": {
                            key: value
                            for key, value in task_result.items()
                            if key != "task_path"
                        },
                        "prefix_frame_index": int(prefix_frame),
                        "raw_cosmos_action_chunk_json": str(iter_dir / "cosmos_live_prefix" / "live_prefix_action_chunk.json"),
                    },
                )
                iter_record["residual_executor_action_chunk_json"] = str(executor_json)
                iter_record["residual_executor_action_stats"] = executor_chunk.get("executor_action_stats")
                iter_record["residual_executor_dp_prior_stats"] = executor_chunk.get("dp_prior_action_stats")
                iter_record["residual_executor_residual_stats"] = executor_chunk.get("residual_action_stats")
                iter_record["residual_executor_scaled_residual_stats"] = executor_chunk.get("scaled_residual_action_stats")
                iter_record["residual_executor_residual_scale"] = executor_chunk.get("residual_scale")
                iter_record["cosmos_task_path_source"] = task_result.get("source")
                iter_record["cosmos_task_path_padded_to_horizon"] = task_result.get("padded_to_horizon")
                iter_record["residual_executor_boundary"] = executor_chunk.get("boundary")
                actions = executor_chunk.get("denormalized_robot_action_chunk") or []
            elif args.controller_action_source == "contact_executor":
                if contact_executor is None:
                    raise RuntimeError("contact executor was not loaded")
                if dp_agent is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
                assert dp_args is not None
                executor_horizon = int(contact_executor["horizon"])
                task_result = task_path_from_cosmos_prediction(
                    sample_output_json=Path(str(chunk.get("sample_output_json"))),
                    condition_root=Path(args.condition_root).resolve(),
                    prefix_frame=prefix_frame,
                    horizon=executor_horizon,
                )
                dp_prior = dp_prior_chunk_from_agent(
                    dp_agent=dp_agent,
                    dp_args=dp_args,
                    dp_obs_history=dp_obs_history,
                    stack=stack,
                    horizon=executor_horizon,
                )
                current_state = current_executor_state_from_live(base_env, stack, last_live)
                executor_chunk = contact_executor_action_chunk(
                    bundle=contact_executor,
                    current_state=current_state,
                    task_path=task_result["task_path"],
                    dp_prior=dp_prior,
                    live=last_live,
                    history=history,
                    prefix_frame=prefix_frame,
                    args=args,
                    residual_scale=float(args.executor_residual_scale),
                )
                if not bool(executor_chunk.get("ok", False)):
                    raise RuntimeError(f"contact executor produced invalid action chunk: {executor_chunk}")
                executor_json = iter_dir / "contact_executor_action_chunk.json"
                write_json(
                    executor_json,
                    {
                        **executor_chunk,
                        "cosmos_task_path": {
                            key: value
                            for key, value in task_result.items()
                            if key != "task_path"
                        },
                        "prefix_frame_index": int(prefix_frame),
                        "raw_cosmos_action_chunk_json": str(iter_dir / "cosmos_live_prefix" / "live_prefix_action_chunk.json"),
                    },
                )
                iter_record["contact_executor_action_chunk_json"] = str(executor_json)
                iter_record["contact_executor_action_stats"] = executor_chunk.get("executor_action_stats")
                iter_record["contact_executor_dp_prior_stats"] = executor_chunk.get("dp_prior_action_stats")
                iter_record["contact_executor_residual_stats"] = executor_chunk.get("residual_action_stats")
                iter_record["contact_executor_scaled_residual_stats"] = executor_chunk.get("scaled_residual_action_stats")
                iter_record["contact_executor_residual_scale"] = executor_chunk.get("residual_scale")
                iter_record["contact_executor_progress_readout"] = executor_chunk.get("predicted_progress_readout")
                iter_record["contact_executor_inserted_probability"] = executor_chunk.get("predicted_inserted_probability")
                iter_record["contact_executor_dp_continuable_probability"] = executor_chunk.get("predicted_dp_continuable_probability")
                iter_record["contact_executor_live_contact_context"] = executor_chunk.get("live_contact_context")
                iter_record["cosmos_task_path_source"] = task_result.get("source")
                iter_record["cosmos_task_path_padded_to_horizon"] = task_result.get("padded_to_horizon")
                iter_record["contact_executor_boundary"] = executor_chunk.get("boundary")
                actions = executor_chunk.get("denormalized_robot_action_chunk") or []
            elif args.controller_action_source == "candidate_executor":
                if candidate_executor is None:
                    raise RuntimeError("candidate executor was not loaded")
                if dp_agent is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dp_agent, dp_args = _load_dp_agent(env, stack, args, device)
                assert dp_args is not None
                executor_horizon = int(candidate_executor["horizon"])
                task_result = task_path_from_cosmos_prediction(
                    sample_output_json=Path(str(chunk.get("sample_output_json"))),
                    condition_root=Path(args.condition_root).resolve(),
                    prefix_frame=prefix_frame,
                    horizon=executor_horizon,
                )
                dp_prior = dp_prior_chunk_from_agent(
                    dp_agent=dp_agent,
                    dp_args=dp_args,
                    dp_obs_history=dp_obs_history,
                    stack=stack,
                    horizon=executor_horizon,
                )
                current_state = current_executor_state_from_live(base_env, stack, last_live)
                executor_chunk = candidate_executor_action_chunk(
                    bundle=candidate_executor,
                    outcome_scorer=candidate_outcome_scorer,
                    current_state=current_state,
                    task_path=task_result["task_path"],
                    dp_prior=dp_prior,
                    live=last_live,
                    history=history,
                    prefix_frame=prefix_frame,
                    iteration=iteration,
                    args=args,
                )
                if not bool(executor_chunk.get("ok", False)):
                    raise RuntimeError(f"candidate executor produced invalid action chunk: {executor_chunk}")
                executor_json = iter_dir / "candidate_executor_action_chunk.json"
                candidate_action_bank_payload = executor_chunk.pop("_candidate_action_bank_npz_payload", None)
                if candidate_action_bank_payload is not None:
                    action_bank_npz = iter_dir / "candidate_action_bank.npz"
                    np.savez_compressed(action_bank_npz, **candidate_action_bank_payload)
                    executor_chunk["candidate_action_bank_npz"] = str(action_bank_npz)
                    executor_chunk["candidate_action_bank_candidate_count"] = int(
                        np.asarray(candidate_action_bank_payload["candidate_names"]).shape[0]
                    )
                write_json(
                    executor_json,
                    {
                        **executor_chunk,
                        "cosmos_task_path": {
                            key: value
                            for key, value in task_result.items()
                            if key != "task_path"
                        },
                        "prefix_frame_index": int(prefix_frame),
                        "raw_cosmos_action_chunk_json": str(iter_dir / "cosmos_live_prefix" / "live_prefix_action_chunk.json"),
                    },
                )
                iter_record["candidate_executor_action_chunk_json"] = str(executor_json)
                iter_record["candidate_executor_action_stats"] = executor_chunk.get("executor_action_stats")
                iter_record["candidate_executor_dp_prior_stats"] = executor_chunk.get("dp_prior_action_stats")
                iter_record["candidate_executor_selected_residual_stats"] = executor_chunk.get("selected_residual_action_stats")
                iter_record["candidate_executor_selected_candidate_name"] = executor_chunk.get("selected_candidate_name")
                iter_record["candidate_executor_selected_candidate_index"] = executor_chunk.get("selected_candidate_index")
                iter_record["candidate_executor_selector_mode"] = executor_chunk.get("candidate_selector_mode")
                iter_record["candidate_action_bank_npz"] = executor_chunk.get("candidate_action_bank_npz")
                iter_record["candidate_outcome_scorer_checkpoint"] = executor_chunk.get("candidate_outcome_scorer_checkpoint")
                iter_record["candidate_executor_candidate_count"] = executor_chunk.get("candidate_count")
                iter_record["candidate_executor_dp_fallback_applied"] = executor_chunk.get("dp_fallback_applied")
                iter_record["candidate_executor_progress_readout"] = executor_chunk.get("predicted_progress_readout")
                iter_record["candidate_executor_inserted_probability"] = executor_chunk.get("predicted_inserted_probability")
                iter_record["candidate_executor_dp_continuable_probability"] = executor_chunk.get("predicted_dp_continuable_probability")
                iter_record["candidate_executor_predicted_value"] = executor_chunk.get("predicted_value")
                iter_record["candidate_executor_live_contact_context"] = executor_chunk.get("live_contact_context")
                iter_record["cosmos_task_path_source"] = task_result.get("source")
                iter_record["cosmos_task_path_padded_to_horizon"] = task_result.get("padded_to_horizon")
                iter_record["candidate_executor_boundary"] = executor_chunk.get("boundary")
                actions = executor_chunk.get("denormalized_robot_action_chunk") or []
            else:
                actions = chunk.get("denormalized_robot_action_chunk") or []
            iter_record["executed_steps"] = []
            iter_record["step_continuability_gates"] = []
            for local_i, action in enumerate(actions):
                if prefix_frame >= args.expected_action_steps:
                    break
                step_action, action_record = _prepare_step_action(action, low, high, bool(args.clip_live_actions))
                obs, reward, terminated, truncated, info = env.step(step_action)
                external_target = apply_external_target_pose(
                    base_env=base_env,
                    stack=stack,
                    env_states=env_states,
                    source_frame=prefix_frame + 1,
                    args=args,
                )
                current_state_obs = read_state_obs(base_env, stack)
                dp_obs_history.append(current_state_obs.copy())
                live = live_pose_row(base_env, stack, previous_hole_xyz)
                previous_hole_xyz = live["hole_xyz"]
                last_live = live
                fill_live_history_row(history, prefix_frame, np.asarray(action_record["executed"]), live)
                observed_frames.append(_render_frame(env))
                prefix_frame += 1
                step_record = {
                    "local_step": local_i,
                    "global_action_index": prefix_frame - 1,
                    "action": action_record,
                    "external_target": external_target,
                    "reward": jsonable(reward),
                    "terminated": jsonable(terminated),
                    "truncated": jsonable(truncated),
                    "live_eval": _live_eval(base_env),
                }
                if bool(args.cosmos_step_handoff_gate) and int(args.dp_handoff_horizon) > 0:
                    step_gate = continuability_gate(last_live, history, prefix_frame, args)
                    step_record["continuability_gate_after_step"] = step_gate
                    iter_record["step_continuability_gates"].append(
                        {
                            "local_step": local_i,
                            "global_action_index": prefix_frame - 1,
                            "gate": step_gate,
                        }
                    )
                iter_record["executed_steps"].append(step_record)
                if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                    iter_record["terminated_or_truncated"] = True
                    break
                if (
                    bool(args.cosmos_step_handoff_gate)
                    and int(args.dp_handoff_horizon) > 0
                    and bool(step_record.get("continuability_gate_after_step", {}).get("ok", False))
                ):
                    iter_record["cosmos_chunk_stop_reason"] = "step_level_continuability_gate_ok"
                    iter_record["step_level_handoff_ready"] = True
                    break
            iter_record["after_eval"] = _live_eval(base_env)
            if bool(args.save_live_state_snapshots):
                iter_record["live_state_snapshot_after_controller"] = write_live_state_snapshot(
                    path=iter_dir / "live_state_after_controller.h5",
                    base_env=base_env,
                    stack=stack,
                    prefix_frame=prefix_frame,
                    iteration=iteration,
                    label="after_cosmos_chunk",
                )
            if iter_record["after_eval"].get("success"):
                iter_record["live_success_observed_after_cosmos_chunk"] = True
                if not bool(args.full_episode_rollout):
                    iter_record["stop_reason"] = "live_success_after_cosmos_chunk"
                    summary["iterations"].append(iter_record)
                    write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                    break
            if iter_record.get("stop_reason") == "live_success_after_cosmos_chunk":
                summary["iterations"].append(iter_record)
                write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                break
            if iter_record.get("terminated_or_truncated"):
                iter_record["stop_reason"] = "terminated_or_truncated_after_cosmos_chunk"
                summary["iterations"].append(iter_record)
                write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
                break

            iter_record["dp_handoff_executed"] = False
            iter_record["post_cosmos_continuability_gate"] = continuability_gate(last_live, history, prefix_frame, args)
            summary["iterations"].append(iter_record)
            write_partial_summary(output_root, summary, base_env, prefix_frame, observed_frames)
            if iter_record.get("terminated_or_truncated") or prefix_frame >= args.expected_action_steps:
                break
            iteration += 1

        final_observed_video = output_root / "live_observed_rollout.mp4"
        write_video(final_observed_video, observed_frames, args.video_fps)
        summary["final_observed_video"] = str(final_observed_video)
        summary["final_observed_video_inspection"] = inspect_video_file(final_observed_video)
        summary["final_observed_frames"] = len(observed_frames)
        summary["final_eval"] = _live_eval(base_env)
        summary["final_prefix_frame_index"] = prefix_frame
        summary["completed_iterations"] = len(summary["iterations"])
        summary["full_episode_length_ok"] = bool(
            prefix_frame >= args.expected_action_steps
            and len(observed_frames) == args.expected_video_frames
        )
        summary["unified_detector_controller_boundary"] = (
            "Controller selection is made by one causal target-motion detector "
            "over observed hole poses. Before that detector fires, frozen DP "
            "runs. After it fires, Cosmos3 may generate short rebind chunks, "
            "and frozen DP can resume only through the same real-state C_pi "
            "gate. If the detector never fires, DP runs the full episode by "
            "the same rule; static samples are not handled by a separate "
            "scenario branch."
        )
        summary["controller_timeline"] = controller_timeline_from_summary(summary, len(observed_frames))
        summary["controller_frame_counts"] = {}
        for meta in summary["controller_timeline"]:
            controller = str(meta.get("controller"))
            summary["controller_frame_counts"][controller] = summary["controller_frame_counts"].get(controller, 0) + 1
        summary["wm_active_frame_count"] = sum(1 for meta in summary["controller_timeline"] if meta.get("wm_active"))
        summary["dp_active_frame_count"] = sum(1 for meta in summary["controller_timeline"] if meta.get("dp_active"))
        if bool(args.annotate_video):
            annotated_video = output_root / "live_observed_rollout_annotated.mp4"
            annotation_summary = write_annotated_video(
                annotated_video,
                observed_frames,
                args.video_fps,
                summary,
            )
            summary["annotated_video_summary"] = annotation_summary
            summary["final_observed_annotated_video"] = str(annotated_video)
            summary["final_observed_annotated_video_inspection"] = annotation_summary.get("video_inspection")
        inspections = [summary.get("final_observed_video_inspection")]
        if bool(args.annotate_video):
            inspections.append(summary.get("final_observed_annotated_video_inspection"))
        summary["video_file_contract_ok"] = video_inspections_match_contract(
            inspections,
            expected_video_frames=int(args.expected_video_frames),
            expected_inspection_count=2 if bool(args.annotate_video) else 1,
        )
        write_json(output_root / "live_receding_loop_summary.json", summary)
        print(json.dumps({"summary": str(output_root / "live_receding_loop_summary.json")}, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
