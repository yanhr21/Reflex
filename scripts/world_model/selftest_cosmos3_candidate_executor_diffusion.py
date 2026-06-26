#!/usr/bin/env python3
"""Login-safe smoke test for diffusion candidate-executor wiring."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_cosmos3_candidate_executor as train_mod  # noqa: E402
import run_cosmos3_live_receding_loop as live_mod  # noqa: E402


def _args(generator_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_temps="0.5,1.0",
        candidate_scales="0.05,0.1",
        candidate_samples=3,
        generator_type=generator_type,
        diffusion_steps=4,
        diffusion_beta_start=1e-4,
        diffusion_beta_end=2e-2,
        seed=123,
        score_inserted_weight=0.6,
        score_dp_continuable_weight=0.3,
        score_value_weight=0.4,
        score_logprob_weight=0.05,
        score_residual_l2_penalty=0.02,
        score_mean_source_penalty=0.0,
        score_scale_source_penalty=0.0,
        score_large_scale_source_penalty=0.0,
        score_stochastic_source_penalty=0.25,
        candidate_rank_loss_weight=0.35,
        candidate_rank_random_count=2,
        candidate_rank_diffusion_count=1 if generator_type == "diffusion" else 0,
        candidate_rank_temperature=1.0,
        dp_fallback_phases="",
        dp_fallback_score_margin=0.0,
        selector_residual_l2_cap_quantile=0.9,
        selector_residual_l2_cap_min=1e-4,
        selector_residual_l2_cap_max=100.0,
        selector_residual_l2_cap_multiplier=1.0,
    )


def _live_loop_args() -> SimpleNamespace:
    return SimpleNamespace(
        continuability_min_rel_x=-0.03,
        continuability_max_rel_x=0.04,
        continuability_max_abs_y=0.02,
        continuability_max_abs_z=0.02,
        continuability_max_hole_speed=0.05,
    )


def _assert_live_diffusion_action_chunk(bundle: dict, rng: np.random.Generator) -> None:
    horizon = int(bundle["horizon"])
    current_state = rng.normal(scale=0.05, size=(len(live_mod.CURRENT_STATE_NAMES),)).astype(np.float32)
    task_path = rng.normal(scale=0.02, size=(horizon, len(live_mod.TASK_PATH_NAMES))).astype(np.float32)
    dp_prior = rng.normal(scale=0.03, size=(horizon, 7)).astype(np.float32)
    history = np.zeros((301, 32), dtype=np.float32)
    live = {
        "peg_head_at_hole": np.asarray([-0.02, 0.004, -0.003], dtype=np.float32),
        "grasped": True,
        "inserted": False,
    }
    result = live_mod.candidate_executor_action_chunk(
        bundle=bundle,
        current_state=current_state,
        task_path=task_path,
        dp_prior=dp_prior,
        live=live,
        history=history,
        prefix_frame=8,
        iteration=2,
        args=_live_loop_args(),
    )
    assert result["ok"], result
    assert result["controller_action_source"] == "candidate_executor", result
    assert result["generator_type"] == "diffusion", result
    assert result["diffusion_steps"] == 4, result
    assert result["candidate_rank_diffusion_count"] == 1, result
    assert result["candidate_count"] == 7, result
    assert any(
        str(record["name"]).startswith("diffusion_") for record in result["candidate_records"]
    ), result
    assert result["selected_candidate"]["name"] == result["selected_candidate_name"], result
    action = np.asarray(result["denormalized_robot_action_chunk"], dtype=np.float32)
    assert action.shape == (horizon, 7), result
    assert np.isfinite(action).all(), result


def main() -> int:
    import torch

    rng = np.random.default_rng(20260615)
    feature_dim = 9
    target_dim = 14
    rows = 6
    x_norm = rng.normal(size=(rows, feature_dim)).astype(np.float32)
    prior_raw = rng.normal(scale=0.05, size=(rows, target_dim)).astype(np.float32)
    residual_raw = rng.normal(scale=0.02, size=(rows, target_dim)).astype(np.float32)
    y_raw = (prior_raw + residual_raw).astype(np.float32)
    residual_mean = residual_raw.mean(axis=0, keepdims=True).astype(np.float32)
    residual_std = np.maximum(residual_raw.std(axis=0, keepdims=True), 1e-3).astype(np.float32)
    residual_norm = ((residual_raw - residual_mean) / residual_std).astype(np.float32)
    progress_raw = rng.normal(size=(rows, 2)).astype(np.float32)
    binary_raw = (rng.random(size=(rows, 2)) > 0.5).astype(np.float32)
    used_rows = [{"current_phase": "far", "uuid": f"toy_{idx}"} for idx in range(rows)]
    caps = {"__global__": 100.0, "far": 100.0}
    indices = np.arange(rows, dtype=np.int64)
    device = torch.device("cpu")

    model = train_mod.DiffusionCandidateExecutorNet(
        feature_dim=feature_dim,
        target_dim=target_dim,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        logstd_min=-4.5,
        logstd_max=1.0,
    ).to(device)
    diffusion_metrics = train_mod.candidate_eval(
        model=model,
        x_norm=x_norm,
        y_raw=y_raw,
        prior_raw=prior_raw,
        residual_norm=residual_norm,
        residual_mean=residual_mean,
        residual_std=residual_std,
        progress_raw=progress_raw,
        binary_raw=binary_raw,
        indices=indices,
        used_rows=used_rows,
        phase_residual_l2_caps=caps,
        args=_args("diffusion"),
        device=device,
    )
    assert diffusion_metrics["generator_type"] == "diffusion", diffusion_metrics
    assert diffusion_metrics["diffusion_steps"] == 4, diffusion_metrics
    assert diffusion_metrics["num_candidate_sources"] == 7, diffusion_metrics
    assert sum(diffusion_metrics["candidate_source_counts"].values()) == rows, diffusion_metrics
    assert "selected_non_dp_candidate_count" in diffusion_metrics, diffusion_metrics
    assert "selected_diffusion_candidate_count" in diffusion_metrics, diffusion_metrics
    assert np.isfinite(diffusion_metrics["selected_action_mse"]), diffusion_metrics
    pure_dp_gate = train_mod.offline_gate_from_eval(
        {
            "teacher_progress_mse": 0.01,
            "teacher_inserted_acc": 1.0,
            "teacher_dp_continuable_acc": 1.0,
            "selected_action_mse": 0.1,
            "dp_prior_action_mse": 0.1,
            "candidate_source_counts": {"dp_prior": 6},
        }
    )
    non_dp_gate = train_mod.offline_gate_from_eval(
        {
            "teacher_progress_mse": 0.01,
            "teacher_inserted_acc": 1.0,
            "teacher_dp_continuable_acc": 1.0,
            "selected_action_mse": 0.09,
            "dp_prior_action_mse": 0.1,
            "candidate_source_counts": {"dp_prior": 5, "scale_0.05": 1},
        }
    )
    assert not pure_dp_gate
    assert non_dp_gate
    x_t = torch.from_numpy(x_norm).to(device)
    resid_t = torch.from_numpy(residual_norm).to(device)
    z_t = model.encode(x_t)
    mean_t, logstd_t = model.distribution(z_t)
    rank_loss = train_mod.candidate_rank_loss_for_batch(
        module=model,
        z=z_t,
        mean_norm=mean_t,
        logstd=logstd_t,
        resid_target_norm=resid_t,
        raw_mean=torch.from_numpy(residual_mean).to(device),
        raw_std=torch.from_numpy(residual_std).to(device),
        args=_args("diffusion"),
    )
    assert torch.isfinite(rank_loss), rank_loss

    with tempfile.TemporaryDirectory(prefix="cosmos3_candidate_diffusion_selftest_") as tmpdir:
        ckpt = Path(tmpdir) / "checkpoint_final.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "target_dim": target_dim,
                "x_mean": np.zeros((feature_dim,), dtype=np.float32),
                "x_std": np.ones((feature_dim,), dtype=np.float32),
                "residual_mean": residual_mean.reshape(-1),
                "residual_std": residual_std.reshape(-1),
                "phase_residual_l2_caps": {"__global__": 0.5, "far": 0.25},
                "args": {
                    "generator_type": "diffusion",
                    "diffusion_steps": 4,
                    "diffusion_beta_start": 1e-4,
                    "diffusion_beta_end": 2e-2,
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "logstd_min": -4.5,
                    "logstd_max": 1.0,
                    "candidate_scales": "0.05,0.1",
                    "candidate_samples": 3,
                    "candidate_rank_loss_weight": 0.35,
                    "candidate_rank_random_count": 2,
                    "candidate_rank_diffusion_count": 1,
                    "candidate_rank_temperature": 1.0,
                },
                "latest_metrics": {"eval": diffusion_metrics},
            },
            ckpt,
        )
        bundle = live_mod.load_candidate_executor_checkpoint(ckpt, robot_action_dim=7)
        assert bundle["generator_type"] == "diffusion", bundle
        assert bundle["diffusion_steps"] == 4, bundle
        assert bundle["candidate_samples"] == 3, bundle
        assert bundle["candidate_rank_loss_weight"] == 0.35, bundle
        assert bundle["candidate_rank_random_count"] == 2, bundle
        assert bundle["candidate_rank_diffusion_count"] == 1, bundle
        assert bundle["horizon"] == 2, bundle
        assert bundle["phase_residual_l2_caps"]["far"] == 0.25, bundle

        live_horizon = 2
        live_target_dim = live_horizon * 7
        live_context = live_mod.contact_context_from_live(
            live={
                "peg_head_at_hole": np.asarray([-0.02, 0.004, -0.003], dtype=np.float32),
                "grasped": True,
                "inserted": False,
            },
            history=np.zeros((301, 32), dtype=np.float32),
            prefix_frame=8,
            horizon=live_horizon,
            args=_live_loop_args(),
        )["context"]
        live_feature_dim = (
            len(live_mod.CURRENT_STATE_NAMES)
            + live_horizon * len(live_mod.TASK_PATH_NAMES)
            + live_horizon * 7
            + int(live_context.shape[0])
        )
        live_model = train_mod.DiffusionCandidateExecutorNet(
            feature_dim=live_feature_dim,
            target_dim=live_target_dim,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            logstd_min=-4.5,
            logstd_max=1.0,
        ).to(device)
        live_ckpt = Path(tmpdir) / "checkpoint_live_action_chunk.pt"
        torch.save(
            {
                "model_state_dict": live_model.state_dict(),
                "feature_dim": live_feature_dim,
                "target_dim": live_target_dim,
                "x_mean": np.zeros((live_feature_dim,), dtype=np.float32),
                "x_std": np.ones((live_feature_dim,), dtype=np.float32),
                "residual_mean": np.zeros((live_target_dim,), dtype=np.float32),
                "residual_std": np.ones((live_target_dim,), dtype=np.float32),
                "phase_residual_l2_caps": {"__global__": 100.0, "preinsert_aligned": 100.0},
                "args": {
                    "generator_type": "diffusion",
                    "diffusion_steps": 4,
                    "diffusion_beta_start": 1e-4,
                    "diffusion_beta_end": 2e-2,
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "logstd_min": -4.5,
                    "logstd_max": 1.0,
                    "candidate_scales": "0.05,0.1",
                    "candidate_samples": 3,
                    "candidate_rank_loss_weight": 0.35,
                    "candidate_rank_random_count": 2,
                    "candidate_rank_diffusion_count": 1,
                    "candidate_rank_temperature": 1.0,
                    "dp_fallback_phases": "",
                    "dp_fallback_score_margin": 0.0,
                    "selector_residual_l2_cap_max": 100.0,
                },
            },
            live_ckpt,
        )
        _assert_live_diffusion_action_chunk(
            live_mod.load_candidate_executor_checkpoint(live_ckpt, robot_action_dim=7),
            rng,
        )

    gaussian_model = train_mod.CandidateExecutorNet(
        feature_dim=feature_dim,
        target_dim=target_dim,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        logstd_min=-4.5,
        logstd_max=1.0,
    ).to(device)
    gaussian_args = _args("gaussian")
    gaussian_args.candidate_samples = 0
    gaussian_metrics = train_mod.candidate_eval(
        model=gaussian_model,
        x_norm=x_norm,
        y_raw=y_raw,
        prior_raw=prior_raw,
        residual_norm=residual_norm,
        residual_mean=residual_mean,
        residual_std=residual_std,
        progress_raw=progress_raw,
        binary_raw=binary_raw,
        indices=indices,
        used_rows=used_rows,
        phase_residual_l2_caps=caps,
        args=gaussian_args,
        device=device,
    )
    assert gaussian_metrics["generator_type"] == "gaussian", gaussian_metrics
    assert gaussian_metrics["diffusion_steps"] == 0, gaussian_metrics
    assert gaussian_metrics["num_candidate_sources"] == 4, gaussian_metrics
    x_t = torch.from_numpy(x_norm).to(device)
    resid_t = torch.from_numpy(residual_norm).to(device)
    z_t = gaussian_model.encode(x_t)
    mean_t, logstd_t = gaussian_model.distribution(z_t)
    rank_loss = train_mod.candidate_rank_loss_for_batch(
        module=gaussian_model,
        z=z_t,
        mean_norm=mean_t,
        logstd=logstd_t,
        resid_target_norm=resid_t,
        raw_mean=torch.from_numpy(residual_mean).to(device),
        raw_std=torch.from_numpy(residual_std).to(device),
        args=gaussian_args,
    )
    assert torch.isfinite(rank_loss), rank_loss

    with tempfile.TemporaryDirectory(prefix="cosmos3_candidate_gaussian_selftest_") as tmpdir:
        ckpt = Path(tmpdir) / "checkpoint_final.pt"
        torch.save(
            {
                "model_state_dict": gaussian_model.state_dict(),
                "feature_dim": feature_dim,
                "target_dim": target_dim,
                "x_mean": np.zeros((feature_dim,), dtype=np.float32),
                "x_std": np.ones((feature_dim,), dtype=np.float32),
                "residual_mean": residual_mean.reshape(-1),
                "residual_std": residual_std.reshape(-1),
                "phase_residual_l2_caps": {"__global__": 0.4, "far": 0.2},
                "args": {
                    "generator_type": "gaussian",
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "logstd_min": -4.5,
                    "logstd_max": 1.0,
                    "candidate_scales": "0.05,0.1",
                    "candidate_samples": 0,
                    "candidate_rank_loss_weight": 0.35,
                    "candidate_rank_random_count": 2,
                    "candidate_rank_diffusion_count": 0,
                    "candidate_rank_temperature": 1.0,
                },
                "latest_metrics": {"eval": gaussian_metrics},
            },
            ckpt,
        )
        bundle = live_mod.load_candidate_executor_checkpoint(ckpt, robot_action_dim=7)
        assert bundle["generator_type"] == "gaussian", bundle
        assert bundle["candidate_samples"] == 0, bundle
        assert bundle["candidate_rank_random_count"] == 2, bundle
        assert bundle["candidate_rank_diffusion_count"] == 0, bundle
        assert bundle["horizon"] == 2, bundle
        assert bundle["phase_residual_l2_caps"]["far"] == 0.2, bundle

    print(
        {
            "diffusion_sources": diffusion_metrics["num_candidate_sources"],
            "gaussian_sources": gaussian_metrics["num_candidate_sources"],
            "live_load_generator_types": ["diffusion", "gaussian"],
            "live_diffusion_action_chunk": "passed",
            "status": "passed",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
