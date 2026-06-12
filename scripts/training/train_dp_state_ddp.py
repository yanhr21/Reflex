#!/usr/bin/env python3
"""DDP training entry for ManiSkill state Diffusion Policy.

This keeps the official ManiSkill state-DP model, dataset, and evaluation
semantics, while adding distributed sampling, rank-0 eval, and explicit global
batch accounting.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import tyro
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import train as ms_train
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import worker_init_fn
from dp_eval_compat import evaluate


@dataclass
class Args(ms_train.Args):
    run_dir: Optional[str] = None
    """Project-local output directory. Defaults to runs/<exp_name>."""
    global_batch_size: int = 4096
    """Global batch size across all DDP ranks."""
    per_gpu_batch_size: Optional[int] = None
    """Per-rank batch size. If unset, computed from global_batch_size/world_size."""
    ddp_backend: str = "nccl"
    """Distributed backend."""
    eval_on_start: bool = False
    """Whether rank 0 evaluates before the first optimizer step."""
    save_final: bool = True
    """Always save final checkpoint on rank 0."""


class LossWrapper(nn.Module):
    def __init__(self, agent: nn.Module):
        super().__init__()
        self.agent = agent

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        return self.agent.compute_loss(obs_seq=obs_seq, action_seq=action_seq)


def init_distributed(args: Args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA devices")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.ddp_backend, init_method="env://")

    return rank, local_rank, world_size


def cleanup_distributed(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return value.detach()
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= world_size
    return value


def ddp_barrier(world_size: int, local_rank: int):
    if world_size <= 1:
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def make_infinite_loader(loader: DataLoader, sampler: DistributedSampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def assert_dataset_control_mode(args: Args):
    if not args.demo_path.endswith(".h5"):
        return
    json_file = args.demo_path[:-2] + "json"
    with open(json_file, "r") as f:
        demo_info = json.load(f)
    if "control_mode" in demo_info["env_info"]["env_kwargs"]:
        control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
    elif "control_mode" in demo_info["episodes"][0]:
        control_mode = demo_info["episodes"][0]["control_mode"]
    else:
        raise RuntimeError("Control mode not found in demo json")
    assert control_mode == args.control_mode, (
        f"Control mode mismatched. Dataset has {control_mode}, "
        f"but args has {args.control_mode}"
    )


def write_manifest(run_dir: Path, args: Args, rank: int, world_size: int):
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rank": rank,
        "world_size": world_size,
        "args": vars(args),
        "env": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_NODELIST": os.environ.get("SLURM_JOB_NODELIST"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    with (run_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main():
    args = tyro.cli(Args)
    rank, local_rank, world_size = init_distributed(args)

    if args.per_gpu_batch_size is None:
        if args.global_batch_size % world_size != 0:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} is not divisible by "
                f"world_size={world_size}"
            )
        args.per_gpu_batch_size = args.global_batch_size // world_size
    else:
        args.global_batch_size = args.per_gpu_batch_size * world_size

    if args.exp_name is None:
        args.exp_name = f"dp_state_ddp_{args.env_id}_{int(time.time())}"
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / args.exp_name
    if is_rank0(rank):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        write_manifest(run_dir, args, rank, world_size)

    ddp_barrier(world_size, local_rank)

    assert_dataset_control_mode(args)
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.max_episode_steps is not None

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not torch.cuda.is_available() and args.cuda:
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    ms_train.args = args
    ms_train.device = device

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs if is_rank0(rank) else 1,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=str(run_dir / "videos") if (is_rank0(rank) and args.capture_video) else None,
    )

    if is_rank0(rank):
        writer = SummaryWriter(str(run_dir))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
        )
    else:
        writer = None

    dataset = ms_train.SmallDemoDataset_DiffusionPolicy(
        args.demo_path, device, num_traj=args.num_demos
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_gpu_batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    infinite_loader = make_infinite_loader(loader, sampler)

    agent = ms_train.Agent(eval_envs, args).to(device)
    ddp_model = DistributedDataParallel(
        LossWrapper(agent),
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
    )
    optimizer = optim.AdamW(
        params=ddp_model.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    if is_rank0(rank):
        ema = EMAModel(parameters=agent.parameters(), power=0.75)
        ema_agent = ms_train.Agent(eval_envs, args).to(device)
    else:
        ema = None
        ema_agent = None

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    def save_ckpt(iteration: int, tag: str):
        if not is_rank0(rank):
            return
        assert ema is not None and ema_agent is not None
        ema.copy_to(ema_agent.parameters())
        ckpt_path = run_dir / "checkpoints" / f"{tag}.pt"
        torch.save(
            {
                "agent": agent.state_dict(),
                "ema_agent": ema_agent.state_dict(),
                "args": vars(args),
                "iteration": iteration,
                "world_size": world_size,
                "global_batch_size": args.global_batch_size,
                "per_gpu_batch_size": args.per_gpu_batch_size,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}", flush=True)

    def evaluate_and_save_best(iteration: int):
        if iteration % args.eval_freq != 0:
            return
        ddp_barrier(world_size, local_rank)
        if not is_rank0(rank):
            ddp_barrier(world_size, local_rank)
            return

        last_tick = time.time()
        assert ema is not None and ema_agent is not None
        ema.copy_to(ema_agent.parameters())
        eval_metrics = evaluate(
            args.num_eval_episodes, ema_agent, eval_envs, device, args.sim_backend
        )
        timings["eval"] += time.time() - last_tick

        print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes", flush=True)
        for key in list(eval_metrics.keys()):
            eval_metrics[key] = np.mean(eval_metrics[key])
            writer.add_scalar(f"eval/{key}", eval_metrics[key], iteration)
            print(f"{key}: {eval_metrics[key]:.4f}", flush=True)

        for key in ["success_once", "success_at_end"]:
            if key in eval_metrics and eval_metrics[key] > best_eval_metrics[key]:
                best_eval_metrics[key] = eval_metrics[key]
                save_ckpt(iteration, f"best_eval_{key}")
                print(
                    f"New best {key}_rate: {eval_metrics[key]:.4f}. Saving checkpoint.",
                    flush=True,
                )
        ddp_barrier(world_size, local_rank)

    if is_rank0(rank):
        print(
            "DDP config: "
            f"world_size={world_size}, global_batch_size={args.global_batch_size}, "
            f"per_gpu_batch_size={args.per_gpu_batch_size}, total_iters={args.total_iters}",
            flush=True,
        )
        pbar = tqdm(total=args.total_iters, dynamic_ncols=True)
    else:
        pbar = None

    if args.eval_on_start:
        evaluate_and_save_best(0)

    agent.train()
    last_tick = time.time()
    for iteration in range(args.total_iters):
        data_batch = next(infinite_loader)
        timings["data_loading"] += time.time() - last_tick

        last_tick = time.time()
        total_loss = ddp_model(data_batch["observations"], data_batch["actions"])
        timings["forward"] += time.time() - last_tick

        last_tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        timings["backward"] += time.time() - last_tick

        if is_rank0(rank):
            last_tick = time.time()
            assert ema is not None
            ema.step(agent.parameters())
            timings["ema"] += time.time() - last_tick

            mean_loss = reduce_mean(total_loss, world_size)
            if iteration % args.log_freq == 0:
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                writer.add_scalar("losses/total_loss", mean_loss.item(), iteration)
                for key, value in timings.items():
                    writer.add_scalar(f"time/{key}", value, iteration)
                print(
                    f"iter={iteration} loss={mean_loss.item():.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g}",
                    flush=True,
                )
            if args.save_freq is not None and iteration % args.save_freq == 0:
                save_ckpt(iteration, str(iteration))
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": mean_loss.item()})
        else:
            reduce_mean(total_loss, world_size)

        evaluate_and_save_best(iteration)
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    if args.save_final:
        save_ckpt(args.total_iters, "final")

    if pbar is not None:
        pbar.close()
    eval_envs.close()
    if writer is not None:
        writer.close()
    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
