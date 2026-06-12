"""Compatibility wrapper for ManiSkill diffusion-policy evaluation.

The upstream DP evaluator assumes vector env infos always contain
``final_info``. With the current ManiSkill/Gymnasium CPU wrapper, episode
metrics can instead arrive directly under ``episode``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from mani_skill.utils import common
from tqdm import tqdm


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _append_metric(eval_metrics, key: str, value: Any, mask: np.ndarray | None):
    arr = _as_numpy(value)
    if mask is not None and arr.ndim > 0 and arr.shape[0] == mask.shape[0]:
        arr = arr[mask]
    if arr.ndim == 0:
        eval_metrics[key].append(arr)
    else:
        for item in arr:
            eval_metrics[key].append(item)


def _append_episode_info(eval_metrics, episode_info: Any, mask: np.ndarray | None):
    if isinstance(episode_info, dict):
        for key, value in episode_info.items():
            if key.startswith("_"):
                continue
            _append_metric(eval_metrics, key, value, mask)
        return

    for item in episode_info:
        if item is None:
            continue
        nested = item.get("episode", item)
        for key, value in nested.items():
            if key.startswith("_"):
                continue
            _append_metric(eval_metrics, key, value, None)


def _collect_final_metrics(eval_metrics, info: dict, truncated: Any) -> int:
    truncated_mask = _as_numpy(truncated).astype(bool).reshape(-1)
    if "final_info" in info:
        mask = _as_numpy(info.get("_final_info", truncated_mask)).astype(bool).reshape(-1)
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "episode" in final_info:
            _append_episode_info(eval_metrics, final_info["episode"], mask)
        else:
            _append_episode_info(eval_metrics, final_info, mask)
        return int(mask.sum())

    if "episode" in info:
        mask = _as_numpy(info.get("_episode", truncated_mask)).astype(bool).reshape(-1)
        _append_episode_info(eval_metrics, info["episode"], mask)
        return int(mask.sum())

    raise KeyError(f"Could not find episode metrics in info keys: {sorted(info.keys())}")


def evaluate(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    agent.eval()
    pbar = tqdm(total=n) if progress_bar else None
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, _ = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, _, _, truncated, info = eval_envs.step(action_seq[:, i])
                if _as_numpy(truncated).any():
                    break

            if _as_numpy(truncated).any():
                completed = _collect_final_metrics(eval_metrics, info, truncated)
                eps_count += completed
                if pbar is not None:
                    pbar.update(completed)
    agent.train()
    if pbar is not None:
        pbar.close()
    for key in list(eval_metrics.keys()):
        eval_metrics[key] = np.stack(eval_metrics[key])
    return eval_metrics
