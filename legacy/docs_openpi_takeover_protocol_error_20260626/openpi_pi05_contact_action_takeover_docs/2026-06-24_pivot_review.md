# 2026-06-24 OpenPI pi0.5 Pivot Review

## Question

Why is insertion failing, and should the base policy switch away from
Diffusion Policy?

## Current Answer

Insertion itself is not supposed to be impossible, and the evidence says it is
not impossible. The failure is that current short-horizon action generators do
not produce the actual contact/insertion motion.

Concrete evidence:

- selected causal-suffix saved replay: `16/16` valid, `0/16` direct success,
  `0/16` direct gate-ok, but DP96 succeeded on `8/16`;
- larger causal-suffix candidate replay: useful handoff headroom existed
  (`55/192` candidate+DP96 successes), but direct chunk success/gate remained
  `0/192`;
- same-prefix Policy-DROID replay generated a finite 8-step chunk and
  preserved grasp, but did not directly insert; DP96 finished after `63`
  suffix steps;
- direct-contact risk-head training met the one-GPU-hour floor, but it is now
  a historical diagnostic because the user pivot is to official OpenPI/pi0.5.

So the blocker is not mainly "the scorer is bad." The scorer can recognize
some good/bad consequences, but the candidate distribution does not contain
enough direct contact-completion behavior. The base action model is too weak
for the dynamic contact state.

## Why OpenPI/pi0.5 Is A Reasonable Next Step

The local OpenPI repo has official pi0.5 support and official pretrained
checkpoints. The current first-run config is `pi05_maniskill_peg733`, which
keeps the model OpenPI-native:

- `Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False)`;
- Libero-style LeRobot data config;
- fresh norm stats from the 733 converted dataset;
- `gs://openpi-assets/checkpoints/pi05_base/params`.

This avoids the forbidden direction of inventing another local VAE/MLP/action
diffusion layer.

## Immediate Blockers

1. The 733 trajectories have now been converted in held Slurm job `149062`
   into an OpenPI/LeRobot dataset:
   `experiments/world_model_task_rebinding/openpi/lerobot_home/yanhongru/maniskill_peg733_openpi_libero`.
   The audit
   `experiments/world_model_task_rebinding/openpi/pi05_peg733_lerobot_audit_20260625_after_h5lock_convert_auditfix_alloc149062/audit_summary.json`
   passed with `733` parquet episodes, `219900` total rows, and `300` rows
   per episode. This is data readiness only, not a model-training result.
2. The first converter duplicates the approved external RGB view as
   `wrist_image`; this is acceptable for a first official OpenPI run but may
   need a custom OpenPI-style image mask transform if it hurts.
3. OpenPI norm stats/training/inference must run only inside Slurm; none of
   that has reached model training after the pivot.
4. The first Slurm conversion attempts did not reach data conversion. The
   failures were setup/resource failures:
   `80G` step memory unavailable in allocation `148732`, dead inherited proxy
   `127.0.0.1:37890`, then a `GnuTLS recv error (-110)` while `uv` fetched
   official pinned LeRobot commit
   `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`.
5. Allocation `148732` was revoked by the cluster. A new tmux-held
   interactive allocation, job `149062`, is allocated on `server24` and is
   being preserved.
6. The conversion retry in allocation `149062` also failed before data
   conversion because the official pinned LeRobot dependency could not be
   fetched from GitHub. Two codeload tarball attempts ended with incomplete
   archives, and a filtered clone also disconnected. This is a dependency
   mirror/transport blocker, not evidence that pi0.5, Cosmos, or the 733 data
   cannot produce insertion actions.
7. A later GitHub zip archive download for the same LeRobot commit completed
   and passed `unzip -t`. The archive was extracted as a local official source
   mirror, and OpenPI's `lerobot` dependency source now points to that path.
   This does not change OpenPI's pi0.5 architecture or checkpoint weights.
8. The same dependency-source repair was applied to OpenPI's pinned `dlimp`
   commit after the next conversion retry stalled on that git fetch.
9. The wrappers now force `uv run --locked` on Linux and use compute-node
   `/tmp` for uv cache/project envs, because putting uv cache/envs on the
   shared experiment path triggered `No locks available (os error 37)`.
10. The conversion path eventually succeeded by running through the existing
    compute-node converter runtime and official local LeRobot source mirror,
    with `HDF5_USE_FILE_LOCKING=FALSE`.
11. The next blocker is official OpenPI dependency locking/install. After the
    data audit, `uv run --locked scripts/compute_norm_stats.py --config-name
    pi05_maniskill_peg733` stopped before norm stats because `uv.lock` needed
    a content-hash update after the local official LeRobot/dlimp source-mirror
    repair. A compute-node `uv lock` attempt ran about `9.5` minutes, grew the
    cache only to about `103M`, and was interrupted because it was low-CPU
    network/metadata work inside the held GPU allocation. No norm stats,
    pi0.5 training, or checkpoint exists yet.

## Next Move

The immediate operational next step is dependency repair, not model redesign:
refresh the OpenPI lock metadata for the exact local official LeRobot/dlimp
source mirrors, pre-stage the official OpenPI wheel/cache if needed, then run
official OpenPI norm stats and at least one GPU-hour of pi0.5 finetuning. Only
after that should saved snapshot replay determine whether pi0.5 can generate
direct insertion chunks. Do not change the OpenPI model architecture or
official pi0.5 checkpoint path.
