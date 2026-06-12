# Cluster sapien GPU rendering — final config (2026-05-09)

## TL;DR

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:15:00 \
    -u --export=ALL,VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json,DISPLAY= \
    .venv/bin/python -u your_render_script.py
```

The single magic env var: `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.

For ManiSkill trajectory replay/render jobs that open HDF5 files on the shared
filesystem, also set:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

Smoke job `94247` failed with `OSError: No locks available` until this was set;
retry job `94253` succeeded.

Node-specific failures from old jobs are diagnostic observations only. Do not
turn them into a static blacklist. For current jobs, use live Slurm state,
`sbatch --test-only`, and targeted canaries. If a current drain/down node must
be excluded, pass that exclusion at submission time and record the live evidence
in the job manifest; do not bake it into wrapper source.

## Why every other path fails

| Setting | Result | Symptom |
|---|---|---|
| (no VK_ICD_FILENAMES) | sapien scans `/usr/share/vulkan/icd.d/` only — nvidia_icd.json **NOT there** on this cluster | `Failed to find Vulkan ICD file` warning + lavapipe CPU fallback (100x slower, hangs in ManiSkill) |
| `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` | lavapipe (Mesa CPU Vulkan) | works for minimal sapien (4.4s/frame) but `RenderSystem(sapien.Device('cpu'))` rejects → ManiSkill cannot init |
| `VK_ICD_FILENAMES=<sapien_bundled_nvidia_icd.json>` | sapien's bundled `vulkan_library/nvidia_icd.json` — `library_path: libGLX_nvidia.so.0` | Works but uses NVIDIA's loader from random path; cluster's actual driver is at `/etc/vulkan/...`, prefer that |
| `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` | **WORKS** ✅ | sapien sees H200 + cudaId=0 + rayTrace=1, 0.04s/step at 1024×768 default shader |

## Verification

```bash
$ nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
580.95.05, NVIDIA H200

$ ls /etc/vulkan/icd.d/
nvidia_icd.json

$ ls /usr/share/vulkan/icd.d/
intel_hasvk_icd.x86_64.json  intel_icd.x86_64.json  lvp_icd.x86_64.json
radeon_icd.x86_64.json       virtio_icd.x86_64.json
# ← no nvidia entry; this is why default sapien scan fails
```

After `export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`:

```python
>>> import sapien.render as sr; print(sr.get_device_summary())
GPU: NVIDIA H200
  Supported: 1
  Present:   0
  cudaId:    0
  rayTrace:  1
  cudaMode   0
```

## Rendering speed (H200, default shader)

| Resolution | Per-step render | 100-step episode |
|---|---|---|
| 320×240 | ~0.03s | 3-4s |
| 640×480 | ~0.04s | 4-5s |
| 1024×768 | ~0.04s | 4-5s |

GPU is bottlenecked by sapien overhead more than pixel count up to 1024×768.

## Investigation history (do not redo this)

1. ❌ `vk::ErrorDeviceLost` — happens when sapien finds CUDA NVIDIA but driver has issue. Fixed by using the **system** ICD path not sapien's bundled one
2. ❌ lavapipe CPU 渲染 — works at 5s/frame but ManiSkill `RenderSystem(Device('cpu'))` rejects
3. ❌ `--render-backend cpu` to ManiSkill replay_trajectory CLI — CLI doesn't expose this kwarg
4. ❌ Patching sapien_env.py to bypass `RenderSystem(Device(...))` — hack, not needed
5. ✅ Set `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` — done

## Hi-res Render Hang Diagnostic Pattern (2026-05-10)

Some old H200 render jobs hung at 100% GPU + low CPU with 0 episode progress
when the deploy script did sapien hi-res (1024x768), per-step `_grab_frame()`,
and DP forward. Treat that as a failure mode to diagnose from the current job's
stderr/artifacts, not as a standing node exclusion list.

**False trail (do not redo this)**: the symptom looked like "oracle works
because it reads `box.pose.raw_pose[0,0].cpu()` per step which acts as a
sync barrier; non-oracle modes hang because they don't sync." We added a
`--render-barrier {none, sapien_state, cuda_sync}` flag and tested on the
exact same node (server24): **all three barriers stuck, all 0 ep**. Barrier
hypothesis was a lurking-variable misattribution — oracle that "worked"
was just lucky to land on a good node. Diagnostic record:

| log (top-level outputs/) | barrier | node | result |
|---|---|---|---|
| `d4_barr_63263_0.err` | none | server23 | ✅ 19s 1ep |
| `d4_barr_63263_1.err` | sapien_state | server24 | ❌ 10min 0ep |
| `d4_barr_63263_2.err` | cuda_sync | server26 | ❌ 10min 0ep |
| `d4_barr_63279_0.err` | none | **server24** | ❌ **12min 0ep** ← controls out barrier |

→ `--render-barrier` flag stays in `scripts/abort_replan_moving_peg.py` as
instrumentation (default `none`, no perf cost). It is **not** the fix.

Root cause was not proven to be node identity alone. When future hi-res hangs,
inspect the actual job logs, output artifacts, Slurm state, and a targeted
small canary before making a job-local node decision.

Canonical hi-res slurm template (current of 2026-05-10):

```bash
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --export=ALL,VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json,DISPLAY=,CUDA_VISIBLE_DEVICES=0
```

Reference slurm script: `slurm/d4_hires_render.sh`.

## Demo solver fixes (additional, beyond rendering)

When generating the actual paper Fig 1 demo (scripted motion planner on
MovingPegInsertion-v1), discovered these per-step bugs (2026-05-09):

1. **Box lateral direction**: peg long axis is `[0.23, 0.97, 0]` in world frame
   (mostly +Y). So PERPENDICULAR (= lateral) box motion = world **X axis**.
   Earlier patches that used Y direction were actually parallel to peg axis
   (= withdrawal motion, not lateral).

2. **One-shot perturbation > continuous drift**: for "online interrupt" narrative,
   want one event invalidating the chunk, not continuous PID-style tracking. Set
   `_perturb_steps=30, _perturb_total_dx=0.03` for 3cm total lateral X displacement
   over 30 env.steps then freeze.

3. **Manual trigger** (skip scout move): `env._box_triggered = ones; _box_trigger_step
   = step_count.clone()` directly. Saves planner state confusion from a
   half-finished move into the env's natural trigger zone.

4. **Hold-pose during pause**: action `[7 arm qpos targets, gripper_cmd=-1]`. The
   8th element of `agent.robot.get_qpos()` is finger qpos (~0.04), NOT a valid
   gripper-action [-1, 1] value — gripper would slowly RELEASE peg over the pause.

5. **Multi-step incremental insert**: motion planner's `move_to_pose_with_screw`
   to a target 15cm away returns "ok" but doesn't actually move. Splitting into
   `Pose([0.02 * (i+1), 0, 0])` for i in 0..8 makes planner step-by-step solve.

## References (web searches that found this)

- [SAPIEN Vulkan ICD discussion #86](https://github.com/haosulab/SAPIEN/discussions/86) — maintainer recommends "ICD file doesn't have to be in default location, set VK_ICD_FILENAMES anywhere"
- [SAPIEN H100 headless issue #250](https://github.com/haosulab/SAPIEN/issues/250) — driver >470 hard requirement (we have 580 ✅)
- [Vulkan ArchWiki](https://wiki.archlinux.org/title/Vulkan) — Vulkan loader checks `/etc/vulkan/icd.d/` AND `/usr/share/vulkan/icd.d/`; cluster has nvidia in the former
- [NVIDIA Container Toolkit issue #1392](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1392) — "ICD mounted at /etc/vulkan/icd.d expected at /usr/share/vulkan/icd.d" — exactly our scenario
