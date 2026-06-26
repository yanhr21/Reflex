# OpenPI pi0.5 Contact-Action Pivot

Date: 2026-06-24

## One Sentence

Replace the weak DP/scorer/contact-diffusion executor line with an official
OpenPI/pi0.5 action policy adapted on the accepted 733 dynamic/static
PegInsertionSide trajectories, then evaluate whether it can generate the
contact/insertion actions that the current candidates cannot.

## Why This Pivot Is Necessary

The current failures are not explained by "insertion is impossible." Saved
snapshot replay showed states where a short chunk plus DP96 can finish, and
Policy-DROID chunks can preserve grasp and leave DP-continuable states. The
failure is that the learned or selected short chunk itself does not push the
peg through the insertion axis into a contact-stable/insertion state.

Recent evidence:

- selected causal-suffix replay: `0/16` direct post-chunk success and `0/16`
  direct gate-ok, while DP96 still finished `8/16`;
- causal-suffix candidate pool: useful handoff headroom existed, but
  `0/192` direct insertion/gate-ok;
- same-prefix Policy-DROID diagnostic: finite 8-step action, grasp preserved,
  DP96 succeeds afterward in `63` steps, but direct insertion/contact-stable
  remains false;
- direct-contact risk-head training reached the one-GPU-hour floor, but this
  is now historical because the method has pivoted away from custom in-repo
  action diffusion.

Therefore, adding another scalar scorer over the same candidate family is not
the right next move. The main model must become a stronger action generator.

## Official OpenPI Basis

The local OpenPI checkout at `/public/home/yanhongru/ICLR2027/openpi` contains
official pi0.5 support:

- README lists `pi05_base` and `pi05_droid` checkpoints under
  `gs://openpi-assets/checkpoints/`;
- `src/openpi/training/config.py` defines `pi05_droid` and
  `pi05_droid_finetune`;
- the local project config now adds `pi05_maniskill_peg733`, using
  `Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False)`,
  `LeRobotLiberoDataConfig`, and
  `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`;
- OpenPI examples convert custom robot data to LeRobot format, which is the
  supported finetuning input path.

This satisfies the user requirement that weights and model structure remain
OpenPI-native rather than a hand-written intermediate VAE/MLP/diffusion model.

## Data Source

Initial data must come from the accepted 733 source:

`experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733`

The strict H5 audit reports `733/733` valid records and all nine scenario
classes present. The derived full-episode WAM condition root has:

- `733` source episodes;
- `9271` prefix rows;
- RGB-only visual input;
- `301` RGB/state frames and `300` action steps;
- source WAM sidecar rows with 32 columns, of which only the first 7 are
  robot `pd_ee_delta_pose` actions for OpenPI policy training;
- train/val JSONL plus per-row `.npy` action arrays and causal metadata.

For OpenPI, this must be converted into an official LeRobot-style dataset with
RGB image fields, proprio/state fields, 7D robot action targets, and a task
prompt. OpenPI pads state/action tensors to the model action dimension inside
its official transforms.
Future target poses may be labels/metadata, but controller inputs must remain
causal prefix/current observations.

## Hypothesis

Pi0.5-base has a stronger pretrained VLA/action prior than the frozen static DP
and the local contact-suffix diffusion models. With fresh normalization on the
733 LeRobot conversion, finetuning should produce short chunks that more
directly maintain grasp, align, and enter insertion contact.

The first success criterion is not a pretty training loss. It is replay from
saved dynamic live snapshots with real simulator state:

- direct post-chunk inserted/contact-stable/handoff-gate rate improves above
  the old `0/16` and `0/192` direct baselines;
- grasp is visually preserved;
- DP is not required as the main executor, but may be used as a historical
  suffix label/baseline when comparing continuability;
- videos/contact sheets must be inspected before claiming dynamic manipulation
  success.

## Non-Goals

- Do not train another custom MLP/VAE/diffusion executor as the main method.
- Do not treat OpenPI zero-shot or a few-second run as progress.
- Do not use frozen DP as the required final base model.
- Do not relax the causal observation boundary by feeding future ground-truth
  hole/peg poses into policy input.

## Evidence Update 2026-06-25

The OpenPI/pi0.5 pivot has produced a valid trained action model, but not direct
insertion yet.

Positive evidence:

- official OpenPI/pi0.5 training on the accepted 733-derived LeRobot data met
  the real one-GPU-hour floor;
- all active checkpoints use OpenPI-native weights/configs, with no custom
  VAE/MLP/diffusion intermediate model;
- step `3600` replay preserves grasp on all four checked dynamic snapshots;
- DP96 continuability improves from `1/4` at the first `1599` checkpoint to
  `2/4` at the `3600` checkpoint.

Negative evidence:

- OpenPI direct post-chunk success remains `0/4`;
- OpenPI direct inserted/contact-stable remains `0/4`;
- executing the full native 16-action horizon does not improve direct
  insertion over the 8-step diagnostic;
- RGB replay video is still blocked by the live-render/Vulkan path, so current
  visual evidence is limited to contact-state sheets generated from saved
  replay step records.

Interpretation:

The original diagnosis still holds: insertion is not physically impossible,
but the current learned action chunk is still mostly a grasp-preserving
handoff-state generator rather than a direct contact/insertion executor. The
next useful improvement should increase direct insertion/contact-positive
supervision for the OpenPI-native policy, not add another scalar scorer over
weak candidates.

## Contact-Suffix Update 2026-06-25/26

A Slurm-side audit of the accepted 733 H5 trajectories found that the data do
contain the missing behavior:

- `733/733` episodes are final-success-like;
- `733/733` episodes contain inserted frames;
- `733/733` episodes have eligible 16-step insertion suffix windows;
- `71045` total eligible suffix windows exist around first insertion.

This changes the diagnosis. Cosmos/OpenPI are not blocked because the accepted
dataset has no insertable actions. The stronger explanation is that the
full-episode OpenPI finetune spreads training mass over grasp/approach/handoff
and does not sufficiently specialize the short contact/insertion segment that
the saved dynamic snapshots require.

Active repair status: `yanhongru/maniskill_peg733_openpi_contact_suffix16` has
now been built as an official LeRobot dataset and audited. It contains `5853`
16-step suffix episodes / `93648` frames from all `733` source episodes, with
inserted labels used only as offline window-selection metadata, not runtime
oracle inputs.

The corresponding official OpenPI/pi0.5 config
`pi05_maniskill_peg733_contact_suffix16` has a formal one-GPU-hour training
result. The successful run trained `1700` steps for `4303` seconds in held
Slurm allocation `150773`, step `44`, using official `pi05_base` loading and
final-only checkpoint save. The preserved checkpoint is:

`experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16/pi05_peg733_contact_suffix16_noema_direct1700_1gpu1h_20260626_alloc150773/1699`

This is still not an insertion success claim. It is the first valid trained
contact-suffix OpenPI action model.

The first saved-snapshot replay from checkpoint `1699` was negative for direct
insertion: exec16 panel replay produced direct success `0/4`, inserted `0/4`,
contact-stable `0/4`, and grasp preserved `4/4`. It did produce a limited
handoff signal, DP96 continuable `2/4` and DP96 success `1/4`, but this is not
OpenPI direct task completion.
So the pivot has improved engineering validity and preserved grasp, but has
not yet solved the actual insertion action. The next idea revision should
target object-frame/action alignment, receding execution from refreshed
observations, or stronger contact-target supervision, not a scorer-only wrapper
over the same weak chunks.

The contact-state sheets make the likely failure concrete. The OpenPI chunks
keep the peg grasped for all executed steps, yet no replay label contains an
inserted step during the chunk. In f106, rel-x moves from `-0.148` to `-0.198`
and abs(y)+abs(z) grows from `0.054` to `0.122`; in f132, abs(y)+abs(z) grows
from `0.013` to `0.089`. The model is not simply too weak to move; it is often
moving in a direction that does not close the hole-frame contact geometry.

The first diagnosis narrows the failure. Baseline replay made `abs_yz_sum`
worse in `4/4` samples and `abs_x` worse in `3/4`; a privileged prompt that
injected source first-insert timing stayed at `0/4` direct insertion. The
model is not simply missing the training prompt's suffix-offset text. It is
more likely reproducing source-style insertion motion without enough causal
object/task-frame state to bind that motion to the moved hole.

Updated idea: keep OpenPI/pi0.5 as the action model, but add an OpenPI-native
object/task-frame conditioning path. The policy input should include causal
peg/hole/TCP relative geometry from allowed observations or metadata, with
fresh OpenPI norm stats and the same official checkpoint loading. This is not
a license to add a custom VAE/MLP/diffusion executor; it is a data/config
conditioning repair inside the OpenPI model family.

## Object-State Conditioning Update 2026-06-26

The first version of that idea has now been partially implemented as an
upper-bound diagnostic, not as final method evidence.

Implemented:

- object17 LeRobot rewrite
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17`;
- 17D current state:
  `tcp_pose3, peg_pose3, hole_pose3, peg_head_at_hole3,
  hole_velocity_step3, grasped, inserted`;
- same `5853` contact-suffix episodes / `93648` rows as the qpos8 suffix
  branch;
- official OpenPI config
  `pi05_maniskill_peg733_contact_suffix16_object17`, using the same pi0.5
  model family and official `pi05_base` loader;
- OpenPI-format norm stats computed inside Slurm with OpenPI
  `normalize.RunningStats/save` after the official norm-stats script stalled.

Positive signal:

The project now has a concrete path to test whether object/task-frame
conditioning fixes the insertion failure. The previous qpos8 suffix model
already proved that the official pi0.5 policy can train on the 733-derived
data and preserve grasp, but it moves in the wrong hole-frame direction. The
object17 branch directly targets that failure mode without adding a scorer or
a homemade action model.

Current status:

The first object17 tree was invalidated because a hardlink-based rewrite
mutated the canonical qpos8 contact-suffix repo. That has now been repaired by
fresh noncanonical qpos8/object17 repo ids and a video-backed object17 LeRobot
repo:

`yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626`

The H.264 video-backed repo passed strict audit with `5853` suffix episodes,
`93648` rows, `11706` mp4 files, state dim `17`, action dim `7`, and
OpenPI-format norm stats. Direct H.264/pyav decode works; the remaining data
loading issue was Hugging Face `datasets.load_dataset` using shared cache
storage. Redirecting `HF_HOME`, `HF_DATASETS_CACHE`, and `XDG_CACHE_HOME` to
compute-node `/tmp` fixed the constructor/item and first-batch gates.

Formal object17-video OpenPI/pi0.5 training completed in allocation `153455`
for `4748` seconds, satisfying the one-GPU-hour floor. It used official
OpenPI/pi0.5 config and official `pi05_base` weights, with no custom VAE, MLP,
diffusion executor, or scorer-only selector. Preserved checkpoint:

`experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455/1699`

Saved-snapshot replay from that checkpoint produced the current sharpest
diagnosis:

- direct success `0/4`;
- direct inserted `0/4`;
- direct contact-stable `0/4`;
- grasp preserved `4/4`;
- DP96 historical continuability/success `3/4`.

So object/task-frame conditioning is a real positive signal for handoff
quality, but it still does not solve direct insertion. The model can preserve
grasp and often move into a state the old DP can finish from, but the OpenPI
chunk itself still fails to execute the final contact/insertion action.

A first receding diagnostic tested whether this was merely open-loop chunk
drift. The new split-env receding wrapper calls official OpenPI inference
repeatedly while refreshing live simulator object17 state between short
executions. On one hard snapshot (`sample_00_hole_late_move_stop`, prefix
`106`), `3` queries with `4` executed actions each stayed negative: direct
success `0/1`, inserted `0/1`, contact-stable `0/1`, grasp preserved `1/1`,
and `abs(y)+abs(z)` worsened from `0.0539` to `0.1341`. This is only a small
upper-bound diagnostic because it uses privileged object17 state and a static
observed-prefix image, but it suggests the issue is not only single-chunk
open-loop drift.

Important limitation:

The object17 state uses simulator/source task slots. Its current value is an
upper-bound diagnosis: object binding helps but is insufficient as a single
open-loop chunk. The publishable method still needs those peg/hole/TCP/task
slots to come from RGB-derived perception or another allowed causal
observation pipeline, then feed the same OpenPI-native conditioning interface.

Next idea revision: keep OpenPI/pi0.5 as the action model, but put more weight
on stronger near-contact insertion supervision and contact-mode action
distribution repair. A broader receding panel is still useful only if it
changes the query/execution contract meaningfully; the first naive q3x4
receding smoke did not fix direct insertion. Do not fall back to DP as the main
method, scorer-only selection, or homemade intermediate action models.

The concrete next hypothesis is now an OpenPI-native near-contact object17/video
branch, not a new base model yet. The current object17-video dataset still
contains long approach windows (`64,48,32,24` steps before insertion), so the
learned pi0.5 action distribution may spend too much probability mass on
grasp-preserving approach/handoff and too little on the final insertion-axis
contact mode. The next config
`pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626`
uses the same official `Pi0Config(pi05=True, action_horizon=16,
discrete_state_input=False)` and official `pi05_base` loader, but points to a
new repo id:

`yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_nearcontact_20260626`

The intended conversion uses the accepted 733 source with object17 causal
task-state slots, video-backed LeRobot camera keys, and offsets
`16,12,8,4,2,1`. Offset `1` is the closest pre-insertion causal action target;
offset `0` is avoided as the main data branch because it observes an already
inserted frame. This branch still needs Slurm-side conversion, audit, fresh
norm stats, one-GPU-hour OpenPI training, and replay before it is evidence.

Execution update: this near-contact branch has now been run. It produced a
clean audited LeRobot repo with `4375` suffix episodes and `70000` rows, trained
official OpenPI/pi0.5 from official `pi05_base` for `5044` seconds, and
preserved checkpoint step `1699`. Saved-snapshot replay stayed negative:
direct success `0/4`, inserted `0/4`, contact-stable `0/4`, grasp preserved
`4/4`, and DP96 handoff `1/4`.

This falsifies the narrow explanation that the previous object17-video failure
was mainly caused by far approach windows diluting insertion suffix actions.
Near-contact supervision made the action prior keep grasp but did not produce
the missing insertion-axis contact behavior, and it reduced handoff quality
relative to the previous object17-video checkpoint. The next idea should focus
on action-coordinate/contact-mode mismatch, better contact-conditioned OpenPI
supervision, or a stronger official OpenPI/VLA-family action model only if it
keeps official weights/code compatibility. Do not repeat scorer-only selection
or offset-only data rewrites as the main method.
