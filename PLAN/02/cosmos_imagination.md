# Phase 02: Cosmos-3 Imagination

Goal: verify RGB Cosmos-3 can imagine future video from live/trajectory RGB
history and produce inspectable controller-facing task-state charts.

Assets:

- RGB Cosmos-3 checkpoint:
  `experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam/`
- Approved 733 data:
  `experiments/maniskill/data/fix3_733/`

Steps:

1. Start a tmux-held interactive Slurm allocation.
2. Record command, allocation, node, checkpoint, input sample, and output path
   under `logs/`.
3. Select representative samples: static, forward motion, reverse motion, and
   continuous motion.
4. Feed RGB history into Cosmos-3.
5. Generate future RGB videos.
6. Generate a task-state chart aligned to frames:
   - observed prefix;
   - predicted future;
   - target/hole motion;
   - peg head / hole center when visible;
   - insertion-axis estimate when extractable;
   - near-hole/preinsert/contact flags when extractable;
   - extractor confidence.
7. Save overlays on Cosmos frames showing the extracted task variables.
8. Visually inspect whether the imagined target/hole state matches the task
   change.

Pass condition:

- Cosmos consumes RGB history, not state-only input;
- output videos are nonblank and correspond to the selected scenario;
- task-state chart and overlays are saved and interpretable;
- every chart clearly labels whether a value came from RGB extraction,
  diagnostic simulator-state audit, or manual visual review.

Failure handling:

- classify as checkpoint loading, RGB preprocessing, prompt/conditioning,
  video generation, or chart extraction issue.
