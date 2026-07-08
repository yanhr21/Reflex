# Phase 02 TODO: Cosmos-3 Imagination

- [x] Open tmux-held interactive Slurm allocation.
- [x] Create a run directory under
      `experiments/maniskill/runs/02_cosmos_imagination/`.
- [x] Create a log file under `logs/02_cosmos_imagination/`.
- [x] Select representative samples from `experiments/maniskill/data/fix3_733/`.
- [x] Include forward, reverse, continuous, and static/no-motion cases.
- [x] Run RGB Cosmos-3 from live/trajectory RGB history.
- [x] Save future RGB videos.
- [x] Save task-state charts aligned to frame index:
      target/hole motion, peg head, hole center, insertion axis,
      near-hole/preinsert/contact flags, and confidence when extractable.
- [ ] Save overlays on Cosmos frames showing extracted task variables.
- [x] Label every chart value as RGB-extracted, diagnostic simulator-state
      audit, or manual visual review.
- [x] Save input prefix frames and prediction metadata.
- [x] Visually inspect videos and charts.
- [x] Classify pass/fail before Phase 03 Oracle.

Current evidence:

- `experiments/maniskill/runs/02_cosmos_imagination/p02_cosmos_rgb6_config_20260703_012531_162203_server44/`
  is the current Phase 02 evidence run.
- It loads the active RGB Cosmos-3 checkpoint and writes under the approved
  Phase 02 run/log roots.
- It produced six nonempty Cosmos `vision.mp4` files covering static/no-motion,
  forward shift, reverse shift, continuous insert, peg drop, and peg disturb.
- Review frames were extracted under `cosmos_review_frames/`.
- Visual review is recorded in `reports/visual_review.md`.
- Ruling: RGB Cosmos V2V inference is operational, but generated mid/final
  frames have visible peg/gripper/contact artifacts. This is a pipeline pass
  with controller-facing limitations, not insertion success and not Oracle
  evidence.
- The saved charts are diagnostic simulator-state audit charts, not deployed
  RGB extraction. Phase 03 may run Oracle as an upper-bound diagnostic; Phase
  04 must build and label the bridge/extractor/trust gate before any live
  control claim.

Remaining before Phase 04 can execute actions:

- [ ] Implement or select a first RGB extractor that operates on live/Cosmos
      RGB frames, or explicitly classify why extraction fails.
- [ ] Save overlays for extracted task variables on Cosmos frames, not only
      diagnostic state-audit overlays.
