# Phase 04 TODO: DP Plus Cosmos Integration

- [x] Open tmux-held interactive Slurm allocation.
- [x] Create a run directory under `experiments/maniskill/runs/04_integration/`.
- [x] Create a log file under `logs/04_integration/`.
- [x] Update or wrap the existing Phase 03 bridge scripts so active Phase 04
      outputs go under `experiments/maniskill/runs/04_integration/` and logs go
      under `logs/04_integration/`.
- [x] Run the Phase 04 bridge diagnostic inside a tmux-held interactive Slurm
      allocation using Phase 01, Phase 02, and Phase 03 Oracle evidence.
- [ ] Start from reset.
- [ ] Run DP initial policy segment.
- [ ] Detect target/hole motion causally.
- [ ] Run RGB Cosmos-3 after target motion.
- [x] Implement or select a first RGB task-state extractor.
- [ ] Extract from live RGB and Cosmos RGB:
      peg head, hole center, peg axis, hole axis, peg-to-hole vector, lateral
      error, axial error, angular error, near-hole flag, preinsert flag, and
      confidence.
- [ ] Save overlay images/video showing extracted task state on live frames and
      Cosmos future frames.
- [ ] Build a frame-aligned task-state chart with trigger frame and confidence.
- [x] Generate DP-compatible candidate chunks:
      DP continuation, small task-frame residual, bounded insertion-axis push,
      bounded retreat-reapproach, and hold/reobserve.
- [x] Reject any candidate that requires `set_pose`, state restore, future
      simulator labels, gripper-open finisher, or discontinuous peg motion.
- [ ] Score candidates by predicted progress, lateral/angular error,
      executability, trust, and discontinuity penalty.
- [x] Save candidate table with selected and rejected candidates.
- [ ] Execute only a DP-compatible chunk if the trust gate permits execution.
- [ ] Reobserve after execution and compute prediction-observation discrepancy.
- [ ] Save live RGB, Cosmos future video, task-state overlays, candidate table,
      action chart, DP actions, selected action, rejected actions, trigger
      frame, trust score, and final state.
- [ ] Identify the first failing component:
      RGB future, extractor, candidate generator, scorer, trust gate,
      DP-compatible execution, or final contact/insertion.
- [ ] Do not apply Oracle in this phase.

Current entry point:

- `scripts/world_model/phase03_bridge_diagnostic_entry.py` prepares an offline
  bridge package from the valid Phase 01 static DP trace and the Phase 02 RGB
  Cosmos run.
- `scripts/world_model/phase03_rgb_task_state_extractor.py` is the first
  RGB-only extractor diagnostic. It reads Cosmos review frames, writes
  RGB-derived peg/hole detections, confidence values, and overlays, and does
  not read simulator state or execute control.
- It writes candidate chunks, a candidate table, manual review overlays, a
  trust-gate JSON, and a bridge report.
- It does not execute a controller, does not use Oracle, and does not edit
  simulator state.
- Expected first ruling is conservative: `trust_cosmos=false` until a deployed
  RGB task-state extractor produces reliable Cosmos/live overlays.
- These scripts are historical Phase 03 bridge implementations after the plan
  reorder. The active wrapper now writes new bridge-entry outputs to
  `04_integration` and logs to `logs/04_integration`.

Current active Phase 04 evidence:

- `experiments/maniskill/runs/04_integration/bridge_entry/try01/`, Slurm job
  `167531` / `server02`, is the first active bridge-entry diagnostic.
- The command first passed `py_compile` for
  `scripts/world_model/phase03_bridge_diagnostic_entry.py`, then ran
  `scripts/slurm/run_phase03_bridge_entry_in_allocation.sh` with
  `PHASE=04_integration`, `RUN_GROUP=bridge_entry`, and `RUN_NAME=try01`.
- The run wrote `manifest.json`, `candidate_chunks.json`,
  `candidate_table.csv`, `trust_gate.json`, `bridge_entry_report.md`,
  `classification.txt`, and manual review overlay images.
- It loaded Phase 03 action diagnostics from accepted continuous references
  `action_diag/try04` and `action_diag/try05`, plus reverse failures
  `action_diag/try09` and `action_diag/try10`.
- Ruling: `trust_cosmos=false`, `execute_chunk_len=0`,
  `handoff_mode=hold_reobserve_only`, selected candidate `hold_reobserve`,
  `method_evidence_allowed=false`.
- Phase 03 action ruling is `reverse_cosmos_action_not_trusted` with two
  loaded reverse failures and two accepted continuous Cosmos-action references.
- No controller was executed, no Oracle was used, no simulator state was
  edited, and no physical insertion success is claimed.
- This result blocks live insertion candidate execution until a reliable RGB
  task-state extractor and action-interface trust signal exist.

Current evidence from the pre-reorder bridge attempt:

- `experiments/maniskill/runs/03_integration/p03_live_like_frames_20260703_020149_162256_server51/`
  contains extracted live-like review frames from Phase 02 state-audit RGB
  videos.
- `experiments/maniskill/runs/03_integration/p03_rgb_extractor_livelike_v11_20260703_023719_162292_server02/`
  is the current live-like RGB-only extractor diagnostic.
- Live-like v11 result: 66 frames, 7 `rgb_extracted`, 59
  `rgb_extracted_low_confidence`, and 0 `rgb_extraction_failed`.
- `experiments/maniskill/runs/03_integration/p03_rgb_extractor_cosmos_v11_20260703_023740_162292_server02/`
  is the current Cosmos RGB-only extractor diagnostic.
- Cosmos v11 result: 30 frames, 3 `rgb_extracted`, 13
  `rgb_extracted_low_confidence`, and 14 `rgb_extraction_failed`.
- The v11 chart adds RGB-only image-plane peg/hole axis proxies plus
  axis-angle, lateral-error, axial-error, sequence-gate diagnostic columns,
  selected-candidate metadata, and `peg_artifact_risk`.
- The sequence gate downgrades high-confidence frames when per-sample peg/hole
  detections are sparse or peg tracks jump discontinuously.
- v11 also writes `peg_candidate_debug.jsonl` and `peg_candidate_overlays/`
  for every reviewed frame.
- Working-tree code now contains an unverified v12 sequence-level candidate
  track selector. It has not produced run artifacts yet because the requested
  GPU and CPU allocations remained pending. Do not treat v12 as evidence until
  it is run inside a tmux-held Slurm allocation and its outputs are reviewed.
- A later retry also failed to acquire an allocation: short `gpu` stayed
  `Priority` pending and was canceled; `long` and `gaosh` interactive attempts
  did not create usable sessions. Active evidence remains v11.
- v12 code was further updated to redraw final overlays after track selection
  so chart rows, candidate JSONL, and visualization refer to the same selected
  peg candidate. This overlay-consistency change is also unverified until v12
  runs inside an allocation.
- A minimal retry with 1 CPU, 4G memory, and 5 minutes on `gpu` also remained
  `Priority` pending and was canceled. No v12 artifacts exist.
- A smaller retry with 1 CPU, 2G memory, and 3 minutes on `gpu` also remained
  `Priority` pending and was canceled. No v12 artifacts exist.
- Overlay review still shows peg false positives on robot/gripper/Cosmos
  artifacts in representative frames.
- `experiments/maniskill/runs/03_integration/p03_bridge_with_rgbext_cosmos_v11_20260703_023753_162292_server02/`
  is the current historical bridge gate that consumes Cosmos extractor v11.
- Ruling: `trust_cosmos=false`, `execute_chunk_len=0`,
  `handoff_mode=hold_reobserve_only`, selected candidate `hold_reobserve`.
- The trust-gate reasons include Phase 02 visual artifacts and v11 extractor
  failure/low-confidence counts. No physical insertion success is claimed. No
  Oracle evidence is claimed.
- Comparison report:
- `experiments/maniskill/runs/03_integration/p03_bridge_with_rgbext_cosmos_v11_20260703_023753_162292_server02/rgb_extractor_comparison.md`.
- Superseded v4, v5, v6, and v10 Phase 03 extractor and bridge runs were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_integration/`.
- Relaxed table-color extractor attempts v7/v7b/v8 were also archived after
  they either stalled on broad masks or worsened the Cosmos extractor counts.
  Do not revive that threshold-only direction without a faster/safer component
  constraint and visual evidence that peg false positives are reduced.

Next blocker to clear after Phase 03 Oracle:

- The current RGB extractor is not reliable enough to unlock deployed
  insertion candidates. It must be improved or replaced so live/Cosmos overlays
  reliably show peg head, hole center, peg/hole axes, lateral/axial/angular
  error, near-hole/preinsert flags, and confidence before any live insertion
  candidate may run.
