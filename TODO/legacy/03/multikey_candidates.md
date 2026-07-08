# Phase 03 Multi-Key Candidate Coverage

This file lists approved `fix3_733` candidates for later Phase 03 coverage.
It is planning context only. None of these keys count as covered until the
full-pipeline script can reset from the key, run DP static prefix, run Cosmos
RGB/action dynamic control, run the physical finisher, write artifacts, and pass
visual review.

Initial candidate keys:

- `hole_late_fast_shift_seed10300001_idx5000`
  - H5:
    `experiments/maniskill/data/fix3_733/canonical_h5/hole_late_fast_shift_seed10300001_idx5000.fix3/hole_late_fast_shift_seed10300001_idx5000.h5`
  - Purpose: target fast-shift dynamic case.
- `hole_late_reverse_seed1040017_idx1250`
  - H5:
    `experiments/maniskill/data/fix3_733/canonical_h5/hole_late_reverse_seed1040017_idx1250.fix3/hole_late_reverse_seed1040017_idx1250.h5`
  - Purpose: reverse target motion case.
- `hole_late_move_stop_seed1080064_idx0000`
  - H5:
    `experiments/maniskill/data/fix3_733/canonical_h5/hole_late_move_stop_seed1080064_idx0000.fix3/hole_late_move_stop_seed1080064_idx0000.h5`
  - Purpose: target starts moving and stops.
- `peg_disturb_seed1051032_idx0008`
  - H5:
    `experiments/maniskill/data/fix3_733/canonical_h5/peg_disturb_seed1051032_idx0008.fix3/peg_disturb_seed1051032_idx0008.h5`
  - Purpose: peg / wooden-stick disturbance.
- `peg_drop_seed36705002_pseed39705002_idx12420`
  - H5:
    `experiments/maniskill/data/fix3_733/canonical_h5/peg_drop_seed36705002_pseed39705002_idx12420.fix3/peg_drop_seed36705002_pseed39705002_idx12420.h5`
  - Purpose: peg drop / recovery stress case.
- `hole_late_continuous_insert_seed10241044_idx5004`
  - H5:
    `experiments/maniskill/data/fix3_733/canonical_h5/hole_late_continuous_insert_seed10241044_idx5004.fix3/hole_late_continuous_insert_seed10241044_idx5004.h5`
  - Purpose: continuous insertion-like dynamic reference.
- `hole_late_constant_seed10250253_idx5009`
  - H5:
    `experiments/maniskill/data/fix3_733/canonical_h5/hole_late_constant_seed10250253_idx5009.fix3/hole_late_constant_seed10250253_idx5009.h5`
  - Purpose: target static / constant baseline contrast.

Current implementation status:

- The full-pipeline runner now accepts explicit `SOURCE_H5_PATH` /
  `SOURCE_KEY`, records the selected key and H5 path in run artifacts, and can
  enforce `REQUIRE_SOURCE_H5_PROTOCOL=true`.
- New source-H5 launches should use short grouped paths through
  `scripts/slurm/phase03_h5_source.sh`, for example
  `h5_move_stop/try03`.
- Do not use the H5 future trajectory as controller-facing future labels.
- Do not count a key as covered unless its annotated video passes the Phase 03
  visual review checklist, including the active-insertion rule: target / hole
  motion must not create success by moving onto a mostly stationary peg or
  wooden stick.
- Current strict validation-key single-case success count is `2`:
  `h5_continuous_insert/try04` and `h5_continuous_insert/try11`. Overall
  Oracle completion still requires forward/backward target motion, left/right
  target motion, peg/wooden-stick disturbance, and multiple approved keys.
- `h5_continuous_insert/try06` on
  `hole_late_continuous_insert_seed10241574_idx5018` reached simulator metric
  true but is rejected after visual review as target-assisted / insufficient
  active-insertion evidence. It is not a second key success.
- Latest fastshift retry `h5_fastshift/try04` is archived negative evidence,
  not coverage: it passed artifact audit without snap, but ended with
  `simulator_success_metric=false` and final `peg_head_l2` about `0.1219`.
- Current completion gate says the next missing coverage group is
  `forward_backward_target_motion`. The prepared real-coverage launcher is
  `scripts/slurm/phase03_forward_backward_probe.sh`, defaulting to
  `hole_late_reverse_seed1040038_idx0004` as `h5_reverse/try21`. It explicitly
  refuses row-offset and future-label diagnostics.
