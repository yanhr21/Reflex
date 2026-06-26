# World-Model Task Rebinding

One sentence: train the manipulation skill in static scenes, then deploy it in
dynamic scenes by streaming object slots, predicting future task frames, moving
the peg through a short physical bridge, and handing control back to the frozen
policy only when the real relative state is continuable.

Canonical high-level note: `IDEA.md`.

Current method state:

- The goal is task completion in the changed world, not restoration of an old
  scene layout.
- The base DP is treated as a static skill prior. It should not be asked to
  absorb absolute dynamic-scene OOD directly.
- The world model predicts object/task-frame futures and uncertainty.
- The bridge controller handles reachable absolute motion in task-frame space.
- `C_pi` decides whether the frozen DP can safely continue from the current
  relative state.
- RGB-D is not optional method polish. State/oracle slots are scaffold,
  debugging evidence, or upper bounds only. A method claim requires RGB-D
  perception to produce object slots or latent state, a world model trained on
  those RGB-D-derived representations, and a controller that consumes those
  RGB-D-derived representations.

Current evidence entry points:

- Active TODO: `TODO/world_model_task_rebinding/00_active.md`
- Dynamic shard: `docs/world_model_task_rebinding/2026-06-02_dynamic_shard_94510.md`
- Online world-model controller hook:
  `docs/world_model_task_rebinding/2026-06-02_online_world_model_controller.md`
- Controller scoring issue:
  `docs/world_model_task_rebinding/2026-06-02_controller_scoring_issue.md`
- Controller scaffold:
  `docs/world_model_task_rebinding/2026-06-02_rebinding_controller_scaffold.md`

Working rule: failures are data. A failure should first be localized to a
physical limitation, data issue, model issue, controller issue, or evaluation
precondition. Do not turn it into an easier task, change the metric, or claim
success without final-state metrics and inspected video/replay evidence.

Current hard boundary: no full RGB-D data means no method claim. No
RGB-D-derived slot/latent means no method claim. No inspected video or replay
artifact means no major dynamic manipulation success claim.
