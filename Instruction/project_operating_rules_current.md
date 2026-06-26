# Project Operating Rules Current

Date: 2026-06-22

This file records the current project-level judgment and execution rules. It is
separate from method-specific Cosmos notes so future work does not mix cluster
discipline, evidence discipline, and experimental conclusions.

## Active Research Boundary

The project is static-policy-to-dynamic-scene transfer through world-model task
rebinding. The goal is not to restore an old scene layout and not to enumerate
error cases. The goal is to finish the original manipulation task in the
changed world by using live perception, world-model prediction/imagination,
task-frame rebinding, short physical bridging, and conservative handoff to the
base policy only when the real state is continuable.

Do not turn the method into "error detection and recovery." If a proposed fix
depends on listing every possible failure and writing a recovery for each one,
it is outside the current research direction.

## Compute Rules

Login-node work is limited to read-only inspection, downloads, `git clone`,
`git commit`, and `git push`.

Rendering, rollout, replay, data generation, preflight, training, evaluation,
and project-code debug checks must run on compute nodes.

Do not use one-shot `sbatch` jobs for this project. Use tmux-held interactive
Slurm allocations. Request 1-2 days for real experiment work and keep useful
allocations alive for follow-up work.

If a running experiment must be replaced, interrupt the foreground command
inside the tmux allocation and reuse the allocation. Do not release a useful
GPU allocation just because the foreground command changes.

Formal training evidence may use 1, 2, or 4 GPUs, whichever valid allocation
starts first, but it must reserve/run for at least 3 hours. Do not present short
training as method evidence.

Short overfit/sanity training is only a debug gate. It may use 1-2 GPUs, does
not need the 3-hour floor, and should usually be about 50-100 steps.

Keep GPU utilization meaningfully above the cluster release threshold. If a
GPU allocation is idle and useful, either start the aligned compute work or keep
it alive without destroying the user-owned tmux state.

Do not clear or delete the user-owned `reflex` tmux session.

## Evidence Rules

Do not claim method success without metrics plus video/replay evidence from the
current method path.

A major Cosmos/controller result needs:

- the strict `301` RGB/state frames and `300` action steps contract;
- causal live/receding inference, not stale open-loop replay;
- final real simulator state;
- DP baseline comparison;
- structured visual/contact-sheet review.

Oracle state, restored planner success, short cropped diagnostics, and
state-only scaffolds are debugging evidence only. They are not the publishable
RGB/world-model method result.

If a result is bad or surprising, inspect logs, manifests, metrics, and visual
artifacts before explaining it. Classify the failure as data, rendering,
perception, world-model, controller, physics, scheduling, or evaluation
implementation.

Do not weaken the task, loosen gates, change evaluation, or switch to an easier
surrogate because the current result is bad. Only change evaluation for a
documented implementation bug or missing measurement that preserves the same
dynamic task-completion objective.

## Current Work Style

Keep watching active Slurm/tmux experiments and artifacts. Continue aligned
work when resources start or summaries appear.

Latest user instruction: keep root-level operating judgments under
`Instruction/`, observe current state continuously, continue aligned work, and
avoid pause/ending language in routine user updates.

When reporting to the user, use plain causal language:

- what is running;
- what has failed or passed;
- what the blocker is;
- what the next concrete action is;
- what does and does not count as evidence.

Avoid long generic recaps unless explicitly requested. If user direction is
needed later, state the concrete blocker and the options.
