# Windows Session Summary

This note preserves the Codex planning context from the Windows-based
`gello_dagger` session so the work can resume cleanly from Linux.

## Repository State

- Working branch during analysis: `gello_dagger`.
- Remote comparison target: `upstream/main` from `https://github.com/Brunch-Life/RLinf.git`.
- `gello_dagger` is 28 commits ahead of `upstream/main`.
- High-level branch purpose: dual-Franka real-world GELLO joint teleoperation,
  data collection, rot6d conversion, SFT/eval plumbing, monitoring tools, and
  intervention flag saving.
- The Windows filesystem created friction around `.git` locking, path handling,
  PowerShell profile errors, and mojibake in Chinese/Unicode output. Linux
  native ext4 is preferred for continuing this hardware-control work.

## Existing Local Notes

The session found local documentation edits that should be preserved separately:

- `AGENTS.md` references `DEVELOPMENT_PRINCIPLES.zh-CN.md`.
- `DEVELOPMENT_PRINCIPLES.zh-CN.md` exists as a local project-principles note.

Do not accidentally mix unrelated user edits into implementation commits.

## Core Code Areas Identified

The GELLO intervention path is:

```text
GELLO reader -> intervention wrapper -> Franka env step/direct stream -> collector
```

Important files:

- `rlinf/envs/realworld/common/wrappers/dual_gello_joint_intervention.py`
  - Central mode/state logic.
  - Replaces policy actions with GELLO actions.
  - Owns direct-stream behavior and `intervene_flag` emission.
- `rlinf/envs/realworld/common/gello/gello_joint_expert.py`
  - Current GELLO joint/gripper reader.
  - Should remain read-focused.
- `rlinf/envs/realworld/franka/dual_franka_joint_env.py`
  - Executes joint-space actions unless `teleop_direct_stream=True`.
- `rlinf/envs/realworld/common/wrappers/apply.py`
  - Builds wrapper stack and passes GELLO config.
- `examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml`
  - Main collection config.
- `examples/embodiment/collect_real_data.py` and
  `rlinf/envs/wrappers/collect_episode.py`
  - Save `intervene_action` and `intervene_flag`; not the main control logic.

## Desired Runtime Behavior

The target behavior is not immediate GELLO takeover. It should be:

```text
policy
  -> user requests intervention by keyboard
aligning
  -> snapshot current Franka joints
  -> pause GELLO-to-Franka direct stream
  -> Franka freezes at the snapshot
  -> GELLO leader aligns to the Franka snapshot
aligned
  -> GELLO is aligned and released or low-damping held
  -> Franka remains frozen
  -> wait for keyboard confirmation
teleop
  -> GELLO controls Franka
  -> direct stream can resume
  -> intervention data is marked
```

`intervene_flag=True` should only describe actual teleoperation takeover, not
the alignment wait phase.

## FACTR-Inspired Design Decision

The preferred alignment strategy is FACTR-style leader alignment:

- Preserve `GelloJointExpert` as the reader.
- Add a dedicated actuator/controller layer for GELLO Dynamixel motors.
- Use calibrated mapping between raw Dynamixel positions and Franka-shaped
  joint coordinates:

```python
q_gello = sign * (q_raw - offset)
q_raw_target = sign * q_robot_target + offset
```

Position control is only a fallback strategy. Preferred strategy name:
`factr_pd`; fallback strategy: `position`.

Do not migrate FACTR gravity compensation, null-space torque, bilateral force
feedback, or follower external torque feedback in the first implementation.

## Shared Port Risk And Resolution

Risk: a reader and actuator cannot safely open the same Dynamixel serial port
independently. It may cause port locking failures, packet collisions,
read/write interleaving, or torque/control-mode state desynchronization.

Resolution: keep reader/actuator responsibilities separate but share a single
low-level bus object:

```text
DualGelloJointIntervention
  -> GelloJointExpert
  -> GelloJointActuator

GelloJointExpert
  -> shared GelloDynamixelBus
  -> GelloJointMapper

GelloJointActuator
  -> shared GelloDynamixelBus
  -> GelloJointMapper
```

`GelloDynamixelBus` owns the serial port once and protects all reads/writes
with one lock.

## Project Contribution Rules To Preserve

From `CONTRIBUTING.md` and `AGENTS.md`:

- Use Conventional Commits: `<type>(<scope>): <description>`.
- Every commit must be signed off with `git commit -s`.
- New behavior needs tests and documentation.
- Follow Google-style docstrings and type hints for public APIs.
- Use logging instead of `print` in library code.
- Keep YAML values static; do not compute dynamic values in YAML.
- Do not mix unrelated local edits into feature commits.

## Aborted Windows Implementation Attempt

An implementation attempt for commit 1 was started but interrupted before any
files were created. A follow-up `Test-Path` check showed these files did not
exist at that moment:

- `rlinf/envs/realworld/common/gello/gello_joint_mapper.py`
- `rlinf/envs/realworld/common/gello/gello_dynamixel_bus.py`
- `tests/unit_tests/test_gello_joint_mapper.py`

Continue from the plan in `docs/codex/next-steps.md`.
