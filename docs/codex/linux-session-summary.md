# Linux Session Summary

This note preserves the Linux continuation context for the GELLO DAgger work.
It is intended to let another Linux machine resume the same branch and verify
the hardware-facing pieces without relying on local Codex chat history.

## Session Scope

- Branch: `gello_dagger`.
- Working directory: `/home/zhihao/research/RLinf`.
- Primary continuation notes:
  - `docs/codex/next-steps.md`
  - `docs/codex/windows-session-summary.md`
- Current policy from the project owner:
  - Keep assistant replies in Chinese.
  - Keep generated code, comments, docstrings, tests, and commit messages in
    English.
  - Do not install heavy realworld dependencies on this laptop.
  - Run only light local checks here; realworld and hardware checks will run on
    a shared lab/company machine.

## Commit 1 Completed

Commit message:

```text
feat(gello): add shared Dynamixel bus and joint mapper
```

Implemented files:

- `rlinf/envs/realworld/common/gello/gello_joint_mapper.py`
  - `GelloJointMapper.raw_to_joint(raw)`
  - `GelloJointMapper.joint_to_raw(q)`
  - `GelloJointMapper.unwrap(q, reference)`
- `rlinf/envs/realworld/common/gello/gello_dynamixel_bus.py`
  - Owns one GELLO Dynamixel driver.
  - Serializes reads, writes, arbitrary driver calls, and close through one
    lock.
  - Supports fake driver injection for unit tests.
- `rlinf/envs/realworld/common/gello/gello_joint_expert.py`
  - Keeps the legacy `GelloJointExpert(port=...)` path.
  - Adds `GelloJointExpert(bus=..., mapper=...)` for shared-bus reads.
  - Uses the mapper for unwrap continuity.
  - Adds `close()` to stop the reader thread.
- `rlinf/envs/realworld/common/gello/__init__.py`
  - Exports `GelloDynamixelBus` and `GelloJointMapper`.
- `rlinf/envs/realworld/common/wrappers/dual_gello_joint_intervention.py`
  - Calls `close()` on both GELLO readers during wrapper shutdown.
- `tests/unit_tests/test_gello_joint_mapper.py`
  - Covers mapper round-trip, unwrap, and validation.
- `tests/unit_tests/test_gello_dynamixel_bus.py`
  - Covers fake-driver locked reads, write/call/close delegation, and
    `GelloJointExpert(bus=...)` consumption.

Out of scope for commit 1:

- GELLO actuator implementation.
- Intervention state machine changes.
- Keyboard-controlled alignment flow.
- Real robot motion.

## Local Verification Already Run

This laptop does not have the project runtime dependencies installed, and the
owner does not want heavy realworld dependency setup here.

Light local check run successfully:

```bash
python -m py_compile \
  rlinf/envs/realworld/common/gello/gello_joint_mapper.py \
  rlinf/envs/realworld/common/gello/gello_dynamixel_bus.py \
  rlinf/envs/realworld/common/gello/gello_joint_expert.py \
  tests/unit_tests/test_gello_joint_mapper.py \
  tests/unit_tests/test_gello_dynamixel_bus.py
```

## Company Machine Verification For Commit 1

Run these software checks on a machine with the normal RLinf environment:

```bash
pytest tests/unit_tests/test_gello_joint_mapper.py
pytest tests/unit_tests/test_gello_dynamixel_bus.py
ruff check \
  rlinf/envs/realworld/common/gello/gello_joint_mapper.py \
  rlinf/envs/realworld/common/gello/gello_dynamixel_bus.py \
  rlinf/envs/realworld/common/gello/gello_joint_expert.py \
  rlinf/envs/realworld/common/gello/__init__.py \
  rlinf/envs/realworld/common/wrappers/dual_gello_joint_intervention.py \
  tests/unit_tests/test_gello_joint_mapper.py \
  tests/unit_tests/test_gello_dynamixel_bus.py
ruff format --check \
  rlinf/envs/realworld/common/gello/gello_joint_mapper.py \
  rlinf/envs/realworld/common/gello/gello_dynamixel_bus.py \
  rlinf/envs/realworld/common/gello/gello_joint_expert.py \
  rlinf/envs/realworld/common/gello/__init__.py \
  rlinf/envs/realworld/common/wrappers/dual_gello_joint_intervention.py \
  tests/unit_tests/test_gello_joint_mapper.py \
  tests/unit_tests/test_gello_dynamixel_bus.py
```

Run these hardware smoke checks without actuator motion:

1. Legacy reader compatibility:

   ```python
   from rlinf.envs.realworld.common.gello import GelloJointExpert

   expert = GelloJointExpert(port="/dev/serial/by-id/<gello-port>")
   # Wait until expert.ready, then inspect expert.get_action().
   expert.close()
   ```

2. Shared-bus reader path:

   ```python
   from rlinf.envs.realworld.common.gello import (
       GelloDynamixelBus,
       GelloJointExpert,
       GelloJointMapper,
   )

   bus = GelloDynamixelBus(port="/dev/serial/by-id/<gello-port>")
   # Replace signs/offsets with calibration output from gello_calibrate.py.
   mapper = GelloJointMapper(signs=[...], offsets=[...])
   expert = GelloJointExpert(bus=bus, mapper=mapper)
   # Wait until expert.ready, then inspect expert.get_action().
   expert.close()
   bus.close()
   ```

3. Confirm only one process/path owns each GELLO serial port at a time.
4. Slowly move the leader and confirm the seven joint readings are continuous,
   with no obvious `2π` jumps.
5. Confirm `expert.close()` lets the process exit cleanly.

Important calibration note:

- `GelloJointMapper()` defaults to identity signs and zero offsets. This is only
  suitable for mock tests.
- Real shared-bus reader and actuator checks must use calibrated
  `joint_signs` and `joint_offsets` from
  `toolkits/realworld_check/gello_calibrate.py`.
- The dual-Franka GELLO YAML now exposes explicit placeholders:
  - `left_gello_joint_signs`
  - `left_gello_joint_offsets`
  - `right_gello_joint_signs`
  - `right_gello_joint_offsets`

## Next Work After Commit 1

Continue with `docs/codex/next-steps.md`:

1. Commit 2: add `GelloJointActuator` on top of the shared bus.
2. Commit 3: implement the `policy -> aligning -> aligned -> teleop` state
   machine.
3. Commit 4: add keyboard, config, and docs for the alignment flow.

## 2026-04-29 Draft Progress

The owner approved writing high-quality first drafts locally overnight, then
running heavy dependency and real hardware validation on the shared lab machine.

Local commits completed after commit 1:

- `feat(gello): add shared-bus leader actuator`
  - Adds `GelloJointActuator`.
  - Adds current-mode `factr_pd` alignment.
  - Adds release-on-timeout and release-on-error behavior.
  - Adds mock actuator unit tests.
- `feat(gello): gate teleop with leader alignment`
  - Adds `policy -> aligning -> aligned -> teleop` state API.
  - Adds `request_align()`, `confirm_teleop()`, and `cancel_to_policy()`.
  - Ensures `aligning` and `aligned` freeze Franka actions.
  - Ensures direct streaming only sends in `teleop`.
  - Adds mock state-machine unit tests.
- `feat(gello): add keyboard-controlled alignment flow`
  - Adds keyboard mapping:
    - `t`: request alignment.
    - `y`: confirm teleop after aligned.
    - `p`: cancel/back to policy.
  - Updates the dual-Franka GELLO collection YAML to start in `policy`.
  - Adds explicit YAML placeholders for left/right GELLO joint signs and
    offsets.
  - Wires shared bus, mapper, reader, actuator, and keyboard wrapper when
    `gello_align_on_intervention: true`.

Local lightweight validation:

```bash
python -m py_compile <changed-python-files>
```

Heavy validation still belongs on the shared lab machine:

1. Run the unit tests added for commits 1-4.
2. Run `ruff check` and `ruff format --check` on changed files.
3. Repeat commit 1 GELLO reader smoke tests.
4. Fill `left/right_gello_joint_signs` and `left/right_gello_joint_offsets`
   from calibration before any shared-bus real hardware check.
5. Test commit 2 actuator alone with Franka disconnected.
6. Test commit 3/4 with fake or disabled actuator before allowing real motion.
7. Test the final hardware sequence:
   `policy -> t -> aligning -> aligned -> y -> teleop -> p -> policy`.
