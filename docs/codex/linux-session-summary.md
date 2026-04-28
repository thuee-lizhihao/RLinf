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
   mapper = GelloJointMapper()
   expert = GelloJointExpert(bus=bus, mapper=mapper)
   # Wait until expert.ready, then inspect expert.get_action().
   expert.close()
   bus.close()
   ```

3. Confirm only one process/path owns each GELLO serial port at a time.
4. Slowly move the leader and confirm the seven joint readings are continuous,
   with no obvious `2π` jumps.
5. Confirm `expert.close()` lets the process exit cleanly.

## Next Work After Commit 1

Continue with `docs/codex/next-steps.md`:

1. Commit 2: add `GelloJointActuator` on top of the shared bus.
2. Commit 3: implement the `policy -> aligning -> aligned -> teleop` state
   machine.
3. Commit 4: add keyboard, config, and docs for the alignment flow.
