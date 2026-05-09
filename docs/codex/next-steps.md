# Next Steps

Resume this work on a Linux native filesystem, preferably an ext4 checkout.
Avoid developing from `/mnt/c/...` or a Windows-managed worktree for this
hardware-control path.

## Migration Commands

From Linux:

```bash
mkdir -p ~/research
cd ~/research
git clone https://github.com/thuee-lizhihao/RLinf.git
cd RLinf
git remote add upstream https://github.com/Brunch-Life/RLinf.git
git fetch --all
git checkout -b gello_dagger upstream/feature/gello_dagger
git fetch origin backup/windows-local-notes
git merge origin/backup/windows-local-notes
```

Recommended Git line-ending settings:

```bash
git config core.autocrlf input
git config core.eol lf
```

Hardware permissions:

```bash
sudo usermod -aG dialout,input "$USER"
```

Log out and back in, then check:

```bash
ls /dev/serial/by-id/
ls /dev/input/
```

## Commit 1: Shared Bus And Mapper

Commit message:

```text
feat(gello): add shared Dynamixel bus and joint mapper
```

Goal: build the safe low-level foundation without changing runtime behavior.

Implement:

- `rlinf/envs/realworld/common/gello/gello_joint_mapper.py`
  - `GelloJointMapper.raw_to_joint(raw)`
  - `GelloJointMapper.joint_to_raw(q)`
  - `GelloJointMapper.unwrap(q, reference)`
- `rlinf/envs/realworld/common/gello/gello_dynamixel_bus.py`
  - owns one GELLO serial port
  - exposes locked reads/writes
  - supports fake driver injection for tests
- Update `GelloJointExpert`
  - still supports old `port=...` reader path
  - additionally supports `bus=shared_bus, mapper=mapper`
- Export new classes from `rlinf/envs/realworld/common/gello/__init__.py`.
- Add unit tests for mapper round-trip and fake locked bus delegation.

Acceptance:

- Existing `GelloJointExpert(port=...)` behavior remains compatible.
- Mapper tests pass without hardware.
- No actuator logic yet.

## Commit 2: GelloJointActuator

Commit message:

```text
feat(gello): add shared-bus leader actuator
```

Goal: add GELLO leader control while keeping reader and actuator separated.

Implement:

- `rlinf/envs/realworld/common/gello/gello_joint_actuator.py`
  - uses shared `GelloDynamixelBus`
  - never opens the port itself
  - `enable_torque()`
  - `disable_torque()`
  - `read_joints()`
  - `move_to_joints_factr_pd(target_q, ...)`
  - `emergency_release()`

Alignment strategy:

- `factr_pd`

Safety:

- timeout must release torque
- exceptions must release torque
- current/velocity limits are configurable

Acceptance:

- Mock bus tests cover success, timeout, and release-on-error.
- Reader and actuator share the same fake bus in tests.

## Commit 3: Intervention State Machine

Commit message:

```text
feat(gello): gate teleop with leader alignment
```

Goal: implement policy -> aligning -> aligned -> teleop in
`DualGelloJointIntervention`.

States:

```text
policy
aligning
aligned
teleop
```

Implement:

- `request_align()`
  - snapshot current Franka joints
  - pause direct stream
  - freeze Franka
  - start GELLO actuator alignment
- `confirm_teleop()`
  - only succeeds from `aligned`
  - resumes direct stream
- `cancel_to_policy()`
  - releases actuator
  - pauses direct stream
  - returns to policy
- `_stream_loop()` must only send Franka commands in `teleop`.
- `step()` must freeze Franka in `aligning` and `aligned`.

Info fields:

- `gello_mode`
- `gello_align_error`
- `gello_align_ready`
- `gello_align_strategy`

Acceptance:

- Non-teleop modes never direct-stream to Franka.
- Alignment does not automatically enter teleop.
- `intervene_flag=True` only in `teleop`.

## Commit 4: Keyboard, Config, Docs

Commit message:

```text
feat(gello): add keyboard-controlled alignment flow
```

Goal: connect the state machine to real runs.

Keyboard proposal:

- `t`: request align
- `y`: confirm teleop after alignment
- `p`: cancel/back to policy

Implement:

- Update `keyboard_listener.py` to support `KEY_T`, `KEY_Y`, `KEY_P`.
- Add a small keyboard wrapper that only calls mode methods on
  `DualGelloJointIntervention`.
- Update `apply.py` to create/pass shared bus, mapper, reader, actuator, and
  keyboard wrapper when enabled.
- Update GELLO dual-Franka YAML with static values:

```yaml
gello_default_mode: "policy"
gello_align_on_intervention: true
gello_align_tolerance: 0.06
gello_align_timeout: 5.0
gello_align_dwell_steps: 5
gello_align_current_limit: []
gello_align_kp: []
gello_align_kd: []
```

Acceptance:

- Keyboard can request alignment, confirm teleop, and cancel to policy.
- YAML starts in policy mode by default.
- Alignment frames are not marked as intervention data.
- Teleop frames are marked as intervention data.

## Suggested Verification Order

Run targeted tests after each commit:

```bash
pytest tests/unit_tests/test_gello_joint_mapper.py
pytest tests/unit_tests/test_gello_joint_actuator.py
pytest tests/unit_tests/test_dual_arm_data_collection.py
```

Before each commit:

```bash
ruff check <changed-files>
ruff format <changed-files>
git status --short
git commit -s -m "<message>"
```

Hardware validation order:

1. Test GELLO actuator alone with Franka disconnected.
2. Test Franka freeze with mock/fake GELLO.
3. Test real GELLO alignment to current Franka snapshot.
4. Press confirm and verify no jump when entering teleop.
5. Press cancel/back-to-policy and verify direct stream stops.
