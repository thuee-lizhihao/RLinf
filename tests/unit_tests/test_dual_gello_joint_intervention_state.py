# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.gello import GelloJointActuatorResult
from rlinf.envs.realworld.common.wrappers.dual_gello_joint_intervention import (
    DualGelloJointIntervention,
)


class _FakeDualJointEnv(gym.Env):
    """Minimal dual-arm env exposing the methods used by the wrapper."""

    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(16,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({})
        self.joints = np.array(
            [
                [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
                [-0.1, -0.2, -0.3, 0.4, -0.5, 0.6, -0.7],
            ],
            dtype=np.float64,
        )
        self.last_action: np.ndarray | None = None

    def get_joint_positions(self) -> np.ndarray:
        return self.joints.copy()

    def reset(self, *, seed=None, options=None):
        return {}, {}

    def step(self, action):
        self.last_action = np.asarray(action)
        return {}, 0.0, False, False, {}


class _FakeExpert:
    """Ready GELLO reader with mutable joint output."""

    def __init__(self, q: np.ndarray) -> None:
        self.q = q.astype(np.float64)
        self.ready = True
        self.closed = False

    def get_action(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), np.array([0.0], dtype=np.float64)

    def close(self) -> None:
        self.closed = True


class _FakeActuator:
    """Fake actuator that can block until the test releases it."""

    def __init__(self, release_event: threading.Event | None = None) -> None:
        self.release_event = release_event
        self.targets: list[np.ndarray] = []
        self.released = False

    def move_to_joints_factr_pd(self, target_q, **kwargs) -> GelloJointActuatorResult:
        self.targets.append(np.asarray(target_q, dtype=np.float64).copy())
        if self.release_event is not None:
            self.release_event.wait(timeout=2.0)
        return GelloJointActuatorResult(
            success=True,
            strategy="factr_pd",
            error=np.zeros(7, dtype=np.float64),
            elapsed=0.0,
        )

    def emergency_release(self) -> None:
        self.released = True


def _make_wrapper(
    *,
    release_event: threading.Event | None = None,
    direct_stream: bool = False,
) -> tuple[DualGelloJointIntervention, _FakeDualJointEnv, _FakeActuator, _FakeActuator]:
    env = _FakeDualJointEnv()
    left_expert = _FakeExpert(env.joints[0])
    right_expert = _FakeExpert(env.joints[1])
    left_actuator = _FakeActuator(release_event)
    right_actuator = _FakeActuator(release_event)
    wrapper = DualGelloJointIntervention(
        env,
        left_port="unused-left",
        right_port="unused-right",
        direct_stream=direct_stream,
        default_mode="policy",
        left_expert=left_expert,
        right_expert=right_expert,
        left_actuator=left_actuator,
        right_actuator=right_actuator,
    )
    return wrapper, env, left_actuator, right_actuator


def test_aligning_and_aligned_freeze_without_intervention_flag() -> None:
    """Alignment modes freeze the robot and never mark intervention data."""
    release = threading.Event()
    wrapper, env, _left_actuator, _right_actuator = _make_wrapper(
        release_event=release
    )

    assert wrapper.request_align()
    _obs, _rew, _done, _truncated, info = wrapper.step(np.ones(16))
    np.testing.assert_allclose(env.last_action, wrapper._freeze_action())
    assert info["gello_mode"] == "aligning"
    assert not info["intervene_flag"][0]

    release.set()
    for _ in range(100):
        if wrapper.mode == "aligned":
            break
        time.sleep(0.01)

    _obs, _rew, _done, _truncated, info = wrapper.step(np.ones(16))
    assert wrapper.mode == "aligned"
    assert info["gello_align_ready"][0]
    assert info["gello_mode"] == "aligned"
    assert not info["intervene_flag"][0]


def test_confirm_teleop_only_from_aligned() -> None:
    """Teleop confirmation is gated on successful alignment."""
    wrapper, env, left_actuator, right_actuator = _make_wrapper()

    assert not wrapper.confirm_teleop()
    assert wrapper.request_align()
    for _ in range(100):
        if wrapper.mode == "aligned":
            break
        time.sleep(0.01)

    np.testing.assert_allclose(left_actuator.targets[0], env.joints[0])
    np.testing.assert_allclose(right_actuator.targets[0], env.joints[1])
    assert wrapper.confirm_teleop()
    assert wrapper.mode == "teleop"


def test_cancel_to_policy_releases_actuators_and_pauses_stream() -> None:
    """Cancel returns to policy and releases both GELLO actuators."""
    release = threading.Event()
    wrapper, _env, left_actuator, right_actuator = _make_wrapper(
        release_event=release
    )

    assert wrapper.request_align()
    wrapper.cancel_to_policy()

    assert wrapper.mode == "policy"
    assert left_actuator.released
    assert right_actuator.released
    assert not wrapper._stream_paused.is_set()
