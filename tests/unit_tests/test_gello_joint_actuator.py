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

import numpy as np
import pytest

from rlinf.envs.realworld.common.gello import GelloDynamixelBus, GelloJointActuator


class _FakeActuatorDriver:
    """Fake Dynamixel driver for actuator safety tests."""

    def __init__(self, num_servos: int = 7) -> None:
        self.positions = np.zeros(num_servos, dtype=np.float64)
        self.velocities = np.zeros(num_servos, dtype=np.float64)
        self.torque_enabled = False
        self.mode: int | None = None
        self.currents: list[np.ndarray] = []
        self.fail_on_current = False

    def get_joints(self) -> np.ndarray:
        return self.positions.copy()

    def get_positions_and_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.copy(), self.velocities.copy()

    def set_current(self, currents: np.ndarray) -> None:
        if self.fail_on_current:
            raise RuntimeError("current write failed")
        self.currents.append(currents.copy())
        self.positions[:7] = self.positions[:7] + 0.5 * currents[:7] / 100.0

    def set_torque_mode(self, enabled: bool) -> None:
        self.torque_enabled = enabled

    def set_operating_mode(self, mode: int) -> None:
        self.mode = mode

    def verify_operating_mode(self, expected_mode: int) -> None:
        assert self.mode == expected_mode


def test_factr_pd_success_releases_torque() -> None:
    """Current-mode PD reaches the target and releases torque on success."""
    driver = _FakeActuatorDriver()
    actuator = GelloJointActuator(
        GelloDynamixelBus(driver=driver),
        kp=np.full(7, 200.0),
        kd=np.zeros(7),
        current_limit=np.full(7, 100.0),
    )
    target = np.full(7, 0.2)

    result = actuator.move_to_joints_factr_pd(
        target,
        tolerance=0.03,
        timeout=1.0,
        dwell_steps=1,
        period=0.0,
    )

    assert result.success
    assert result.strategy == "factr_pd"
    assert not driver.torque_enabled
    assert driver.mode == 0
    assert driver.currents


def test_factr_pd_timeout_releases_torque() -> None:
    """A timeout returns a failed result and releases torque."""
    driver = _FakeActuatorDriver()
    actuator = GelloJointActuator(
        GelloDynamixelBus(driver=driver),
        kp=np.zeros(7),
        kd=np.zeros(7),
    )

    result = actuator.move_to_joints_factr_pd(
        np.ones(7),
        tolerance=0.01,
        timeout=0.01,
        dwell_steps=1,
        period=0.0,
    )

    assert not result.success
    assert result.message == "timeout"
    assert not driver.torque_enabled


def test_factr_pd_error_releases_torque() -> None:
    """Exceptions during control release torque before propagating."""
    driver = _FakeActuatorDriver()
    driver.fail_on_current = True
    actuator = GelloJointActuator(GelloDynamixelBus(driver=driver))

    with pytest.raises(RuntimeError, match="current write failed"):
        actuator.move_to_joints_factr_pd(
            np.ones(7),
            tolerance=0.01,
            timeout=1.0,
            dwell_steps=1,
            period=0.0,
        )

    assert not driver.torque_enabled


def test_factr_pd_pads_current_for_extra_servo() -> None:
    """Current-mode PD pads non-arm servos with zero current."""
    driver = _FakeActuatorDriver(num_servos=8)
    actuator = GelloJointActuator(
        GelloDynamixelBus(driver=driver),
        kp=np.full(7, 200.0),
        kd=np.zeros(7),
        current_limit=np.full(7, 100.0),
    )

    result = actuator.move_to_joints_factr_pd(
        np.full(7, 0.2),
        tolerance=0.03,
        timeout=1.0,
        dwell_steps=1,
        period=0.0,
    )

    assert result.success
    assert driver.currents
    assert driver.currents[0].shape == (8,)
    assert driver.currents[0][7] == 0.0
