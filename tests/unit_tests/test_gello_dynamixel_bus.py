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

import numpy as np

from rlinf.envs.realworld.common.gello import GelloDynamixelBus, GelloJointExpert


class _FakeDynamixelDriver:
    """Fake driver that records delegation and detects concurrent access."""

    def __init__(self) -> None:
        self.joints = np.arange(8, dtype=np.float64)
        self.writes: list[np.ndarray] = []
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.closed = False
        self.active = 0
        self.max_active = 0
        self.counter_lock = threading.Lock()

    def get_joints(self) -> np.ndarray:
        with self.counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            return self.joints.copy()
        finally:
            with self.counter_lock:
                self.active -= 1

    def set_joints(self, joints: np.ndarray) -> str:
        self.writes.append(joints.copy())
        return "written"

    def set_mode(self, mode: str, *, current_limit: float) -> tuple[str, float]:
        self.calls.append(("set_mode", (mode,), {"current_limit": current_limit}))
        return mode, current_limit

    def close(self) -> None:
        self.closed = True


def test_locked_reads_delegate_to_fake_driver() -> None:
    """Concurrent readers are serialized through one bus lock."""
    driver = _FakeDynamixelDriver()
    bus = GelloDynamixelBus(driver=driver)
    outputs: list[np.ndarray] = []
    threads = [
        threading.Thread(target=lambda: outputs.append(bus.read_joints()))
        for _ in range(8)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert driver.max_active == 1
    assert len(outputs) == len(threads)
    for output in outputs:
        np.testing.assert_array_equal(output, driver.joints)


def test_write_call_and_close_delegate_to_fake_driver() -> None:
    """Bus helpers delegate while preserving the injected fake driver."""
    driver = _FakeDynamixelDriver()
    bus = GelloDynamixelBus(driver=driver)
    target = np.linspace(-1.0, 1.0, 8)

    assert bus.write_joints(target) == "written"
    assert bus.call_locked("set_mode", "current", current_limit=0.2) == (
        "current",
        0.2,
    )
    bus.close()

    np.testing.assert_allclose(driver.writes[0], target)
    assert driver.calls == [("set_mode", ("current",), {"current_limit": 0.2})]
    assert driver.closed


def test_joint_expert_reads_from_shared_bus() -> None:
    """GelloJointExpert can consume a shared bus instead of opening a port."""
    driver = _FakeDynamixelDriver()
    driver.joints = np.array(
        [
            2.0 * np.pi + 0.1,
            -2.0 * np.pi + 0.2,
            0.3,
            -1.4,
            0.5,
            2.0 * np.pi + 1.7,
            -0.6,
            0.75,
        ],
        dtype=np.float64,
    )
    bus = GelloDynamixelBus(driver=driver)
    expert = GelloJointExpert(bus=bus)

    try:
        for _ in range(100):
            if expert.ready:
                break
            time.sleep(0.01)
        assert expert.ready
        joints, gripper = expert.get_action()
    finally:
        expert.close()

    np.testing.assert_allclose(joints, [0.1, 0.2, 0.3, -1.4, 0.5, 1.7, -0.6])
    np.testing.assert_allclose(gripper, [0.75])
