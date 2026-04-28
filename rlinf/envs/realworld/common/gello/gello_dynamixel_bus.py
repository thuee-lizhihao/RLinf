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

"""Thread-safe shared owner for a GELLO Dynamixel serial bus."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np


class GelloDynamixelBus:
    """Own one GELLO Dynamixel driver and serialize all bus operations.

    Args:
        port: Serial port for the GELLO Dynamixel chain. Required unless a
            ``driver`` is injected.
        joint_ids: Dynamixel ids to open. Defaults to the seven arm joints and
            the gripper id used by existing GELLO tools.
        baudrate: Serial baudrate.
        driver: Optional prebuilt driver, used by tests and by callers that need
            custom construction.
    """

    def __init__(
        self,
        port: str | None = None,
        joint_ids: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
        baudrate: int = 1_000_000,
        driver: Any | None = None,
    ) -> None:
        if driver is None:
            if port is None:
                raise ValueError("port is required when driver is not provided")
            from gello.dynamixel.driver import DynamixelDriver

            driver = DynamixelDriver(list(joint_ids), port=port, baudrate=baudrate)

        self.driver = driver
        self.port = port
        self.joint_ids = tuple(joint_ids)
        self.baudrate = baudrate
        self._lock = threading.RLock()

    def read_joints(self) -> np.ndarray:
        """Read raw joint positions from the shared driver."""
        with self._lock:
            return np.asarray(self.driver.get_joints(), dtype=np.float64)

    def write_joints(self, joints: Sequence[float] | np.ndarray) -> Any:
        """Write raw joint targets through the shared driver."""
        joints_arr = np.asarray(joints, dtype=np.float64)
        with self._lock:
            return self.driver.set_joints(joints_arr)

    def write_currents(self, currents: Sequence[float] | np.ndarray) -> Any:
        """Write raw motor current targets through the shared driver."""
        currents_arr = np.asarray(currents, dtype=np.float64)
        with self._lock:
            return self.driver.set_current(currents_arr)

    def read_positions_and_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        """Read raw positions and velocities from the shared driver."""
        with self._lock:
            positions, velocities = self.driver.get_positions_and_velocities()
            return (
                np.asarray(positions, dtype=np.float64),
                np.asarray(velocities, dtype=np.float64),
            )

    def set_operating_mode(self, mode: int) -> Any:
        """Set the raw driver operating mode while holding the bus lock."""
        with self._lock:
            return self.driver.set_operating_mode(mode)

    def verify_operating_mode(self, expected_mode: int) -> Any:
        """Verify the raw driver operating mode while holding the bus lock."""
        with self._lock:
            return self.driver.verify_operating_mode(expected_mode)

    def set_torque_enabled(self, enabled: bool) -> Any:
        """Enable or disable Dynamixel torque while holding the bus lock."""
        with self._lock:
            return self.driver.set_torque_mode(enabled)

    def call_locked(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call an arbitrary driver method while holding the bus lock.

        This keeps future actuator operations serialized with the reader even
        when the underlying driver exposes hardware-specific method names.

        Args:
            method_name: Name of the driver method to call.
            *args: Positional arguments passed to the driver method.
            **kwargs: Keyword arguments passed to the driver method.

        Returns:
            The driver method's return value.
        """
        with self._lock:
            method = getattr(self.driver, method_name)
            return method(*args, **kwargs)

    def close(self) -> None:
        """Close the underlying driver if it exposes a close method."""
        with self._lock:
            close = getattr(self.driver, "close", None)
            if close is not None:
                close()
