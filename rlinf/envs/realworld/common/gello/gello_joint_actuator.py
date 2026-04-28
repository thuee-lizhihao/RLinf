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

"""GELLO leader actuator using a shared Dynamixel bus."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rlinf.envs.realworld.common.gello.gello_dynamixel_bus import (
    GelloDynamixelBus,
)
from rlinf.envs.realworld.common.gello.gello_joint_mapper import GelloJointMapper

CURRENT_CONTROL_MODE = 0


@dataclass(frozen=True)
class GelloJointActuatorResult:
    """Result returned by a GELLO leader alignment command."""

    success: bool
    strategy: str
    error: np.ndarray
    elapsed: float
    message: str = ""


class GelloJointActuator:
    """Actuate a GELLO leader through a shared low-level bus.

    The actuator never opens a serial port. It only uses the injected
    :class:`GelloDynamixelBus`, so a reader and actuator can safely share one
    Dynamixel chain.

    Args:
        bus: Shared GELLO Dynamixel bus.
        mapper: Calibration mapper between raw motor and robot joint space.
        current_limit: Per-joint current limits in mA for ``factr_pd``.
        kp: Per-joint proportional gains for ``factr_pd``.
        kd: Per-joint derivative gains for ``factr_pd``.
    """

    def __init__(
        self,
        bus: GelloDynamixelBus,
        mapper: GelloJointMapper | None = None,
        current_limit: Sequence[float] | np.ndarray | None = None,
        kp: Sequence[float] | np.ndarray | None = None,
        kd: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self.bus = bus
        self.mapper = mapper or GelloJointMapper()
        if self.mapper.num_joints != 7:
            raise ValueError(
                "GelloJointActuator expects a 7-joint mapper, got "
                f"{self.mapper.num_joints}"
            )
        self.current_limit = self._vector_or_default(current_limit, 150.0)
        self.kp = self._vector_or_default(kp, 80.0)
        self.kd = self._vector_or_default(kd, 3.0)

    def enable_torque(self) -> None:
        """Enable Dynamixel torque on the shared bus."""
        self.bus.set_torque_enabled(True)

    def disable_torque(self) -> None:
        """Disable Dynamixel torque on the shared bus."""
        self.bus.set_torque_enabled(False)

    def read_joints(self) -> np.ndarray:
        """Read the current GELLO leader joints in calibrated joint space."""
        raw = self.bus.read_joints()
        return self.mapper.raw_to_joint(raw[: self.mapper.num_joints])

    def move_to_joints_factr_pd(
        self,
        target_q: Sequence[float] | np.ndarray,
        *,
        tolerance: float = 0.06,
        timeout: float = 5.0,
        dwell_steps: int = 5,
        period: float = 0.01,
        current_limit: Sequence[float] | np.ndarray | None = None,
        kp: Sequence[float] | np.ndarray | None = None,
        kd: Sequence[float] | np.ndarray | None = None,
        release_on_success: bool = True,
    ) -> GelloJointActuatorResult:
        """Move the GELLO leader to ``target_q`` with current-mode PD control.

        Args:
            target_q: Target leader joints in calibrated joint space.
            tolerance: Maximum absolute joint error considered aligned.
            timeout: Maximum control duration in seconds.
            dwell_steps: Number of consecutive in-tolerance cycles required.
            period: Control-loop period in seconds.
            current_limit: Optional per-command current limits in mA.
            kp: Optional per-command proportional gains.
            kd: Optional per-command derivative gains.
            release_on_success: If ``True``, disable torque after alignment.

        Returns:
            Alignment result with final joint error.
        """
        target = self._as_joint_vector(target_q, "target_q")
        limit = self._vector_or_default(current_limit, self.current_limit)
        p_gain = self._vector_or_default(kp, self.kp)
        d_gain = self._vector_or_default(kd, self.kd)
        start = time.monotonic()
        dwell = 0
        last_error = np.full(self.mapper.num_joints, np.inf, dtype=np.float64)

        try:
            self._set_control_mode(CURRENT_CONTROL_MODE)
            while True:
                q, qd = self._read_joint_positions_and_velocities()
                last_error = target - q
                if float(np.max(np.abs(last_error))) <= tolerance:
                    dwell += 1
                    if dwell >= dwell_steps:
                        if release_on_success:
                            self.disable_torque()
                        return GelloJointActuatorResult(
                            success=True,
                            strategy="factr_pd",
                            error=last_error,
                            elapsed=time.monotonic() - start,
                        )
                else:
                    dwell = 0

                if time.monotonic() - start >= timeout:
                    self.disable_torque()
                    return GelloJointActuatorResult(
                        success=False,
                        strategy="factr_pd",
                        error=last_error,
                        elapsed=time.monotonic() - start,
                        message="timeout",
                    )

                joint_current = p_gain * last_error - d_gain * qd
                raw_current = self.mapper.signs * joint_current
                raw_current = np.clip(raw_current, -limit, limit)
                self.bus.write_currents(raw_current)
                time.sleep(period)
        except Exception:
            self.emergency_release()
            raise

    def emergency_release(self) -> None:
        """Best-effort torque release for error paths."""
        try:
            self.disable_torque()
        except Exception:
            pass

    def close(self) -> None:
        """Release torque and close the shared bus."""
        self.emergency_release()
        self.bus.close()

    def _set_control_mode(self, mode: int) -> None:
        self.disable_torque()
        self.bus.set_operating_mode(mode)
        self.bus.verify_operating_mode(mode)
        self.enable_torque()

    def _read_joint_positions_and_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        raw_q, raw_qd = self.bus.read_positions_and_velocities()
        q = self.mapper.raw_to_joint(raw_q[: self.mapper.num_joints])
        qd = self.mapper.signs * raw_qd[: self.mapper.num_joints]
        return q, qd

    def _as_joint_vector(
        self,
        values: Sequence[float] | np.ndarray,
        name: str,
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        expected = (self.mapper.num_joints,)
        if arr.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def _vector_or_default(
        self,
        values: Sequence[float] | np.ndarray | float | None,
        default: Sequence[float] | np.ndarray | float,
    ) -> np.ndarray:
        source = default if values is None else values
        arr = np.asarray(source, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(self.mapper.num_joints, float(arr), dtype=np.float64)
        expected = (self.mapper.num_joints,)
        if arr.shape != expected:
            raise ValueError(f"value must have shape {expected}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("value must contain only finite values")
        return arr.copy()
