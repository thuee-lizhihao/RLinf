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

"""GELLO teleoperation interface that provides raw joint positions.

Unlike :class:`GelloExpert` which converts GELLO joint readings to TCP poses
via forward kinematics, this module exposes the raw 7-DOF joint positions
directly.  This is used for joint-space control where the GELLO joint
positions map 1:1 to the robot's joint positions.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from rlinf.envs.realworld.common.gello.gello_joint_mapper import GelloJointMapper

if TYPE_CHECKING:
    from rlinf.envs.realworld.common.gello.gello_dynamixel_bus import (
        GelloDynamixelBus,
    )

# Mid-point of each Franka Panda joint range — used as the seed for angle
# unwrapping so that the first Dynamixel reading lands inside the valid range
# even when it is offset by 2kπ after calibration.
# Note: J4 centre ≈ −1.57, J6 centre ≈ 1.87 (J6 upper limit 3.75 > π).
_FRANKA_RANGE_CENTER = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0])


class GelloJointExpert:
    """Interface to the GELLO teleoperation device (joint-space output).

    Continuously reads GELLO joint positions in a background thread and
    exposes them directly through :meth:`get_action`, without performing
    forward kinematics.

    Dynamixel readings may be offset by 2kπ from the robot's valid range.
    To avoid dangerous discontinuities, joint angles are *unwrapped*:
    the first reading is mapped to the nearest equivalent within each
    joint's valid range, and every subsequent reading is kept continuous
    with the previous one (no single-step jump > π).

    Args:
        port: Serial port of the GELLO device. Keeps the legacy
            ``GelloTeleopAgent`` reader path.
        bus: Shared low-level Dynamixel bus. When provided, the expert reads
            raw motor positions from this bus instead of opening ``port``.
        mapper: Mapper used to convert raw bus readings to joint positions.
    """

    def __init__(
        self,
        port: str | None = None,
        bus: "GelloDynamixelBus | None" = None,
        mapper: GelloJointMapper | None = None,
    ) -> None:
        if port is None and bus is None:
            raise ValueError("either port or bus must be provided")
        if port is not None and bus is not None:
            raise ValueError("provide either port or bus, not both")

        self.bus = bus
        self.mapper = mapper or GelloJointMapper()
        if self.mapper.num_joints != 7:
            raise ValueError(
                "GelloJointExpert expects a 7-joint mapper, got "
                f"{self.mapper.num_joints}"
            )
        self.agent = None
        if self.bus is None:
            from gello_teleop.gello_teleop_agent import GelloTeleopAgent

            self.agent = GelloTeleopAgent(port=port)

        self.state_lock = threading.Lock()
        self._running = True
        self._ready = False
        self._prev_joints: np.ndarray | None = None
        self.latest_data = {
            "joint_positions": np.zeros(7),
            "gripper": np.zeros(1),
        }
        self.thread = threading.Thread(target=self._read_gello, daemon=True)
        self.thread.start()

    def _read_action_once(self) -> tuple[np.ndarray, np.ndarray]:
        """Read one GELLO frame from either the legacy agent or shared bus."""
        if self.bus is not None:
            raw = self.bus.read_joints()
            if raw.shape[0] < self.mapper.num_joints:
                raise ValueError(
                    "GELLO bus returned fewer joints than mapper expects: "
                    f"{raw.shape[0]} < {self.mapper.num_joints}"
                )
            joints = self.mapper.raw_to_joint(raw[: self.mapper.num_joints])
            gripper = (
                raw[self.mapper.num_joints]
                if raw.shape[0] > self.mapper.num_joints
                else 0.0
            )
            return joints, np.array([gripper], dtype=np.float64)

        if self.agent is None:
            raise RuntimeError("GELLO agent is not initialized")
        gello_joints, gello_gripper = self.agent.get_action()
        return np.asarray(gello_joints, dtype=np.float64), np.array(
            [gello_gripper],
            dtype=np.float64,
        )

    def _read_gello(self):
        import time

        consecutive_errors = 0
        max_consecutive_errors = 50

        while self._running:
            try:
                joints, gello_gripper = self._read_action_once()
                ref = (
                    self._prev_joints
                    if self._prev_joints is not None
                    else _FRANKA_RANGE_CENTER
                )
                joints = self.mapper.unwrap(joints, ref)
                self._prev_joints = joints

                with self.state_lock:
                    self.latest_data["joint_positions"] = joints.copy()
                    self.latest_data["gripper"] = gello_gripper
                    self._ready = True
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    with self.state_lock:
                        self._ready = False
                backoff = min(0.1, 0.001 * (2 ** min(consecutive_errors, 7)))
                time.sleep(backoff)
                continue

            time.sleep(0.001)

    @property
    def ready(self) -> bool:
        """Whether at least one GELLO frame has been received."""
        with self.state_lock:
            return self._ready

    def get_action(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(joint_positions, gripper)`` from the latest GELLO reading.

        Returns:
            A tuple of ``(joint_positions[7], gripper[1])``.
        """
        with self.state_lock:
            return (
                self.latest_data["joint_positions"].copy(),
                self.latest_data["gripper"].copy(),
            )

    def close(self) -> None:
        """Stop the background reader thread."""
        self._running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Test the GELLO joint expert.")
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port of the GELLO device.",
    )
    args = parser.parse_args()

    gello = GelloJointExpert(port=args.port)
    with np.printoptions(precision=3, suppress=True):
        while True:
            joint_positions, gripper = gello.get_action()
            print(
                f"joints={joint_positions}  gripper={gripper}",
                end="\r",
            )
            time.sleep(0.1)
