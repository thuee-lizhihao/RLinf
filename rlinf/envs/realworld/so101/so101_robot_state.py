# Copyright 2025 The RLinf Authors.
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

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class SO101RobotState:
    """Minimal robot state required by RLinf realworld wrappers.

    By convention:
    - tcp_pose: xyz + quat (7,)
    - tcp_vel: spatial velocity (vx,vy,vz,wx,wy,wz) (6,)
    """

    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))

    # SO101 is 5-DoF arm + 1-DoF gripper in LeRobot convention
    arm_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32)
    )
    arm_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32)
    )

    # Gripper motor is typically normalized to [0, 100] in LeRobot.
    gripper_position: float = 0.0
    gripper_open: bool = True
    timestamp_s: float = 0.0
    is_robot_up: bool = False
    last_joint_max_error_deg: float = 0.0
    last_joint_mean_error_deg: float = 0.0
    last_ee_pos_error_m: float = 0.0
    last_ee_rot_error_deg: float = 0.0

    def to_dict(self):
        return asdict(self)

