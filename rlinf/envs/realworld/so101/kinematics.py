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

from __future__ import annotations

import numpy as np


class SO101Kinematics:
    """SO101 forward/inverse kinematics using placo.

    This mirrors LeRobot's `RobotKinematics` approach but keeps the dependency optional.
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        try:
            import placo  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "placo is required for SO101 kinematics. "
                "Install placo, or run SO101 env in is_dummy mode."
            ) from e

        self._placo = placo
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)

        self.target_frame_name = target_frame_name
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # IK task
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        joint_pos_deg = np.asarray(joint_pos_deg, dtype=float).reshape(-1)
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos_deg: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        current_joint_pos_deg = np.asarray(current_joint_pos_deg, dtype=float).reshape(-1)
        current_joint_rad = np.deg2rad(current_joint_pos_deg[: len(self.joint_names)])
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        self.tip_frame.T_world_frame = desired_ee_pose
        self.tip_frame.configure(
            self.target_frame_name, "soft", position_weight, orientation_weight
        )
        self.solver.solve(True)
        self.robot.update_kinematics()

        joint_pos_rad = [self.robot.get_joint(j) for j in self.joint_names]
        return np.rad2deg(joint_pos_rad)

