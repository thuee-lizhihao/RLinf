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

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.utils.logging import get_logger

from ..so101_env import SO101Env, SO101RobotConfig


@dataclass
class SO101ReachConfig(SO101RobotConfig):
    """A minimal reach task: move EE to target pose."""

    # Stored as euler xyz for orientation: [x, y, z, rx, ry, rz]
    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    use_dense_reward: bool = False
    dense_reward_position_scale: float = 500.0
    dense_reward_orientation_scale: float = 4.0
    success_bonus: float = 1.0

    def __post_init__(self):
        self.target_ee_pose = np.array(self.target_ee_pose, dtype=float)
        self.reward_threshold = np.array(self.reward_threshold, dtype=float)
        self.action_scale = np.array(self.action_scale, dtype=float)
        self.random_reset_joint_delta_deg = np.array(self.random_reset_joint_delta_deg, dtype=float)


class SO101ReachEnv(SO101Env):
    def __init__(
        self,
        override_cfg,
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        default_dummy: bool = False,
    ):
        self._logger = get_logger()
        override_cfg = dict(override_cfg)
        if default_dummy:
            override_cfg.setdefault("is_dummy", True)
            override_cfg.setdefault("camera_serials", ["dummy_camera_1"])
            override_cfg.setdefault("enable_camera_player", False)
        config = SO101ReachConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "so101 reach target pose"

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = super().step(action)

        # Reward based on distance to target ee pose (in base frame; after wrappers it may be relative)
        if self.config.is_dummy:
            reward = 0.0 if not self.config.use_dense_reward else float(np.exp(-1.0))
            info.update(
                {
                    "task_success": False,
                    "task_pos_error_m": float("nan"),
                    "task_rot_error_rad": float("nan"),
                    "task_dense_reward": reward,
                    "task_sparse_reward": 0.0,
                }
            )
            return obs, reward, terminated, truncated, info

        tcp_pose = obs["state"]["tcp_pose"]
        euler = R.from_quat(tcp_pose[3:].copy()).as_euler("xyz")
        cur = np.concatenate([tcp_pose[:3], euler], axis=0)
        raw_delta = cur - self.config.target_ee_pose
        raw_delta[3:] = np.arctan2(np.sin(raw_delta[3:]), np.cos(raw_delta[3:]))
        delta = np.abs(raw_delta)

        pos_error = float(np.linalg.norm(delta[:3]))
        rot_error = float(np.linalg.norm(delta[3:]))
        success = bool(np.all(delta <= self.config.reward_threshold))
        if success:
            sparse_reward = float(self.config.success_bonus)
            dense_reward = sparse_reward
            reward = sparse_reward
            terminated = True
        else:
            if self.config.use_dense_reward:
                dense_reward = float(
                    np.exp(
                        -float(self.config.dense_reward_position_scale) * np.sum(np.square(delta[:3]))
                        -float(self.config.dense_reward_orientation_scale) * np.sum(np.square(delta[3:]))
                    )
                )
                sparse_reward = 0.0
                reward = dense_reward
            else:
                dense_reward = 0.0
                sparse_reward = 0.0
                reward = 0.0
        info.update(
            {
                "task_success": success,
                "task_target_delta": delta.astype(np.float32),
                "task_pos_error_m": pos_error,
                "task_rot_error_rad": rot_error,
                "task_dense_reward": dense_reward,
                "task_sparse_reward": sparse_reward,
            }
        )
        return obs, reward, terminated, truncated, info

