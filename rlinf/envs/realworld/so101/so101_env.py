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

import copy
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import SO101HWInfo, WorkerInfo
from rlinf.utils.logging import get_logger

from .kinematics import SO101Kinematics
from .so101_robot_state import SO101RobotState


@dataclass
class SO101RobotConfig:
    port: Optional[str] = None
    camera_serials: Optional[list[str]] = None
    enable_camera_player: bool = True
    calibration_path: Optional[str] = None

    # Control
    is_dummy: bool = False
    step_frequency: float = 10.0
    action_scale: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.1, 1.0]))
    binary_gripper_threshold: float = 0.5

    # Kinematics
    urdf_path: Optional[str] = None
    target_frame_name: str = "gripper_frame_link"
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 0.01

    # Reset / safety
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    ee_pose_limit_min: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -0.33838826543491113,
                -0.43320932355460706,
                -0.221904771238294,
                -3.1414882589123065,
                -1.5616055459910023,
                -3.1410440964660715,
            ]
        )
    )
    ee_pose_limit_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.47648009544149583,
                0.43426311165235965,
                0.5260975248250594,
                3.1415720814029955,
                1.538700826334587,
                3.14075636588622,
            ]
        )
    )
    min_state_dt: float = 1e-3
    vel_smoothing_alpha: float = 0.0
    tcp_linear_vel_max: float = 1.0
    tcp_angular_vel_max: float = 10.0
    tcp_pose_jump_warn_threshold: float = 0.05
    max_num_steps: int = 100


class SO101Env(gym.Env):
    """SO101 real robot environment with RealSense observations.

    Observation format matches FrankaEnv:
    {
      "state": {"tcp_pose"(7), "tcp_vel"(6), "gripper_position"(1)},
      "frames": {"wrist_1": (128,128,3), ...}
    }
    """

    def __init__(
        self,
        config: SO101RobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[SO101HWInfo],
        env_idx: int,
    ):
        self._logger = get_logger()
        self.config = config
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._state = SO101RobotState()
        self._num_steps = 0

        self._kin = None
        self._controller = None

        if not self.config.is_dummy:
            self._setup_hardware()

        assert (
            self.config.camera_serials is not None and len(self.config.camera_serials) > 0
        ), "At least one camera serial must be provided for SO101Env."
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Init cameras
        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    def _setup_hardware(self):
        from .so101_controller import SO101Controller

        assert isinstance(self.hardware_info, SO101HWInfo), (
            f"hardware_info must be SO101HWInfo, but got {type(self.hardware_info)}."
        )
        if self.config.port is None:
            self.config.port = self.hardware_info.config.port
        if self.config.camera_serials is None:
            self.config.camera_serials = self.hardware_info.config.camera_serials

        # Default URDF path inside repo
        if self.config.urdf_path is None:
            self.config.urdf_path = str(
                Path(__file__).with_suffix("").parent / "assets" / "so101_minimal.urdf"
            )

        # Kinematics uses 5 arm joints (exclude gripper)
        self._kin = SO101Kinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
        )

        # Launch controller
        self._controller = SO101Controller.launch_controller(
            port=self.config.port,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
            calibration_path=self.config.calibration_path,
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
        )
        self._controller.connect().wait()

    def _init_action_obs_spaces(self):
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_position": gym.spaces.Box(0, 100, shape=(1,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"wrist_{k + 1}": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for k in range(len(self.config.camera_serials or []))
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _open_cameras(self):
        self._cameras: list[Camera] = []
        assert self.config.camera_serials is not None
        camera_infos = [
            CameraInfo(name=f"wrist_{i + 1}", serial_number=n)
            for i, n in enumerate(self.config.camera_serials)
        ]
        for info in camera_infos:
            camera = Camera(info)
            camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in getattr(self, "_cameras", []):
            camera.close()
        self._cameras = []

    def _crop_frame(self, frame: np.ndarray, reshape_size: tuple[int, int]):
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
        resized_frame = cv2.resize(cropped_frame, reshape_size)
        return cropped_frame, resized_frame

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        display_frames: dict[str, np.ndarray] = {}
        for camera in self._cameras:
            try:
                frame = camera.get_frame()
                reshape_size = self.observation_space["frames"][camera._camera_info.name].shape[:2][
                    ::-1
                ]
                cropped_frame, resized_frame = self._crop_frame(frame, reshape_size)
                frames[camera._camera_info.name] = resized_frame[..., ::-1]  # BGR->RGB
                display_frames[camera._camera_info.name] = resized_frame
                display_frames[f"{camera._camera_info.name}_full"] = cropped_frame
            except queue.Empty:
                self._logger.warning(
                    f"Camera {camera._camera_info.name} is not producing frames. Reopening..."
                )
                time.sleep(1)
                camera.close()
                self._open_cameras()
                return self._get_camera_frames()

        if hasattr(self, "camera_player"):
            self.camera_player.put_frame(display_frames)
        return frames

    def _clip_pose_to_limits(self, pose7: np.ndarray) -> np.ndarray:
        pose = pose7.copy()
        pose[:3] = np.clip(pose[:3], self.config.ee_pose_limit_min[:3], self.config.ee_pose_limit_max[:3])
        euler = R.from_quat(pose[3:].copy()).as_euler("xyz")
        euler = np.clip(euler, self.config.ee_pose_limit_min[3:], self.config.ee_pose_limit_max[3:])
        pose[3:] = R.from_euler("xyz", euler).as_quat()
        return pose

    def _refresh_state_from_controller(self) -> SO101RobotState:
        assert self._controller is not None
        self._state = self._controller.sync_state().wait()[0]
        return self._state

    def _get_observation(self):
        if self.config.is_dummy:
            return self._base_observation_space.sample()

        self._refresh_state_from_controller()

        frames = self._get_camera_frames()
        state = {
            "tcp_pose": self._state.tcp_pose,
            "tcp_vel": self._state.tcp_vel,
            "gripper_position": np.array([self._state.gripper_position], dtype=np.float32),
        }
        return {"state": state, "frames": frames}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._num_steps = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.config.is_dummy:
            obs = self._get_observation()
            self._num_steps += 1
            truncated = self._num_steps >= self.config.max_num_steps
            return obs, 0.0, False, truncated, {}

        assert self._controller is not None
        assert self._kin is not None

        # Update current state from controller
        self._refresh_state_from_controller()

        # Build desired tcp pose by applying deltas
        xyz_delta = action[:3] * float(self.config.action_scale[0])
        rpy_delta = action[3:6] * float(self.config.action_scale[1])
        gripper_cmd = float(action[6] * float(self.config.action_scale[2]))

        desired_pose = self._state.tcp_pose.copy()
        desired_pose[:3] = desired_pose[:3] + xyz_delta
        desired_pose[3:] = (
            R.from_euler("xyz", rpy_delta) * R.from_quat(desired_pose[3:].copy())
        ).as_quat()
        desired_pose = self._clip_pose_to_limits(desired_pose)

        # Convert desired pose to 4x4
        T_des = np.eye(4, dtype=float)
        T_des[:3, :3] = R.from_quat(desired_pose[3:].copy()).as_matrix()
        T_des[:3, 3] = desired_pose[:3]

        q_curr = self._state.arm_joint_position.astype(float)
        try:
            q_target = self._kin.inverse_kinematics(
                q_curr,
                T_des,
                position_weight=self.config.ik_position_weight,
                orientation_weight=self.config.ik_orientation_weight,
            )
        except Exception as e:
            self._logger.warning(f"SO101 IK failed, keeping current joints. Error: {e}")
            q_target = q_curr

        # Gripper: binary open/close
        is_gripper_effective = False
        if gripper_cmd <= -self.config.binary_gripper_threshold:
            self._controller.close_gripper().wait()
            is_gripper_effective = True
        elif gripper_cmd >= self.config.binary_gripper_threshold:
            self._controller.open_gripper().wait()
            is_gripper_effective = True

        # Send joint targets using blocking wrapper.
        self._controller.move_joints_blocking(
            q_target,
            gripper_pos=None,
            timeout_s=max(0.5, 1.0 / float(self.config.step_frequency)),
            joint_tolerance_deg=3.0,
            max_step_delta_deg=5.0,
            raise_on_fail=False,
        ).wait()

        self._num_steps += 1
        # rate limit
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = self._num_steps >= self.config.max_num_steps
        info = {"gripper_effective": is_gripper_effective}
        return obs, reward, terminated, truncated, info

    def close(self):
        if not self.config.is_dummy:
            try:
                self._close_cameras()
            except Exception:
                pass
            if self._controller is not None:
                try:
                    self._controller.disconnect().wait()
                except Exception:
                    pass

