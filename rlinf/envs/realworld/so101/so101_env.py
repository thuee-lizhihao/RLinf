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
import json
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
    ik_position_only_orientation_weight: float = 0.0
    ik_position_error_threshold_m: float = 0.02
    ik_orientation_error_threshold_deg: float = 30.0

    # Reset / safety
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    enable_random_reset: bool = False
    random_reset_joint_delta_deg: np.ndarray = field(default_factory=lambda: np.zeros(5))
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
    max_xyz_step_m: float = 0.05
    max_rpy_step_rad: float = 0.3
    max_joint_step_delta_deg: float = 8.0
    joint_tolerance_deg: float = 3.0
    step_timeout_s: float = 0.5
    reset_joint_tolerance_deg: float = 4.0
    reset_max_step_delta_deg: float = 4.0
    reset_timeout_s: float = 2.0
    reset_max_iters: int = 6
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

    def _clip_action_deltas(self, xyz_delta: np.ndarray, rpy_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipped_xyz = np.clip(
            np.asarray(xyz_delta, dtype=float),
            -float(self.config.max_xyz_step_m),
            float(self.config.max_xyz_step_m),
        )
        clipped_rpy = np.clip(
            np.asarray(rpy_delta, dtype=float),
            -float(self.config.max_rpy_step_rad),
            float(self.config.max_rpy_step_rad),
        )
        return clipped_xyz, clipped_rpy

    def _limit_joint_step(self, q_curr: np.ndarray, q_target: np.ndarray, max_step: float) -> tuple[np.ndarray, bool]:
        delta = np.asarray(q_target, dtype=float) - np.asarray(q_curr, dtype=float)
        clipped_delta = np.clip(delta, -float(max_step), float(max_step))
        clipped = bool(np.any(np.abs(clipped_delta - delta) > 1e-9))
        return np.asarray(q_curr, dtype=float) + clipped_delta, clipped

    @staticmethod
    def _pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = R.from_quat(np.asarray(pose7[3:], dtype=float)).as_matrix()
        T[:3, 3] = np.asarray(pose7[:3], dtype=float)
        return T

    def _solve_ik_with_fallback(self, q_curr: np.ndarray, desired_pose7: np.ndarray) -> tuple[np.ndarray, dict]:
        assert self._kin is not None
        T_des = self._pose7_to_matrix(desired_pose7)
        diag: dict[str, float | str | bool] = {
            "ik_fallback_level": "hold_current",
            "ik_converged": False,
            "ik_position_error_m": np.nan,
            "ik_orientation_error_deg": np.nan,
            "ik_exception": False,
            "ik_retry_used": False,
            "ik_retry_attempt": 0,
        }

        try:
            full = self._kin.inverse_kinematics_with_result(
                current_joint_pos_deg=q_curr,
                desired_ee_pose=T_des,
                position_weight=float(self.config.ik_position_weight),
                orientation_weight=float(self.config.ik_orientation_weight),
                position_error_threshold_m=float(self.config.ik_position_error_threshold_m),
                orientation_error_threshold_deg=float(self.config.ik_orientation_error_threshold_deg),
                check_orientation=True,
            )
            diag["ik_position_error_m"] = full.position_error_m
            diag["ik_orientation_error_deg"] = full.orientation_error_deg
            if full.converged:
                diag["ik_fallback_level"] = "full"
                diag["ik_converged"] = True
                return full.joint_pos_deg.astype(float), diag
        except Exception as e:
            self._logger.warning(f"SO101 full IK failed: {e}")
            diag["ik_exception"] = True

        try:
            pos_only = self._kin.inverse_kinematics_with_result(
                current_joint_pos_deg=q_curr,
                desired_ee_pose=T_des,
                position_weight=float(self.config.ik_position_weight),
                orientation_weight=float(self.config.ik_position_only_orientation_weight),
                position_error_threshold_m=float(self.config.ik_position_error_threshold_m),
                orientation_error_threshold_deg=float(self.config.ik_orientation_error_threshold_deg),
                check_orientation=False,
            )
            diag["ik_position_error_m"] = pos_only.position_error_m
            diag["ik_orientation_error_deg"] = pos_only.orientation_error_deg
            if pos_only.converged:
                diag["ik_fallback_level"] = "position_only"
                diag["ik_converged"] = True
                diag["ik_exception"] = False  # Reset: valid fallback solution obtained
                return pos_only.joint_pos_deg.astype(float), diag
        except Exception as e:
            self._logger.warning(f"SO101 position-only IK failed: {e}")
            diag["ik_exception"] = True

        # Position-only IK can be non-deterministic near singular/branch boundaries.
        # A short retry often recovers a valid solution without changing control policy.
        for retry_idx in range(2):
            try:
                pos_only_retry = self._kin.inverse_kinematics_with_result(
                    current_joint_pos_deg=q_curr,
                    desired_ee_pose=T_des,
                    position_weight=float(self.config.ik_position_weight),
                    orientation_weight=float(self.config.ik_position_only_orientation_weight),
                    position_error_threshold_m=float(self.config.ik_position_error_threshold_m),
                    orientation_error_threshold_deg=float(self.config.ik_orientation_error_threshold_deg),
                    check_orientation=False,
                )
                diag["ik_position_error_m"] = pos_only_retry.position_error_m
                diag["ik_orientation_error_deg"] = pos_only_retry.orientation_error_deg
                if pos_only_retry.converged:
                    diag["ik_fallback_level"] = "position_only_retry"
                    diag["ik_converged"] = True
                    diag["ik_exception"] = False
                    diag["ik_retry_used"] = True
                    diag["ik_retry_attempt"] = int(retry_idx + 1)
                    return pos_only_retry.joint_pos_deg.astype(float), diag
            except Exception as e:
                self._logger.warning(f"SO101 position-only IK retry failed: {e}")
                diag["ik_exception"] = True

        return np.asarray(q_curr, dtype=float).copy(), diag

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
        info: dict[str, float | bool | str] = {}
        if not self.config.is_dummy:
            try:
                rest_info = self.go_to_rest()
                info.update(rest_info)
            except Exception as e:
                self._logger.warning(f"SO101 reset go_to_rest failed (non-fatal): {e}")
                info["reset_go_to_rest_ok"] = False
                info["reset_error"] = str(e)
        obs = self._get_observation()
        return obs, info

    def go_to_rest(self) -> dict[str, float | bool]:
        if self.config.is_dummy:
            return {"reset_go_to_rest_ok": True, "reset_iters": 0}
        assert self._controller is not None

        def _append_debug_log(
            run_id: str,
            hypothesis_id: str,
            location: str,
            message: str,
            data: dict[str, Any],
        ) -> None:
            payload = {
                "sessionId": "c6202f",
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            try:
                with Path("/home/zhihao/SO101_arm/.cursor/debug-c6202f.log").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass

        run_id = f"reset_ep{self._num_steps}"
        reset_guard_band_deg = 0.5
        applied_reset_step_limit_deg = max(
            0.1,
            float(self.config.reset_max_step_delta_deg) - float(reset_guard_band_deg),
        )
        self._refresh_state_from_controller()
        q_target = np.asarray(self.config.joint_reset_qpos, dtype=float).reshape(5)

        if self.config.enable_random_reset:
            noise = np.asarray(self.config.random_reset_joint_delta_deg, dtype=float).reshape(5)
            q_target = q_target + np.random.uniform(-noise, noise)

        q_curr = self._state.arm_joint_position.astype(float).reshape(5)
        iters = 0
        success = False
        max_err = float(np.max(np.abs(q_target - q_curr)))
        # region agent log
        _append_debug_log(
            run_id=run_id,
            hypothesis_id="R1",
            location="so101_env.py:go_to_rest:entry",
            message="Reset entry state and target.",
            data={
                "q_curr_deg": q_curr.tolist(),
                "q_target_deg": q_target.tolist(),
                "max_err_deg": float(max_err),
                "applied_reset_step_limit_deg": float(applied_reset_step_limit_deg),
                "reset_max_step_delta_deg": float(self.config.reset_max_step_delta_deg),
                "reset_guard_band_deg": float(reset_guard_band_deg),
                "reset_joint_tolerance_deg": float(self.config.reset_joint_tolerance_deg),
            },
        )
        # endregion
        while iters < int(self.config.reset_max_iters):
            iters += 1
            q_next, _ = self._limit_joint_step(
                q_curr, q_target, max_step=float(applied_reset_step_limit_deg)
            )
            # region agent log
            _append_debug_log(
                run_id=run_id,
                hypothesis_id="R2",
                location="so101_env.py:go_to_rest:pre_move",
                message="Reset loop command before move_joints_blocking.",
                data={
                    "iter": int(iters),
                    "q_curr_deg": q_curr.tolist(),
                    "q_next_deg": np.asarray(q_next, dtype=float).tolist(),
                    "max_abs_cmd_delta_deg": float(np.max(np.abs(np.asarray(q_next) - q_curr))),
                    "applied_reset_step_limit_deg": float(applied_reset_step_limit_deg),
                    "configured_reset_step_limit_deg": float(self.config.reset_max_step_delta_deg),
                },
            )
            # endregion
            try:
                move_res = self._controller.move_joints_blocking(
                    q_next,
                    gripper_pos=None,
                    timeout_s=float(self.config.reset_timeout_s),
                    joint_tolerance_deg=float(self.config.reset_joint_tolerance_deg),
                    max_step_delta_deg=float(self.config.reset_max_step_delta_deg),
                    raise_on_fail=False,
                ).wait()[0]
            except Exception as e:
                # region agent log
                _append_debug_log(
                    run_id=run_id,
                    hypothesis_id="R4",
                    location="so101_env.py:go_to_rest:move_exception",
                    message="Reset move raised exception in controller.",
                    data={
                        "iter": int(iters),
                        "exception_type": type(e).__name__,
                        "exception_text": str(e),
                        "max_abs_cmd_delta_deg": float(np.max(np.abs(np.asarray(q_next) - q_curr))),
                        "applied_reset_step_limit_deg": float(applied_reset_step_limit_deg),
                        "configured_reset_step_limit_deg": float(self.config.reset_max_step_delta_deg),
                    },
                )
                # endregion
                raise
            q_curr = np.asarray(move_res.readback_joint_deg, dtype=float).reshape(5)
            max_err = float(np.max(np.abs(q_target - q_curr)))
            # region agent log
            _append_debug_log(
                run_id=run_id,
                hypothesis_id="R3",
                location="so101_env.py:go_to_rest:post_move",
                message="Reset loop move result and readback.",
                data={
                    "iter": int(iters),
                    "move_success": bool(move_res.success),
                    "move_timed_out": bool(move_res.timed_out),
                    "move_max_joint_error_deg": float(move_res.max_joint_error_deg),
                    "readback_joint_deg": np.asarray(move_res.readback_joint_deg, dtype=float).tolist(),
                    "remaining_max_err_deg": float(max_err),
                },
            )
            # endregion
            if max_err <= float(self.config.reset_joint_tolerance_deg) + 1e-9:
                success = True
                break

        return {
            "reset_go_to_rest_ok": bool(success),
            "reset_iters": int(iters),
            "reset_target_max_err_deg": float(max_err),
        }

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
        xyz_delta_raw = action[:3] * float(self.config.action_scale[0])
        rpy_delta_raw = action[3:6] * float(self.config.action_scale[1])
        xyz_delta, rpy_delta = self._clip_action_deltas(xyz_delta_raw, rpy_delta_raw)
        gripper_cmd = float(action[6] * float(self.config.action_scale[2]))

        desired_pose = self._state.tcp_pose.copy()
        desired_pose[:3] = desired_pose[:3] + xyz_delta
        desired_pose[3:] = (
            R.from_euler("xyz", rpy_delta) * R.from_quat(desired_pose[3:].copy())
        ).as_quat()
        unclipped_desired_pose = desired_pose.copy()
        desired_pose = self._clip_pose_to_limits(desired_pose)

        q_curr = self._state.arm_joint_position.astype(float)
        q_target_raw, ik_diag = self._solve_ik_with_fallback(q_curr, desired_pose)
        q_target, clipped_by_joint_step = self._limit_joint_step(
            q_curr, q_target_raw, max_step=float(self.config.max_joint_step_delta_deg)
        )

        # Gripper: binary open/close
        is_gripper_effective = False
        gripper_error = False
        try:
            if gripper_cmd <= -self.config.binary_gripper_threshold:
                self._controller.close_gripper().wait()
                is_gripper_effective = True
            elif gripper_cmd >= self.config.binary_gripper_threshold:
                self._controller.open_gripper().wait()
                is_gripper_effective = True
        except Exception as e:
            gripper_error = True
            self._logger.warning(f"SO101 gripper command failed (non-fatal): {e}")

        # Send joint targets using blocking wrapper.
        move_success = False
        move_timed_out = False
        move_max_joint_error_deg = np.nan
        try:
            _move_timeout_budget = max(
                float(self.config.step_timeout_s), 1.0 / float(self.config.step_frequency)
            )
            move_res = self._controller.move_joints_blocking(
                q_target,
                gripper_pos=None,
                timeout_s=_move_timeout_budget,
                joint_tolerance_deg=float(self.config.joint_tolerance_deg),
                max_step_delta_deg=float(self.config.max_joint_step_delta_deg),
                raise_on_fail=False,
            ).wait()[0]
            move_success = bool(move_res.success)
            move_timed_out = bool(move_res.timed_out)
            move_max_joint_error_deg = float(move_res.max_joint_error_deg)
        except Exception as e:
            self._logger.warning(f"SO101 joint move failed (non-fatal): {e}")

        self._num_steps += 1
        # rate limit
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "gripper_effective": is_gripper_effective,
            "gripper_command_error": gripper_error,
            "action_xyz_delta_clipped": bool(np.any(np.abs(xyz_delta - xyz_delta_raw) > 1e-9)),
            "action_rpy_delta_clipped": bool(np.any(np.abs(rpy_delta - rpy_delta_raw) > 1e-9)),
            "desired_pose_clipped_to_limits": bool(
                np.any(np.abs(desired_pose - unclipped_desired_pose) > 1e-9)
            ),
            "joint_step_clipped": clipped_by_joint_step,
            "move_success": move_success,
            "move_timed_out": move_timed_out,
            "move_max_joint_error_deg": move_max_joint_error_deg,
            "ik_fallback_level": ik_diag["ik_fallback_level"],
            "ik_converged": ik_diag["ik_converged"],
            "ik_position_error_m": ik_diag["ik_position_error_m"],
            "ik_orientation_error_deg": ik_diag["ik_orientation_error_deg"],
            "ik_exception": ik_diag["ik_exception"],
        }
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

