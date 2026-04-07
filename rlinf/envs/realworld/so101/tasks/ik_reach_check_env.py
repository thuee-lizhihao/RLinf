#!/usr/bin/env python3
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

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.utils.logging import get_logger

from ..so101_env import SO101Env, SO101RobotConfig


@dataclass
class SO101IKReachCheckConfig(SO101RobotConfig):
    """Minimal IK reach validation config for real robot checks."""

    # [x, y, z, rx, ry, rz], where orientation is Euler xyz in radians.
    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    success_position_error_m: float = 0.02
    success_orientation_error_deg: float = 30.0
    reset_position: np.ndarray = field(default_factory=lambda: np.zeros(5))
    check_orientation: bool = True

    def __post_init__(self):
        self.target_ee_pose = np.asarray(self.target_ee_pose, dtype=float).reshape(6)
        self.reset_position = np.asarray(self.reset_position, dtype=float).reshape(5)
        self.joint_reset_qpos = self.reset_position.tolist()


class SO101IKReachCheckEnv(SO101Env):
    _DEBUG_LOG_PATH = Path("/home/zhihao/SO101_arm/.cursor/debug-c6202f.log")
    _DEBUG_SESSION_ID = "c6202f"
    _STEP_DELTA_GUARD_BAND_DEG = 0.5

    def __init__(
        self,
        override_cfg,
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    ):
        self._logger = get_logger()
        self._hold_current_streak = 0
        self._prev_move_readback_joint: np.ndarray | None = None
        self._readback_stagnation_streak = 0
        config = SO101IKReachCheckConfig(**dict(override_cfg))
        if config.is_dummy:
            raise ValueError("SO101IKReachCheckEnv only supports real robot mode (is_dummy=False).")
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "so101 ik reach check"

    def _target_pose7(self) -> np.ndarray:
        target = np.zeros(7, dtype=float)
        target[:3] = self.config.target_ee_pose[:3]
        target[3:] = R.from_euler("xyz", self.config.target_ee_pose[3:]).as_quat()
        return target

    def _orientation_error_deg(self, achieved_quat: np.ndarray, target_quat: np.ndarray) -> float:
        rel = R.from_quat(achieved_quat.copy()) * R.from_quat(target_quat.copy()).inv()
        return float(np.linalg.norm(rel.as_rotvec()) * (180.0 / np.pi))

    def _append_debug_log(
        self,
        run_id: str,
        hypothesis_id: str,
        location: str,
        message: str,
        data: dict[str, Any],
    ) -> None:
        payload = {
            "sessionId": self._DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        try:
            with self._DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def step(self, action: np.ndarray):
        del action  # This task always moves to a fixed absolute target pose.
        start_time = time.time()
        run_id = f"ik_reach_ep{self._num_steps}"

        assert self._controller is not None
        self._refresh_state_from_controller()

        desired_pose7 = self._target_pose7()
        unclipped_desired_pose = desired_pose7.copy()
        desired_pose7 = self._clip_pose_to_limits(desired_pose7)

        q_curr = self._state.arm_joint_position.astype(float)
        current_pos_error_m = float(np.linalg.norm(self._state.tcp_pose[:3] - desired_pose7[:3]))
        # region agent log
        self._append_debug_log(
            run_id=run_id,
            hypothesis_id="H1",
            location="ik_reach_check_env.py:step:pre_ik",
            message="Pre-IK target and current joint state.",
            data={
                "q_curr_deg": q_curr.tolist(),
                "current_pos_error_m": float(current_pos_error_m),
                "target_pose_xyz": desired_pose7[:3].tolist(),
                "target_pose_quat": desired_pose7[3:].tolist(),
                "target_pose_clipped": bool(
                    np.any(np.abs(desired_pose7 - unclipped_desired_pose) > 1e-9)
                ),
            },
        )
        # endregion
        q_target_raw, ik_diag = self._solve_ik_with_fallback(q_curr, desired_pose7)
        ik_reseed_used = False
        if (not bool(ik_diag["ik_converged"])) and str(ik_diag["ik_fallback_level"]) == "hold_current":
            self._hold_current_streak += 1
        else:
            self._hold_current_streak = 0

        if str(ik_diag["ik_fallback_level"]) == "hold_current":
            relaxed_probe_converged = False
            relaxed_probe_pos_error_m = np.nan
            relaxed_probe_rot_error_deg = np.nan
            if self._kin is not None:
                try:
                    relaxed_probe = self._kin.inverse_kinematics_with_result(
                        current_joint_pos_deg=q_curr,
                        desired_ee_pose=self._pose7_to_matrix(desired_pose7),
                        position_weight=float(self.config.ik_position_weight),
                        orientation_weight=float(self.config.ik_position_only_orientation_weight),
                        position_error_threshold_m=max(
                            float(self.config.ik_position_error_threshold_m), 0.08
                        ),
                        orientation_error_threshold_deg=float(self.config.ik_orientation_error_threshold_deg),
                        check_orientation=False,
                    )
                    relaxed_probe_converged = bool(relaxed_probe.converged)
                    relaxed_probe_pos_error_m = float(relaxed_probe.position_error_m)
                    relaxed_probe_rot_error_deg = float(relaxed_probe.orientation_error_deg)
                except Exception:
                    pass
            # region agent log
            self._append_debug_log(
                run_id=run_id,
                hypothesis_id="H7",
                location="ik_reach_check_env.py:step:hold_current_probe",
                message="Probe hold_current root cause with relaxed IK threshold.",
                data={
                    "hold_current_streak": int(self._hold_current_streak),
                    "ik_position_error_threshold_m": float(self.config.ik_position_error_threshold_m),
                    "task_success_position_error_m": float(self.config.success_position_error_m),
                    "relaxed_probe_threshold_m": float(
                        max(float(self.config.ik_position_error_threshold_m), 0.08)
                    ),
                    "relaxed_probe_converged": bool(relaxed_probe_converged),
                    "relaxed_probe_position_error_m": float(relaxed_probe_pos_error_m),
                    "relaxed_probe_orientation_error_deg": float(relaxed_probe_rot_error_deg),
                    "q_target_equals_q_curr": bool(
                        np.max(np.abs(np.asarray(q_target_raw, dtype=float) - q_curr)) <= 1e-9
                    ),
                },
            )
            # endregion

        if self._hold_current_streak >= 2:
            q_seed = np.asarray(self.config.reset_position, dtype=float).reshape(5)
            q_target_alt_raw, ik_alt_diag = self._solve_ik_with_fallback(q_seed, desired_pose7)
            # region agent log
            self._append_debug_log(
                run_id=run_id,
                hypothesis_id="H6",
                location="ik_reach_check_env.py:step:ik_reseed",
                message="IK reseed with reset joint pose after repeated hold_current.",
                data={
                    "hold_current_streak": int(self._hold_current_streak),
                    "q_seed_deg": q_seed.tolist(),
                    "ik_alt_fallback_level": str(ik_alt_diag["ik_fallback_level"]),
                    "ik_alt_converged": bool(ik_alt_diag["ik_converged"]),
                    "ik_alt_position_error_m": float(ik_alt_diag["ik_position_error_m"]),
                    "ik_alt_orientation_error_deg": float(ik_alt_diag["ik_orientation_error_deg"]),
                    "ik_alt_exception": bool(ik_alt_diag["ik_exception"]),
                    "max_abs_alt_delta_vs_curr_deg": float(
                        np.max(np.abs(np.asarray(q_target_alt_raw, dtype=float) - q_curr))
                    ),
                },
            )
            # endregion
            if bool(ik_alt_diag["ik_converged"]):
                q_target_raw = q_target_alt_raw
                ik_diag = ik_alt_diag
                ik_reseed_used = True
                self._hold_current_streak = 0

        # region agent log
        self._append_debug_log(
            run_id=run_id,
            hypothesis_id="H2",
            location="ik_reach_check_env.py:step:post_ik",
            message="IK output and diagnostics before joint-step limiting.",
            data={
                "q_target_raw_deg": np.asarray(q_target_raw, dtype=float).tolist(),
                "max_abs_raw_delta_deg": float(
                    np.max(np.abs(np.asarray(q_target_raw, dtype=float) - q_curr))
                ),
                "ik_fallback_level": str(ik_diag["ik_fallback_level"]),
                "ik_converged": bool(ik_diag["ik_converged"]),
                "hold_current_streak": int(self._hold_current_streak),
                "ik_reseed_used": bool(ik_reseed_used),
                "ik_retry_used": bool(ik_diag.get("ik_retry_used", False)),
                "ik_retry_attempt": int(ik_diag.get("ik_retry_attempt", 0)),
                "ik_position_error_m": float(ik_diag["ik_position_error_m"]),
                "ik_orientation_error_deg": float(ik_diag["ik_orientation_error_deg"]),
                "ik_exception": bool(ik_diag["ik_exception"]),
            },
        )
        # endregion
        applied_step_limit_deg = max(
            0.1,
            float(self.config.max_joint_step_delta_deg) - float(self._STEP_DELTA_GUARD_BAND_DEG),
        )
        # Reduce step near target to prevent branch-flip oscillation around the goal.
        if current_pos_error_m <= 0.05:
            applied_step_limit_deg = min(applied_step_limit_deg, 1.0)
        elif current_pos_error_m <= 0.08:
            applied_step_limit_deg = min(applied_step_limit_deg, 2.0)
        elif current_pos_error_m <= 0.12:
            applied_step_limit_deg = min(applied_step_limit_deg, 4.0)
        q_target, clipped_by_joint_step = self._limit_joint_step(
            q_curr, q_target_raw, max_step=applied_step_limit_deg
        )
        self._logger.info(
            "IK reach step debug: q_curr=%s q_target_raw=%s q_target_limited=%s ik_fallback=%s "
            "ik_pos_err_m=%.6f ik_rot_err_deg=%.6f joint_step_clipped=%s "
            "applied_step_limit_deg=%.4f configured_step_limit_deg=%.4f max_abs_limited_delta_deg=%.6f",
            np.array2string(q_curr, precision=4),
            np.array2string(np.asarray(q_target_raw, dtype=float), precision=4),
            np.array2string(np.asarray(q_target, dtype=float), precision=4),
            str(ik_diag["ik_fallback_level"]),
            float(ik_diag["ik_position_error_m"]),
            float(ik_diag["ik_orientation_error_deg"]),
            bool(clipped_by_joint_step),
            float(applied_step_limit_deg),
            float(self.config.max_joint_step_delta_deg),
            float(np.max(np.abs(np.asarray(q_target, dtype=float) - q_curr))),
        )
        # region agent log
        self._append_debug_log(
            run_id=run_id,
            hypothesis_id="H3",
            location="ik_reach_check_env.py:step:post_limit",
            message="Joint-step limiter output and command delta.",
            data={
                "q_target_limited_deg": np.asarray(q_target, dtype=float).tolist(),
                "max_abs_limited_delta_deg": float(
                    np.max(np.abs(np.asarray(q_target, dtype=float) - q_curr))
                ),
                "joint_step_clipped": bool(clipped_by_joint_step),
                "applied_limit_deg": float(applied_step_limit_deg),
                "configured_limit_deg": float(self.config.max_joint_step_delta_deg),
                "guard_band_deg": float(self._STEP_DELTA_GUARD_BAND_DEG),
            },
        )
        # endregion

        move_success = False
        move_timed_out = False
        move_max_joint_error_deg = np.nan
        try:
            move_timeout = max(float(self.config.step_timeout_s), 1.0 / float(self.config.step_frequency))
            move_res = self._controller.move_joints_blocking(
                q_target,
                gripper_pos=None,
                timeout_s=move_timeout,
                joint_tolerance_deg=float(self.config.joint_tolerance_deg),
                max_step_delta_deg=float(self.config.max_joint_step_delta_deg),
                raise_on_fail=False,
            ).wait()[0]
            move_success = bool(move_res.success)
            move_timed_out = bool(move_res.timed_out)
            move_max_joint_error_deg = float(move_res.max_joint_error_deg)
            # region agent log
            self._append_debug_log(
                run_id=run_id,
                hypothesis_id="H4",
                location="ik_reach_check_env.py:step:move_result",
                message="Controller move result and readback joint state.",
                data={
                    "move_success": bool(move_success),
                    "move_timed_out": bool(move_timed_out),
                    "move_max_joint_error_deg": float(move_max_joint_error_deg),
                    "readback_joint_deg": np.asarray(move_res.readback_joint_deg, dtype=float).tolist(),
                },
            )
            # endregion
            readback_joint = np.asarray(move_res.readback_joint_deg, dtype=float)
            if self._prev_move_readback_joint is None:
                self._readback_stagnation_streak = 0
                readback_delta_max = np.nan
            else:
                readback_delta_max = float(
                    np.max(np.abs(readback_joint - np.asarray(self._prev_move_readback_joint, dtype=float)))
                )
                if readback_delta_max <= 1e-9:
                    self._readback_stagnation_streak += 1
                else:
                    self._readback_stagnation_streak = 0
            self._prev_move_readback_joint = readback_joint.copy()
            # region agent log
            self._append_debug_log(
                run_id=run_id,
                hypothesis_id="H8",
                location="ik_reach_check_env.py:step:stagnation_probe",
                message="Consecutive readback stagnation after move command.",
                data={
                    "readback_delta_max_deg": float(readback_delta_max),
                    "readback_stagnation_streak": int(self._readback_stagnation_streak),
                    "q_target_equals_q_curr": bool(
                        np.max(np.abs(np.asarray(q_target, dtype=float) - q_curr)) <= 1e-9
                    ),
                    "ik_fallback_level": str(ik_diag["ik_fallback_level"]),
                },
            )
            # endregion
        except Exception as e:
            self._logger.warning(f"SO101 IK reach check move failed (non-fatal): {e}")
            # region agent log
            self._append_debug_log(
                run_id=run_id,
                hypothesis_id="H4",
                location="ik_reach_check_env.py:step:move_exception",
                message="Controller raised an exception during move.",
                data={
                    "exception_type": type(e).__name__,
                    "exception_text": str(e),
                    "max_abs_limited_delta_deg": float(
                        np.max(np.abs(np.asarray(q_target, dtype=float) - q_curr))
                    ),
                    "limit_deg": float(self.config.max_joint_step_delta_deg),
                },
            )
            # endregion

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        obs = self._get_observation()
        achieved_pose = obs["state"]["tcp_pose"]
        pred_pos_error_m = np.nan
        obs_vs_fk_xyz_error_m = np.nan
        if self._kin is not None:
            try:
                fk_T = self._kin.forward_kinematics(np.asarray(q_target, dtype=float))
                fk_xyz = np.asarray(fk_T[:3, 3], dtype=float)
                pred_pos_error_m = float(np.linalg.norm(fk_xyz - desired_pose7[:3]))
                obs_vs_fk_xyz_error_m = float(np.linalg.norm(achieved_pose[:3] - fk_xyz))
            except Exception:
                pass
        pos_error_m = float(np.linalg.norm(achieved_pose[:3] - desired_pose7[:3]))
        if self.config.check_orientation:
            rot_error_deg = self._orientation_error_deg(achieved_pose[3:], desired_pose7[3:])
        else:
            rot_error_deg = 0.0
        # region agent log
        self._append_debug_log(
            run_id=run_id,
            hypothesis_id="H5",
            location="ik_reach_check_env.py:step:post_obs",
            message="Post-move Cartesian error against desired pose.",
            data={
                "achieved_pose_xyz": achieved_pose[:3].tolist(),
                "desired_pose_xyz": desired_pose7[:3].tolist(),
                "fk_pred_pos_error_m": float(pred_pos_error_m),
                "obs_vs_fk_xyz_error_m": float(obs_vs_fk_xyz_error_m),
                "task_pos_error_m": float(pos_error_m),
                "task_rot_error_deg": float(rot_error_deg),
                "task_success_candidate": bool(
                    pos_error_m <= float(self.config.success_position_error_m)
                ),
            },
        )
        # endregion
        success = bool(pos_error_m <= float(self.config.success_position_error_m)) and bool(
            (not self.config.check_orientation)
            or (rot_error_deg <= float(self.config.success_orientation_error_deg))
        )

        terminated = success
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "task_success": success,
            "task_pos_error_m": pos_error_m,
            "task_rot_error_deg": rot_error_deg,
            "task_check_orientation": bool(self.config.check_orientation),
            "target_pose_clipped_to_limits": bool(
                np.any(np.abs(desired_pose7 - unclipped_desired_pose) > 1e-9)
            ),
            "joint_step_clipped": clipped_by_joint_step,
            "move_success": move_success,
            "move_timed_out": move_timed_out,
            "move_max_joint_error_deg": move_max_joint_error_deg,
            "ik_fallback_level": ik_diag["ik_fallback_level"],
            "ik_converged": ik_diag["ik_converged"],
            "ik_reseed_used": ik_reseed_used,
            "ik_retry_used": bool(ik_diag.get("ik_retry_used", False)),
            "ik_retry_attempt": int(ik_diag.get("ik_retry_attempt", 0)),
            "hold_current_streak": int(self._hold_current_streak),
            "ik_position_error_m": ik_diag["ik_position_error_m"],
            "ik_orientation_error_deg": ik_diag["ik_orientation_error_deg"],
            "ik_exception": ik_diag["ik_exception"],
        }
        reward = 1.0 if success else 0.0
        return obs, reward, terminated, truncated, info
