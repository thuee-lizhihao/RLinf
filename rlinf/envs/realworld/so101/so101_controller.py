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

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .so101_robot_state import SO101RobotState


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass
class SO101JointCommand:
    """Joint-space command for SO101."""

    # degrees for 5 arm joints
    arm_joint_pos_deg: np.ndarray
    # [0, 100] for gripper
    gripper_pos: float


@dataclass
class SO101ControllerRuntimeConfig:
    """Runtime parameters for SO101 high-level motion APIs."""

    poll_period_s: float = 0.05
    default_timeout_s: float = 3.0
    default_joint_tolerance_deg: float = 2.0
    default_max_step_delta_deg: float = 5.0
    settle_time_s: float = 0.05
    min_state_dt: float = 1e-3
    vel_smoothing_alpha: float = 0.2
    tcp_linear_vel_max: float = 1.0
    tcp_angular_vel_max: float = 10.0


@dataclass
class SO101MoveResult:
    """Result of a blocking joint-space motion."""

    success: bool
    timed_out: bool
    duration_s: float
    num_polls: int
    max_joint_error_deg: float
    mean_joint_error_deg: float
    ee_pos_error_m: float
    ee_rot_error_deg: float
    target_joint_deg: np.ndarray
    readback_joint_deg: np.ndarray


class SO101Controller(Worker):
    """SO101 robot arm controller (serial Feetech).

    Notes:
    - The actual motor bus driver is provided by LeRobot's feetech implementation.
      We import it lazily so that dummy runs (or nodes without hardware) can still import RLinf.
    - This controller runs as a Worker so the env can launch it on the designated control node.
    """

    @staticmethod
    def launch_controller(
        port: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        calibration_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        target_frame_name: str = "gripper_frame_link",
    ):
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return SO101Controller.create_group(
            port, calibration_path, urdf_path, target_frame_name
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"SO101Controller-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        port: str,
        calibration_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        target_frame_name: str = "gripper_frame_link",
    ):
        super().__init__()
        self._logger = get_logger()
        self._port = port
        self._calibration_path = calibration_path
        self._state = SO101RobotState()
        self._runtime_cfg = SO101ControllerRuntimeConfig()

        self._bus = None
        self._joint_limit_cache_deg: dict[str, tuple[float, float]] | None = None
        self._kin = None
        self._urdf_path = urdf_path
        self._target_frame_name = target_frame_name
        self._last_state_timestamp = None
        self._last_joint_pos_deg = None
        self._last_tcp_pose = None
        self._last_tcp_vel = None
        self._motor_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

    @staticmethod
    def _default_urdf_path() -> str:
        return str(Path(__file__).with_suffix("").parent / "assets" / "so101_minimal.urdf")

    def connect(self):
        """Connect to the motor bus, ensure calibration, and configure motors.

        Calibration is required by default: you must pass a calibration JSON path at
        construction (or set SO101_CALIBRATION_PATH and pass it when creating the
        controller). If no path is provided, connect() raises with a message that
        suggests possible paths and how to run the official calibration tool.

        Environment variables:
        - `SO101_REQUIRE_CALIBRATION` (default: 1): if 0, allow running without calibration.
        """
        if self._bus is not None:
            return

        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "SO101Controller requires LeRobot Feetech support. "
                "Install LeRobot with feetech extras (and scservo_sdk), or add LeRobot to PYTHONPATH. "
                "For example: `pip install -e \"/path/to/lerobot[feetech]\"`."
            ) from e

        require_calibration = _env_flag("SO101_REQUIRE_CALIBRATION", True)
        if require_calibration and not self._calibration_path:
            robot_type = os.environ.get("SO101_ROBOT_TYPE", "so101_follower")
            robot_id = os.environ.get("SO101_ID", "my_awesome_follower_arm")
            home = Path.home()
            common = home / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / robot_type / f"{robot_id}.json"
            raise RuntimeError(
                "SO101 calibration path is required but was not provided.\n"
                "Pass calibration_path when creating the controller, or set SO101_CALIBRATION_PATH and pass it to the constructor.\n\n"
                "Possible locations to check:\n"
                f"  - {common}\n"
                "  - ./calibration/\n"
                "  - ./calibrations/\n\n"
                "To generate a calibration file, run the official LeRobot calibration tool (with the robot connected):\n"
                f"  lerobot-calibrate --robot.type={robot_type} --robot.port={self._port} --robot.id={robot_id}\n\n"
                "Then set SO101_CALIBRATION_PATH to the output JSON path, or pass it as calibration_path."
            )

        motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
        calibration = None
        if self._calibration_path:
            try:
                import draccus
                from lerobot.motors import MotorCalibration
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "SO101 calibration file requires draccus and lerobot.motors. "
                    "Install LeRobot (and draccus) to use calibration_path."
                )
            with open(self._calibration_path) as f, draccus.config_type("json"):
                calibration = draccus.load(dict[str, MotorCalibration], f)

        self._bus = FeetechMotorsBus(
            port=self._port, motors=motors, calibration=calibration
        )
        self._bus.connect()

        # Basic configuration
        self._bus.disable_torque()
        self._bus.configure_motors()
        for m in motors:
            self._bus.write("Operating_Mode", m, OperatingMode.POSITION.value)

        # Soft-start safety: avoid a "jump" when enabling torque.
        # If motors still have a stale Goal_Position from a previous session, turning torque on can cause
        # the arm to snap toward that target. To prevent this, we latch the current raw Present_Position
        # into Goal_Position *before* enabling torque.
        #
        # Use raw units (normalize=False) to avoid relying on calibration at this stage.
        try:
            present_raw = self._bus.sync_read(
                "Present_Position", motors=list(motors.keys()), normalize=False
            )
            self._bus.sync_write("Goal_Position", present_raw, normalize=False)
            # Small settle time so the bus clears writes before torque enable.
            time.sleep(0.05)
        except Exception as e:
            self._logger.warning(
                f"SO101 soft-start latch (Present_Position→Goal_Position) failed: {e}. "
                "Torque enable may cause a transient motion if stale goals exist."
            )

        self._bus.enable_torque()

        # If calibration is required, we should never silently proceed without it.
        # Keep this "read from motors" as a best-effort fallback, but fail hard if it isn't available.
        if calibration is None:
            try:
                self._bus.calibration = self._bus.read_calibration()
                if require_calibration and not self._bus.calibration:
                    raise RuntimeError("Empty calibration read from motors.")
            except Exception as e:
                if require_calibration:
                    raise RuntimeError(
                        "SO101 calibration is required but could not be loaded from file "
                        "and could not be read from motors.\n"
                        f"Port: {self._port}\n"
                        f"calibration_path: {self._calibration_path}\n"
                        f"Error: {e}\n"
                        "Run `lerobot-calibrate` to generate a calibration JSON and set SO101_CALIBRATION_PATH."
                    ) from e
                self._logger.warning(
                    f"Failed to read calibration from motors on {self._port}: {e}. "
                    "Joint positions will be read without normalization."
                )

        # Cache joint limits in user units (degrees / 0-100) for safety checks on command.
        self._joint_limit_cache_deg = None
        self._init_kinematics()

    def _init_kinematics(self) -> None:
        if self._kin is not None:
            return
        urdf_path = self._urdf_path or self._default_urdf_path()
        try:
            from .kinematics import SO101Kinematics

            self._kin = SO101Kinematics(
                urdf_path=urdf_path,
                target_frame_name=self._target_frame_name,
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
            )
        except ModuleNotFoundError as e:
            self._logger.warning(
                f"SO101 kinematics is unavailable, TCP metrics disabled. Error: {e}"
            )
            self._kin = None

    def _enforce_joint_limits_enabled(self) -> bool:
        # Default to enabled for safety. Can be disabled for debugging.
        v = os.environ.get("SO101_ENFORCE_JOINT_LIMITS", "1").strip().lower()
        return v in {"1", "true", "t", "yes", "y", "on"}

    def _joint_limits_deg(self) -> dict[str, tuple[float, float]]:
        """Compute per-motor joint limits in *user units* (degrees for arm, [0,100] for gripper).

        For MotorNormMode.DEGREES, LeRobot uses the calibration [range_min, range_max] (raw) to define a
        symmetric degree range around mid=(min+max)/2, via:
          deg = (raw - mid) * 360 / (max_res)
        so the reachable bounds in degrees are approximately:
          [-((max-min)/2)*360/max_res, +((max-min)/2)*360/max_res]
        """
        assert self._bus is not None, "SO101 motor bus is not connected."
        if self._joint_limit_cache_deg is not None:
            return self._joint_limit_cache_deg

        if not getattr(self._bus, "calibration", None):
            raise RuntimeError(
                "Joint limit enforcement requested but no calibration is available on the motor bus. "
                "Provide a calibration JSON (SO101_CALIBRATION_PATH) or disable enforcement via "
                "SO101_ENFORCE_JOINT_LIMITS=0."
            )

        limits: dict[str, tuple[float, float]] = {}
        # Import types lazily; LeRobot isn't a hard dependency for dummy runs.
        from lerobot.motors.motors_bus import MotorNormMode  # type: ignore

        for name, motor in self._bus.motors.items():
            cal = self._bus.calibration.get(name)
            if cal is None:
                continue
            min_raw = float(cal.range_min)
            max_raw = float(cal.range_max)
            if motor.norm_mode is MotorNormMode.DEGREES:
                # symmetric bounds around 0 degrees
                max_res = float(self._bus.model_resolution_table[self._bus._id_to_model(motor.id)] - 1)  # type: ignore[attr-defined]
                span_raw = max_raw - min_raw
                max_abs_deg = (span_raw * 180.0) / max_res
                limits[name] = (-max_abs_deg, +max_abs_deg)
            elif motor.norm_mode is MotorNormMode.RANGE_0_100:
                limits[name] = (0.0, 100.0)
            elif motor.norm_mode is MotorNormMode.RANGE_M100_100:
                limits[name] = (-100.0, 100.0)
            else:
                # Unknown norm mode; don't enforce.
                continue

        self._joint_limit_cache_deg = limits
        return limits

    def _assert_arm_goal_within_limits(self, goal_deg: dict[str, float]) -> None:
        if not self._enforce_joint_limits_enabled():
            return
        limits = self._joint_limits_deg()
        margin_deg = float(os.environ.get("SO101_JOINT_LIMIT_MARGIN_DEG", "0.0"))

        offenders: list[str] = []
        for name, target in goal_deg.items():
            if name not in limits:
                continue
            lo, hi = limits[name]
            lo2, hi2 = lo + margin_deg, hi - margin_deg
            if not (lo2 <= float(target) <= hi2):
                offenders.append(
                    f"{name}: target={float(target):.3f}deg not in [{lo2:.3f}, {hi2:.3f}] (raw_limit=[{lo:.3f},{hi:.3f}], margin={margin_deg:.3f})"
                )
        if offenders:
            raise RuntimeError(
                "Safety stop: commanded joint target would exceed calibrated limits.\n"
                + "\n".join(offenders)
            )

    def get_joint_limits_deg(self) -> dict[str, tuple[float, float]]:
        """Expose calibrated joint limits (user units) for external tooling."""
        return self._joint_limits_deg()

    def set_torque_enabled(self, enabled: bool) -> None:
        """Enable/disable motor torque without disconnecting the bus."""
        assert self._bus is not None, "SO101 motor bus is not connected."
        if enabled:
            self._bus.enable_torque()
        else:
            self._bus.disable_torque()

    def disconnect(self, disable_torque: bool = True):
        if self._bus is None:
            return
        try:
            self._bus.disconnect(disable_torque)
        finally:
            self._bus = None

    def is_robot_up(self) -> bool:
        """Best-effort health check."""
        return self._bus is not None and self._bus.is_connected

    def get_joint_positions(self) -> dict[str, float]:
        """Read current joint positions.

        Returns:
            Dict[str, float]: {motor_name: pos} where pos is degrees for arm joints and [0,100] for gripper.
        """
        assert self._bus is not None, "SO101 motor bus is not connected."
        normalize = bool(self._bus.calibration)
        pos = self._bus.sync_read("Present_Position", normalize=normalize)
        return {k: float(pos[k]) for k in self._motor_names}

    @staticmethod
    def _joints_dict_to_q5(joints: dict[str, float]) -> np.ndarray:
        return np.array(
            [
                joints["shoulder_pan"],
                joints["shoulder_lift"],
                joints["elbow_flex"],
                joints["wrist_flex"],
                joints["wrist_roll"],
            ],
            dtype=float,
        )

    @staticmethod
    def _rot_error_deg_from_poses(q1: np.ndarray, q2: np.ndarray) -> float:
        return float(np.linalg.norm((R.from_quat(q2) * R.from_quat(q1).inv()).as_rotvec()) * 180.0 / np.pi)

    def _fk_pose(self, joint_pos_deg: np.ndarray) -> np.ndarray | None:
        if self._kin is None:
            return None
        T = self._kin.forward_kinematics(np.asarray(joint_pos_deg, dtype=float).reshape(5))
        pos = T[:3, 3]
        quat = R.from_matrix(T[:3, :3].copy()).as_quat()
        if self._last_tcp_pose is not None and np.dot(quat, self._last_tcp_pose[3:]) < 0:
            quat = -quat
        return np.concatenate([pos, quat], axis=0).astype(np.float32)

    def sync_state(self) -> SO101RobotState:
        """Refresh and return the latest structured robot state."""
        joints = self.get_joint_positions()
        now_ts = time.time()
        q_deg = self._joints_dict_to_q5(joints)
        gripper_pos = float(joints["gripper"])
        tcp_pose = self._fk_pose(q_deg)

        if self._last_state_timestamp is None:
            dt = self._runtime_cfg.min_state_dt
            arm_joint_vel = np.zeros(5, dtype=np.float32)
            tcp_vel = np.zeros(6, dtype=np.float32)
        else:
            dt = max(now_ts - self._last_state_timestamp, self._runtime_cfg.min_state_dt)
            if self._last_joint_pos_deg is not None:
                arm_joint_vel = ((q_deg - self._last_joint_pos_deg) / dt).astype(np.float32)
            else:
                arm_joint_vel = np.zeros(5, dtype=np.float32)

            if tcp_pose is None or self._last_tcp_pose is None:
                tcp_vel = np.zeros(6, dtype=np.float32)
            else:
                dp = (tcp_pose[:3] - self._last_tcp_pose[:3]) / dt
                drot = (
                    (R.from_quat(tcp_pose[3:]) * R.from_quat(self._last_tcp_pose[3:]).inv()).as_rotvec() / dt
                )
                raw_tcp_vel = np.concatenate([dp, drot], axis=0).astype(np.float32)

                linear_norm = np.linalg.norm(raw_tcp_vel[:3])
                if linear_norm > self._runtime_cfg.tcp_linear_vel_max and linear_norm > 0:
                    raw_tcp_vel[:3] = raw_tcp_vel[:3] * (
                        self._runtime_cfg.tcp_linear_vel_max / linear_norm
                    )

                angular_norm = np.linalg.norm(raw_tcp_vel[3:])
                if angular_norm > self._runtime_cfg.tcp_angular_vel_max and angular_norm > 0:
                    raw_tcp_vel[3:] = raw_tcp_vel[3:] * (
                        self._runtime_cfg.tcp_angular_vel_max / angular_norm
                    )

                alpha = float(np.clip(self._runtime_cfg.vel_smoothing_alpha, 0.0, 1.0))
                if alpha > 0.0 and self._last_tcp_vel is not None:
                    tcp_vel = (alpha * raw_tcp_vel + (1.0 - alpha) * self._last_tcp_vel).astype(np.float32)
                else:
                    tcp_vel = raw_tcp_vel

        self._state.arm_joint_position = q_deg.astype(np.float32)
        self._state.arm_joint_velocity = arm_joint_vel
        self._state.gripper_position = gripper_pos
        self._state.gripper_open = gripper_pos >= 50.0
        if tcp_pose is not None:
            self._state.tcp_pose = tcp_pose
            self._state.tcp_vel = tcp_vel
        self._state.timestamp_s = float(now_ts)
        self._state.is_robot_up = bool(self.is_robot_up())

        self._last_state_timestamp = now_ts
        self._last_joint_pos_deg = q_deg.copy()
        if tcp_pose is not None:
            self._last_tcp_pose = tcp_pose.copy()
            self._last_tcp_vel = tcp_vel.copy()
        return self._state

    def command_joints(
        self,
        arm_joint_pos_deg: np.ndarray,
        gripper_pos: Optional[float] = None,
    ):
        """Command target joint positions."""
        assert self._bus is not None, "SO101 motor bus is not connected."
        arm_joint_pos_deg = np.asarray(arm_joint_pos_deg, dtype=float).reshape(-1)
        assert arm_joint_pos_deg.shape[0] == 5, "SO101 expects 5 arm joints."

        goal = {
            "shoulder_pan": float(arm_joint_pos_deg[0]),
            "shoulder_lift": float(arm_joint_pos_deg[1]),
            "elbow_flex": float(arm_joint_pos_deg[2]),
            "wrist_flex": float(arm_joint_pos_deg[3]),
            "wrist_roll": float(arm_joint_pos_deg[4]),
        }
        if gripper_pos is not None:
            goal["gripper"] = float(np.clip(gripper_pos, 0.0, 100.0))

        # Safety: prevent commanding beyond calibrated joint limits (especially important for DEGREES mode,
        # where unnormalization does not clamp by default).
        self._assert_arm_goal_within_limits(goal)

        self._bus.sync_write("Goal_Position", goal)

    def move_joints_blocking(
        self,
        arm_joint_pos_deg: np.ndarray,
        gripper_pos: Optional[float] = None,
        timeout_s: Optional[float] = None,
        joint_tolerance_deg: Optional[float] = None,
        max_step_delta_deg: Optional[float] = None,
        raise_on_fail: bool = True,
    ) -> SO101MoveResult:
        """Command joints and block until tolerance or timeout."""
        q_target = np.asarray(arm_joint_pos_deg, dtype=float).reshape(5)
        timeout = float(timeout_s if timeout_s is not None else self._runtime_cfg.default_timeout_s)
        tolerance = float(
            joint_tolerance_deg
            if joint_tolerance_deg is not None
            else self._runtime_cfg.default_joint_tolerance_deg
        )
        max_step = float(
            max_step_delta_deg
            if max_step_delta_deg is not None
            else self._runtime_cfg.default_max_step_delta_deg
        )

        state0 = self.sync_state()
        q0 = state0.arm_joint_position.astype(float)
        if float(np.max(np.abs(q_target - q0))) > max_step + 1e-9:
            raise RuntimeError(
                f"Safety stop: max step delta {float(np.max(np.abs(q_target - q0))):.4f} exceeds {max_step:.4f} deg."
            )
        if not state0.is_robot_up:
            raise RuntimeError("Robot is not up / controller bus disconnected.")

        self.command_joints(q_target, gripper_pos=gripper_pos)
        if self._runtime_cfg.settle_time_s > 0:
            time.sleep(self._runtime_cfg.settle_time_s)

        start = time.time()
        num_polls = 0
        last_err = np.abs(q0 - q_target)
        last_q = q0.copy()
        success = False
        timed_out = False
        while True:
            s = self.sync_state()
            num_polls += 1
            q_rb = s.arm_joint_position.astype(float)
            last_q = q_rb.copy()
            last_err = np.abs(q_rb - q_target)
            if float(np.max(last_err)) <= tolerance + 1e-9:
                success = True
                break
            elapsed = time.time() - start
            if elapsed >= timeout:
                timed_out = True
                break
            time.sleep(self._runtime_cfg.poll_period_s)

        duration = float(time.time() - start)
        ee_pos_error_m = 0.0
        ee_rot_error_deg = 0.0
        if self._kin is not None:
            T_t = self._kin.forward_kinematics(q_target)
            T_r = self._kin.forward_kinematics(last_q)
            ee_pos_error_m = float(np.linalg.norm(T_t[:3, 3] - T_r[:3, 3]))
            ee_rot_error_deg = float(
                np.linalg.norm(
                    (
                        R.from_matrix(T_r[:3, :3].copy())
                        * R.from_matrix(T_t[:3, :3].copy()).inv()
                    ).as_rotvec()
                )
                * 180.0
                / np.pi
            )

        result = SO101MoveResult(
            success=success,
            timed_out=timed_out,
            duration_s=duration,
            num_polls=num_polls,
            max_joint_error_deg=float(np.max(last_err)),
            mean_joint_error_deg=float(np.mean(last_err)),
            ee_pos_error_m=ee_pos_error_m,
            ee_rot_error_deg=ee_rot_error_deg,
            target_joint_deg=q_target.astype(np.float32),
            readback_joint_deg=last_q.astype(np.float32),
        )
        self._state.last_joint_max_error_deg = result.max_joint_error_deg
        self._state.last_joint_mean_error_deg = result.mean_joint_error_deg
        self._state.last_ee_pos_error_m = result.ee_pos_error_m
        self._state.last_ee_rot_error_deg = result.ee_rot_error_deg

        if raise_on_fail and not result.success:
            raise RuntimeError(
                f"Blocking motion failed (timeout={result.timed_out}) with max joint error {result.max_joint_error_deg:.4f} deg."
            )
        return result

    def run_joint_sequence_blocking(
        self,
        arm_joint_targets_deg: list[np.ndarray],
        gripper_targets: Optional[list[Optional[float]]] = None,
        timeout_s: Optional[float] = None,
        joint_tolerance_deg: Optional[float] = None,
        max_step_delta_deg: Optional[float] = None,
    ) -> list[SO101MoveResult]:
        """Execute a joint sequence with per-step blocking checks."""
        results: list[SO101MoveResult] = []
        if gripper_targets is None:
            gripper_targets = [None] * len(arm_joint_targets_deg)
        if len(gripper_targets) != len(arm_joint_targets_deg):
            raise ValueError("gripper_targets length must match arm_joint_targets_deg length.")

        for q_target, g_target in zip(arm_joint_targets_deg, gripper_targets):
            r = self.move_joints_blocking(
                q_target,
                gripper_pos=g_target,
                timeout_s=timeout_s,
                joint_tolerance_deg=joint_tolerance_deg,
                max_step_delta_deg=max_step_delta_deg,
                raise_on_fail=True,
            )
            results.append(r)
        return results

    def recover_and_home(
        self,
        home_joint_pos_deg: np.ndarray,
        home_gripper_pos: Optional[float] = None,
        retries: int = 1,
        timeout_s: Optional[float] = None,
        joint_tolerance_deg: Optional[float] = None,
        max_step_delta_deg: Optional[float] = None,
    ) -> SO101MoveResult:
        """Best-effort return to home with bounded retries."""
        last_error = None
        for _ in range(max(1, retries + 1)):
            try:
                return self.move_joints_blocking(
                    home_joint_pos_deg,
                    gripper_pos=home_gripper_pos,
                    timeout_s=timeout_s,
                    joint_tolerance_deg=joint_tolerance_deg,
                    max_step_delta_deg=max_step_delta_deg,
                    raise_on_fail=True,
                )
            except Exception as e:
                last_error = e
                time.sleep(self._runtime_cfg.poll_period_s)
        raise RuntimeError(f"Failed to recover and return home: {last_error}")

    def open_gripper(self, pos: float = 100.0):
        """Open gripper without disturbing arm joints."""
        assert self._bus is not None, "SO101 motor bus is not connected."
        pos = float(np.clip(pos, 0.0, 100.0))
        self._bus.sync_write("Goal_Position", {"gripper": pos})
        self._state.gripper_open = True
        self._state.gripper_position = pos

    def close_gripper(self, pos: float = 0.0):
        """Close gripper without disturbing arm joints."""
        assert self._bus is not None, "SO101 motor bus is not connected."
        pos = float(np.clip(pos, 0.0, 100.0))
        self._bus.sync_write("Goal_Position", {"gripper": pos})
        self._state.gripper_open = False
        self._state.gripper_position = pos

    def get_state(self, refresh: bool = False) -> SO101RobotState:
        """Return structured robot state, optionally refreshed from hardware."""
        if refresh:
            return self.sync_state()
        return self._state

    def update_state_from_joints(
        self,
        arm_joint_pos_deg: np.ndarray,
        arm_joint_vel_deg: np.ndarray,
        gripper_pos: float,
        tcp_pose: np.ndarray,
        tcp_vel: np.ndarray,
    ):
        self._state.arm_joint_position = np.asarray(arm_joint_pos_deg, dtype=np.float32)
        self._state.arm_joint_velocity = np.asarray(arm_joint_vel_deg, dtype=np.float32)
        self._state.gripper_position = float(gripper_pos)
        self._state.tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
        self._state.tcp_vel = np.asarray(tcp_vel, dtype=np.float32)
        self._state.timestamp_s = float(time.time())
        self._state.is_robot_up = bool(self.is_robot_up())

    def wait(self, seconds: float):
        time.sleep(seconds)

