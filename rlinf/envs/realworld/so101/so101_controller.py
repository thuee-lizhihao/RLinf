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
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .so101_robot_state import SO101RobotState

# region agent log
def _dlog(
    hypothesisId: str,
    location: str,
    message: str,
    data: dict | None = None,
    runId: str = "pre-fix",
):
    """Write NDJSON logs into <repo_root>/.cursor/debug.log for debug-mode evidence."""
    import json
    import time as _time

    try:
        # file: <repo_root>/RLinf/rlinf/envs/realworld/so101/so101_controller.py
        repo_root = Path(__file__).resolve().parents[6]
        log_path = repo_root / ".cursor" / "debug.log"
        os.makedirs(str(log_path.parent), exist_ok=True)
        payload = {
            "sessionId": "debug-session",
            "runId": runId,
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(_time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            rp = str(p.expanduser().resolve())
        except Exception:
            rp = str(p)
        if rp in seen:
            continue
        seen.add(rp)
        if Path(rp).is_file():
            out.append(Path(rp))
    return out


def _discover_so101_calibration_files(
    *,
    robot_type: str,
    robot_id: str | None,
) -> list[Path]:
    """Best-effort discovery of LeRobot calibration JSON files.

    We intentionally keep the search bounded and deterministic (no full-disk scans).
    """
    home = Path.home()
    bases: list[Path] = [
        # LeRobot cache (common)
        home / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / robot_type,
        # Fallback variants seen in the wild
        home / ".cache" / "huggingface" / "lerobot" / "calibrations" / "robots" / robot_type,
        home / ".cache" / "lerobot" / "calibration" / "robots" / robot_type,
        # Project-local (optional convention)
        Path.cwd() / "calibration",
        Path.cwd() / "calibrations",
        Path.cwd(),
    ]

    candidates: list[Path] = []
    if robot_id:
        for b in bases:
            candidates.append(b / f"{robot_id}.json")

    for b in bases:
        try:
            if b.is_dir():
                candidates.extend(sorted(b.glob("*.json")))
        except Exception:
            # Ignore permission / transient FS errors
            pass

    return _unique_existing_paths(candidates)


def _default_lerobot_calibration_path(*, robot_type: str, robot_id: str) -> Path:
    # Matches the common LeRobot cache convention.
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "calibration"
        / "robots"
        / robot_type
        / f"{robot_id}.json"
    )


def _run_lerobot_calibrate(*, robot_type: str, port: str, robot_id: str) -> None:
    """Run LeRobot calibration (requires `lerobot-calibrate` on PATH).

    We intentionally avoid "source tree" fallbacks (e.g. python -m + PYTHONPATH hacks),
    because they introduce ambiguity about which LeRobot version is being used.
    """
    if not shutil.which("lerobot-calibrate"):
        raise FileNotFoundError(
            "SO101 auto calibration requested, but `lerobot-calibrate` was not found on PATH.\n"
            "Please install LeRobot so its console scripts are available, e.g.:\n"
            "  pip install -e \"/path/to/lerobot[feetech]\"\n"
            "Then rerun, or run `lerobot-calibrate` manually and set SO101_CALIBRATION_PATH."
        )
    cmd = [
        "lerobot-calibrate",
        f"--robot.type={robot_type}",
        f"--robot.port={port}",
        f"--robot.id={robot_id}",
    ]
    subprocess.run(cmd, check=True)


@dataclass
class SO101JointCommand:
    """Joint-space command for SO101."""

    # degrees for 5 arm joints
    arm_joint_pos_deg: np.ndarray
    # [0, 100] for gripper
    gripper_pos: float


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
    ):
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return SO101Controller.create_group(port, calibration_path).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"SO101Controller-{worker_rank}-{env_idx}",
        )

    def __init__(self, port: str, calibration_path: Optional[str] = None):
        super().__init__()
        self._logger = get_logger()
        self._port = port
        self._calibration_path = calibration_path
        self._state = SO101RobotState()

        self._bus = None
        self._motor_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

    def connect(self):
        """Connect to the motor bus, ensure calibration, and configure motors.

        Calibration policy (default: calibration is REQUIRED):
        - If `calibration_path` was provided at construction, it must exist and be loadable.
        - Otherwise, we try to discover an existing calibration JSON from common LeRobot cache locations.
        - If none exists and `SO101_AUTO_CALIBRATE=1`, we invoke `lerobot-calibrate` to generate one.

        Environment variables:
        - `SO101_REQUIRE_CALIBRATION` (default: 1): enforce calibration JSON usage.
        - `SO101_ID` (default: "so101"): used to name/discover LeRobot calibration file.
        - `SO101_ROBOT_TYPE` (default: "so101_follower"): used to form LeRobot cache path.
        - `SO101_CALIBRATION_AUTO_SELECT` (optional): when multiple calibration files are found,
          set to "first" or an integer index (0-based) to auto-pick one; otherwise we raise.
        - `SO101_AUTO_CALIBRATE` (default: 0): if set, run `lerobot-calibrate ...` when missing.
        """
        if self._bus is not None:
            return

        # region agent log
        import sys as _sys
        import importlib.util as _iu

        _dlog(
            "H1",
            "so101_controller.py:SO101Controller.connect:pre_import",
            "About to import lerobot.* in worker",
            {
                "python": _sys.executable,
                "cwd": os.getcwd(),
                "sys_path0": _sys.path[0] if len(_sys.path) > 0 else None,
                "lerobot_spec": str(_iu.find_spec("lerobot")),
                "lerobot_calibrate_in_PATH": bool(shutil.which("lerobot-calibrate")),
            },
        )
        # endregion

        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
        except ModuleNotFoundError as e:
            # region agent log
            _dlog(
                "H2",
                "so101_controller.py:SO101Controller.connect:import_failed",
                "Failed importing lerobot.*",
                {"error": repr(e)},
            )
            # endregion
            raise ModuleNotFoundError(
                "SO101Controller requires LeRobot Feetech support. "
                "Install LeRobot with feetech extras (and scservo_sdk), or add LeRobot to PYTHONPATH. "
                "For example: `pip install -e \"/path/to/lerobot[feetech]\"`."
            ) from e

        require_calibration = _env_flag("SO101_REQUIRE_CALIBRATION", True)
        robot_type = os.environ.get("SO101_ROBOT_TYPE", "so101_follower")
        robot_id = os.environ.get("SO101_ID", "so101")

        if require_calibration and not self._calibration_path:
            found = _discover_so101_calibration_files(robot_type=robot_type, robot_id=robot_id)
            if len(found) == 1:
                self._calibration_path = str(found[0])
                self._logger.info(f"Using discovered SO101 calibration: {self._calibration_path}")
            elif len(found) > 1:
                auto_select = os.environ.get("SO101_CALIBRATION_AUTO_SELECT", "")
                picked: Path | None = None
                if auto_select.strip().lower() == "first":
                    picked = sorted(found)[0]
                else:
                    try:
                        idx = int(auto_select)
                        if 0 <= idx < len(found):
                            picked = found[idx]
                    except Exception:
                        picked = None
                if picked is not None:
                    self._calibration_path = str(picked)
                    self._logger.warning(
                        "Multiple SO101 calibration files found; "
                        f"auto-selected {self._calibration_path} via SO101_CALIBRATION_AUTO_SELECT={auto_select!r}."
                    )
                else:
                    msg = "\n".join([f"- {p}" for p in found])
                    raise RuntimeError(
                        "SO101 calibration is required but multiple calibration files were discovered.\n"
                        "Please set SO101_CALIBRATION_PATH explicitly, or set SO101_CALIBRATION_AUTO_SELECT "
                        "to 'first' or an integer index (0-based).\n"
                        f"Discovered files:\n{msg}"
                    )
            else:
                if _env_flag("SO101_AUTO_CALIBRATE", False):
                    # Try to generate a calibration file via LeRobot CLI.
                    # This is interactive; ensure you run this with a TTY.
                    expected = _default_lerobot_calibration_path(
                        robot_type=robot_type, robot_id=robot_id
                    )
                    expected.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        "lerobot-calibrate",
                        f"--robot.type={robot_type}",
                        f"--robot.port={self._port}",
                        f"--robot.id={robot_id}",
                    ]
                    self._logger.warning(
                        "SO101 calibration file not found; running LeRobot calibration:\n"
                        f"Expected output file: {expected}"
                    )
                    _run_lerobot_calibrate(robot_type=robot_type, port=self._port, robot_id=robot_id)
                    if expected.is_file():
                        self._calibration_path = str(expected)
                    else:
                        # Re-scan in case LeRobot wrote elsewhere.
                        found2 = _discover_so101_calibration_files(
                            robot_type=robot_type, robot_id=robot_id
                        )
                        if len(found2) >= 1:
                            self._calibration_path = str(sorted(found2)[0])
                        else:
                            raise RuntimeError(
                                "LeRobot calibration finished but no calibration JSON was found.\n"
                                f"Tried expected path: {expected}\n"
                                "Please locate the generated file and set SO101_CALIBRATION_PATH."
                            )
                else:
                    expected = _default_lerobot_calibration_path(
                        robot_type=robot_type, robot_id=robot_id
                    )
                    raise RuntimeError(
                        "SO101 calibration is required but no calibration file was provided or discovered.\n"
                        "Set SO101_CALIBRATION_PATH to an existing calibration JSON, or run:\n"
                        f"  lerobot-calibrate --robot.type={robot_type} --robot.port={self._port} --robot.id={robot_id}\n"
                        f"Common expected location: {expected}\n"
                        "If you have multiple robots, make sure SO101_ID matches the one you calibrated."
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

        self._bus.sync_write("Goal_Position", goal)

    def open_gripper(self, pos: float = 100.0):
        self.command_joints(self._state.arm_joint_position, gripper_pos=pos)
        self._state.gripper_open = True

    def close_gripper(self, pos: float = 0.0):
        self.command_joints(self._state.arm_joint_position, gripper_pos=pos)
        self._state.gripper_open = False

    def get_state(self) -> SO101RobotState:
        """Return last cached state. Env is responsible for updating kinematics."""
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

    def wait(self, seconds: float):
        time.sleep(seconds)

