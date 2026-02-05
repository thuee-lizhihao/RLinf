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

"""SO101 run_sequence kinematics check.

This file focuses on: serial closed-loop (write->sleep->read) + kinematics error reporting.
Camera/controller setup is intentionally kept in `test_so101_setup.py`.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.so101.kinematics import SO101Kinematics

from test_so101_setup import connect_so101_controller


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO101: run a safe joint-space sequence with FK checks.")
    p.add_argument("--repeat", type=int, default=int(os.environ.get("SO101_REPEAT", "1")))
    p.add_argument("--step_sleep_s", type=float, default=float(os.environ.get("SO101_STEP_SLEEP_S", "1.0")))
    p.add_argument("--max_delta_deg", type=float, default=float(os.environ.get("SO101_MAX_DELTA_DEG", "3.0")))
    p.add_argument(
        "--error_threshold_deg",
        type=float,
        default=float(os.environ.get("SO101_ERROR_THRESHOLD_DEG", "5.0")),
    )
    p.add_argument(
        "--do_gripper",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SO101_DO_GRIPPER", "1").strip().lower() in {"1", "true", "yes", "y", "on"},
    )
    p.add_argument(
        "--urdf_path",
        type=str,
        default=os.environ.get("SO101_URDF_PATH", ""),
        help="Optional URDF path. If empty, use embedded minimal URDF.",
    )
    return p.parse_args()


def _joints_dict_to_q5(joints: dict) -> np.ndarray:
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


def _default_safe_joint_deltas(amp_deg: float) -> list[np.ndarray]:
    deltas: list[np.ndarray] = []
    for i in range(5):
        d = np.zeros(5, dtype=float)
        d[i] = amp_deg
        # +amp -> 0 -> -amp -> 0 (avoid a 2*amp jump between adjacent steps)
        deltas.append(d.copy())
        deltas.append(np.zeros(5, dtype=float))
        deltas.append(-d.copy())
        deltas.append(np.zeros(5, dtype=float))
    # small multi-joint patterns
    deltas.extend(
        [
            np.array([+0.5 * amp_deg, -0.5 * amp_deg, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, +0.5 * amp_deg, -0.5 * amp_deg, 0.0, 0.0], dtype=float),
            np.zeros(5, dtype=float),
        ]
    )
    return deltas


def _fk_pose_errors(kin: SO101Kinematics, q_target: np.ndarray, q_rb: np.ndarray) -> tuple[float, float]:
    """Return (pos_err_m, rot_err_deg)."""
    T_t = kin.forward_kinematics(np.asarray(q_target, dtype=float).reshape(5))
    T_r = kin.forward_kinematics(np.asarray(q_rb, dtype=float).reshape(5))
    pos_err = float(np.linalg.norm(T_r[:3, 3] - T_t[:3, 3]))
    R_t = R.from_matrix(T_t[:3, :3].copy())
    R_r = R.from_matrix(T_r[:3, :3].copy())
    rot_err_deg = float(np.linalg.norm((R_r * R_t.inv()).as_rotvec()) * 180.0 / np.pi)
    return pos_err, rot_err_deg


def run_sequence(
    *,
    controller,
    kin: SO101Kinematics,
    repeat: int,
    step_sleep_s: float,
    max_delta_deg: float,
    error_threshold_deg: float,
    do_gripper: bool,
) -> None:
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if step_sleep_s <= 0:
        raise ValueError(f"step_sleep_s must be > 0, got {step_sleep_s}")
    if max_delta_deg <= 0:
        raise ValueError(f"max_delta_deg must be > 0, got {max_delta_deg}")
    if error_threshold_deg <= 0:
        raise ValueError(f"error_threshold_deg must be > 0, got {error_threshold_deg}")

    joints0 = controller.get_joint_positions().wait()[0]
    q0 = _joints_dict_to_q5(joints0)
    g0 = float(joints0.get("gripper", 0.0))
    print(f"[run_sequence] initial q(deg)={q0.tolist()} gripper={g0:.3f}")

    amp = min(3.0, max_delta_deg)
    deltas = _default_safe_joint_deltas(amp_deg=amp)

    abs_err_all: list[np.ndarray] = []
    ee_pos_err_all: list[float] = []
    ee_rot_err_all: list[float] = []
    last_cmd_q = q0.copy()

    def _health_check() -> None:
        ok = bool(controller.is_robot_up().wait()[0])
        if not ok:
            raise RuntimeError("Robot is not up / controller bus disconnected.")

    def _command_and_check(q_target: np.ndarray, gripper_target: float | None) -> None:
        nonlocal last_cmd_q
        q_target = np.asarray(q_target, dtype=float).reshape(5)

        step_delta = q_target - last_cmd_q
        max_step = float(np.max(np.abs(step_delta)))
        if max_step > max_delta_deg + 1e-9:
            raise RuntimeError(
                f"Safety stop: per-step delta too large. max_step={max_step:.4f} > max_delta_deg={max_delta_deg:.4f}"
            )

        _health_check()
        controller.command_joints(q_target, gripper_pos=gripper_target).wait()
        last_cmd_q = q_target.copy()
        time.sleep(step_sleep_s)

        _health_check()
        joints_rb = controller.get_joint_positions().wait()[0]
        q_rb = _joints_dict_to_q5(joints_rb)

        abs_err = np.abs(q_rb - q_target)
        abs_err_all.append(abs_err)
        max_err = float(np.max(abs_err))
        mean_err = float(np.mean(abs_err))

        pos_err_m, rot_err_deg = _fk_pose_errors(kin, q_target, q_rb)
        ee_pos_err_all.append(pos_err_m)
        ee_rot_err_all.append(rot_err_deg)

        print(
            "[run_sequence] "
            f"joint_max={max_err:.3f}deg joint_mean={mean_err:.3f}deg "
            f"ee_pos={pos_err_m:.4f}m ee_rot={rot_err_deg:.2f}deg "
            f"target={q_target.tolist()} readback={q_rb.tolist()}"
        )

        if max_err > error_threshold_deg + 1e-9:
            raise RuntimeError(
                f"Safety stop: readback joint error too large. max_err={max_err:.4f} > error_threshold_deg={error_threshold_deg:.4f}"
            )

    for r in range(repeat):
        print(f"[run_sequence] === repeat {r+1}/{repeat} ===")
        if do_gripper:
            _health_check()
            controller.open_gripper(pos=100.0).wait()
            time.sleep(step_sleep_s)
            _health_check()
            controller.close_gripper(pos=0.0).wait()
            time.sleep(step_sleep_s)

        for d in deltas:
            _command_and_check(q0 + d, gripper_target=None)

    print("[run_sequence] returning to initial pose...")
    try:
        _command_and_check(q0, gripper_target=g0)
    except Exception as e:
        print(f"[run_sequence] return-to-home failed: {e!r}")
        try:
            controller.command_joints(q0, gripper_pos=g0).wait()
            time.sleep(step_sleep_s)
        except Exception:
            pass

    if abs_err_all:
        E = np.stack(abs_err_all, axis=0)
        print(f"[run_sequence] summary joint_max={float(np.max(E)):.3f}deg joint_mean={float(np.mean(E)):.3f}deg")
        print(f"[run_sequence] summary per_joint_max(deg)={np.max(E, axis=0).tolist()}")
        print(f"[run_sequence] summary ee_pos_max={max(ee_pos_err_all):.4f}m ee_pos_mean={float(np.mean(ee_pos_err_all)):.4f}m")
        print(f"[run_sequence] summary ee_rot_max={max(ee_rot_err_all):.2f}deg ee_rot_mean={float(np.mean(ee_rot_err_all)):.2f}deg")

    joints_end = controller.get_joint_positions().wait()[0]
    q_end = _joints_dict_to_q5(joints_end)
    home_err = np.abs(q_end - q0)
    print(f"[run_sequence] final home error per joint(deg)={home_err.tolist()}")


def main():
    args = _parse_args()

    # URDF
    urdf_path = args.urdf_path
    if not urdf_path:
        urdf_path = str(
            Path(__file__).resolve().parents[2]
            / "rlinf"
            / "envs"
            / "realworld"
            / "so101"
            / "assets"
            / "so101_minimal.urdf"
        )
    kin = SO101Kinematics(urdf_path=urdf_path)

    controller = connect_so101_controller()
    try:
        run_sequence(
            controller=controller,
            kin=kin,
            repeat=args.repeat,
            step_sleep_s=args.step_sleep_s,
            max_delta_deg=args.max_delta_deg,
            error_threshold_deg=args.error_threshold_deg,
            do_gripper=args.do_gripper,
        )
    finally:
        try:
            controller.disconnect().wait()
        except Exception:
            pass


if __name__ == "__main__":
    main()

