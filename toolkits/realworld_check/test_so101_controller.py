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

import importlib
import os
import subprocess
import time
from pathlib import Path
import shutil
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.so101.kinematics import SO101Kinematics
from rlinf.envs.realworld.so101.so101_controller import (
    SO101Controller,
    _default_lerobot_calibration_path,
    _discover_so101_calibration_files,
)

# region agent log
def _dlog(
    hypothesisId: str,
    location: str,
    message: str,
    data: dict | None = None,
    runId: str = "pre-fix",
):
    import json
    import time as _time

    log_path = "/home/zhihao/SO101_arm/.cursor/debug.log"
    try:
        os.makedirs(str(Path(log_path).parent), exist_ok=True)
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
        # Never fail due to logging.
        pass
# endregion

# region agent log
def _dlog2(
    hypothesisId: str,
    location: str,
    message: str,
    data: dict | None = None,
    runId: str = "pre-fix",
):
    """Debug logger that writes into the repo root `.cursor/debug.log`.

    This avoids hardcoding `/home/...` which may not exist inside containers.
    """
    import json
    import time as _time

    try:
        repo_root = Path(__file__).resolve().parents[3]
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


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    s = input(prompt + suffix).strip().lower()
    if s == "":
        return default
    return s in {"y", "yes", "true", "1"}


def _select_calibration_path(candidates: list[Path]) -> str:
    print("发现以下可能的 SO101 校准文件：")
    for i, p in enumerate(candidates):
        print(f"  [{i}] {p}")
    s = input("请选择要使用的校准文件序号（默认 0，输入 q 退出）： ").strip().lower()
    if s == "q":
        raise SystemExit(1)
    idx = 0 if s == "" else int(s)
    if idx < 0 or idx >= len(candidates):
        raise ValueError(f"无效序号 {idx}，候选数量={len(candidates)}")
    chosen = candidates[idx]
    if not _prompt_yes_no(f"确认使用该校准文件？{chosen}", default=True):
        raise SystemExit(1)
    return str(chosen)

def _run_lerobot_calibrate(robot_type: str, port: str, robot_id: str) -> None:
    if not shutil.which("lerobot-calibrate"):
        raise FileNotFoundError(
            "找不到 `lerobot-calibrate` 命令（未安装 LeRobot 或 PATH 未配置）。\n"
            "请先安装 LeRobot（推荐可编辑安装），例如在 LeRobot 仓库根目录执行：\n"
            "  pip install -e \".\"   # 或 pip install -e \".[feetech]\"\n"
            "然后重新运行本脚本；或手动运行 `lerobot-calibrate` 生成校准文件并设置 SO101_CALIBRATION_PATH。"
        )
    subprocess.run(
        [
            "lerobot-calibrate",
            f"--robot.type={robot_type}",
            f"--robot.port={port}",
            f"--robot.id={robot_id}",
        ],
        check=True,
    )


def _is_truthy_env(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _parse_camera_serials_env() -> list[str]:
    """Parse expected RealSense serials from env.

    Supported:
    - SO101_CAMERA_SERIALS="123,456" (comma/space separated)
    - SO101_CAMERA_SERIAL="123"
    - CAMERA_SERIAL="123"
    """
    raw = (
        os.environ.get("SO101_CAMERA_SERIALS")
        or os.environ.get("SO101_CAMERA_SERIAL")
        or os.environ.get("CAMERA_SERIAL")
        or ""
    ).strip()
    if not raw:
        return []
    # allow comma and/or whitespace separated
    parts = [p.strip() for p in raw.replace(",", " ").split()]
    return [p for p in parts if p]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SO101 controller check: interactive getpos/getpos_euler or run a safe joint-space sequence."
    )
    p.add_argument(
        "--mode",
        choices=["interactive", "run_sequence"],
        default=os.environ.get("SO101_MODE", "interactive"),
        help="Run mode. Can also set env SO101_MODE.",
    )

    # Sequence config (only used in run_sequence)
    p.add_argument("--repeat", type=int, default=int(os.environ.get("SO101_REPEAT", "1")))
    p.add_argument(
        "--step_sleep_s",
        type=float,
        default=float(os.environ.get("SO101_STEP_SLEEP_S", "0.8")),
        help="Sleep seconds between steps (allows motors to settle).",
    )
    p.add_argument(
        "--max_delta_deg",
        type=float,
        default=float(os.environ.get("SO101_MAX_DELTA_DEG", "2.0")),
        help="Per-step maximum absolute joint increment (deg).",
    )
    p.add_argument(
        "--error_threshold_deg",
        type=float,
        default=float(os.environ.get("SO101_ERROR_THRESHOLD_DEG", "3.0")),
        help="Allowed absolute joint error (deg) between readback and target.",
    )
    p.add_argument(
        "--require_calibration",
        action=argparse.BooleanOptionalAction,
        default=_is_truthy_env("SO101_REQUIRE_CALIBRATION", default="1"),
        help="Whether to enforce LeRobot calibration JSON (default: True).",
    )

    # Optional gripper check inside run_sequence
    p.add_argument(
        "--do_gripper",
        action=argparse.BooleanOptionalAction,
        default=_is_truthy_env("SO101_DO_GRIPPER", default="1"),
        help="Whether to include open/close gripper steps in run_sequence (default: True).",
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


def _default_safe_joint_deltas(amp_deg: float = 3.0) -> list[np.ndarray]:
    """A conservative joint-space delta sequence (5 joints), centered around current pose."""
    deltas: list[np.ndarray] = []
    # Single-joint small out-and-back.
    for i in range(5):
        d = np.zeros(5, dtype=float)
        d[i] = amp_deg
        # IMPORTANT: insert a zero step between +amp and -amp so that
        # per-step delta never exceeds amp (avoids a 2*amp jump).
        deltas.append(d.copy())
        deltas.append(np.zeros(5, dtype=float))
        deltas.append(-d.copy())
        deltas.append(np.zeros(5, dtype=float))

    # Small multi-joint patterns.
    deltas.extend(
        [
            np.array([+amp_deg, 0.0, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, +amp_deg, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, +amp_deg, 0.0, 0.0], dtype=float),
            np.array([+0.5 * amp_deg, -0.5 * amp_deg, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, +0.5 * amp_deg, -0.5 * amp_deg, 0.0, 0.0], dtype=float),
            np.zeros(5, dtype=float),
        ]
    )
    return deltas


def _run_sequence(
    *,
    controller: SO101Controller,
    kin: SO101Kinematics,
    repeat: int,
    step_sleep_s: float,
    max_delta_deg: float,
    error_threshold_deg: float,
    do_gripper: bool,
) -> None:
    """Execute a conservative joint sequence with readback checks."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if step_sleep_s <= 0:
        raise ValueError(f"step_sleep_s must be > 0, got {step_sleep_s}")
    if max_delta_deg <= 0:
        raise ValueError(f"max_delta_deg must be > 0, got {max_delta_deg}")
    if error_threshold_deg <= 0:
        raise ValueError(f"error_threshold_deg must be > 0, got {error_threshold_deg}")

    # Read initial joints.
    joints0 = controller.get_joint_positions().wait()[0]
    q0 = _joints_dict_to_q5(joints0)
    g0 = float(joints0.get("gripper", 0.0))
    print(f"[run_sequence] initial q(deg)={q0.tolist()}, gripper={g0:.3f}")

    # Default sequence is centered at q0.
    amp = min(5.0, max_delta_deg)
    deltas = _default_safe_joint_deltas(amp_deg=amp)

    abs_err_all: list[np.ndarray] = []
    last_cmd_q = q0.copy()

    def _health_check() -> None:
        try:
            ok = bool(controller.is_robot_up().wait()[0])
        except Exception:
            ok = False
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
        print(
            f"[run_sequence] step max_err={max_err:.3f}deg mean_err={mean_err:.3f}deg "
            f"target={q_target.tolist()} readback={q_rb.tolist()}"
        )
        if max_err > error_threshold_deg + 1e-9:
            raise RuntimeError(
                f"Safety stop: readback error too large. max_err={max_err:.4f} > error_threshold_deg={error_threshold_deg:.4f}"
            )

    # Execute.
    for r in range(repeat):
        print(f"[run_sequence] === repeat {r+1}/{repeat} ===")

        if do_gripper:
            # Open then close slightly, using current arm pose.
            _health_check()
            controller.open_gripper(pos=100.0).wait()
            time.sleep(step_sleep_s)
            _health_check()
            controller.close_gripper(pos=0.0).wait()
            time.sleep(step_sleep_s)

        for d in deltas:
            _command_and_check(q0 + d, gripper_target=None)

    # Return to initial pose (arm + restore gripper to original position best-effort).
    print("[run_sequence] returning to initial pose...")
    try:
        _command_and_check(q0, gripper_target=g0)
    except Exception as e:
        # Last resort: attempt to command q0 without strict checks.
        print(f"[run_sequence] return-to-home failed with error: {e!r}")
        try:
            controller.command_joints(q0, gripper_pos=g0).wait()
            time.sleep(step_sleep_s)
        except Exception:
            pass

    # Summary.
    if abs_err_all:
        E = np.stack(abs_err_all, axis=0)  # [steps, 5]
        max_err = float(np.max(E))
        mean_err = float(np.mean(E))
        per_joint_max = np.max(E, axis=0)
        print(f"[run_sequence] summary: max_err={max_err:.3f}deg mean_err={mean_err:.3f}deg")
        print(f"[run_sequence] per_joint_max(deg)={per_joint_max.tolist()}")

    joints_end = controller.get_joint_positions().wait()[0]
    q_end = _joints_dict_to_q5(joints_end)
    home_err = np.abs(q_end - q0)
    print(f"[run_sequence] final home error per joint(deg)={home_err.tolist()}")


def _enumerate_realsense_serials() -> set[str]:
    """Enumerate connected RealSense camera serial numbers."""
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception:
        return set()
    return {
        device.get_info(rs.camera_info.serial_number)
        for device in rs.context().devices
    }


def _validate_realsense_cameras(
    *, expected_serials: list[str] | None, require: bool
) -> set[str]:
    serials = _enumerate_realsense_serials()
    serial_list = sorted(list(serials))
    print(f"已检测到 RealSense 相机数量={len(serial_list)}，serials={serial_list}")

    if require:
        try:
            importlib.import_module("pyrealsense2")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "需要 `pyrealsense2` 才能做 SO101 RealSense 相机枚举/校验。请先安装并确保 RealSense SDK/驱动可用。"
            ) from e
        if not serials:
            raise ValueError(
                "未检测到任何 RealSense 相机，但当前 SO101_REQUIRE_CAMERA=1 要求至少连接 1 台相机。"
            )

    if expected_serials:
        missing = [s for s in expected_serials if s not in serials]
        if missing:
            raise ValueError(
                "期望的 RealSense 相机序列号未检测到。\n"
                f"  missing={missing}\n"
                f"  available={serial_list}\n"
                "你可以设置/修正 SO101_CAMERA_SERIALS 或 CAMERA_SERIAL，或检查相机连接与权限。"
            )

    return serials


def main():
    # region agent log
    import sys as _sys

    _dlog(
        "H1",
        "test_so101_controller.py:main:entry",
        "Enter main()",
        {
            "cwd": os.getcwd(),
            "sys_path0": _sys.path[0] if len(_sys.path) > 0 else None,
            "has_PYTHONPATH": bool(os.environ.get("PYTHONPATH")),
        },
    )
    _dlog2(
        "H1",
        "test_so101_controller.py:main:entry",
        "Enter main()",
        {
            "cwd": os.getcwd(),
            "sys_path0": _sys.path[0] if len(_sys.path) > 0 else None,
            "has_PYTHONPATH": bool(os.environ.get("PYTHONPATH")),
        },
    )
    # endregion

    args = _parse_args()

    # Let CLI control calibration requirement.
    os.environ["SO101_REQUIRE_CALIBRATION"] = "1" if args.require_calibration else "0"

    port = os.environ.get("SO101_PORT", None)
    assert port is not None, "Please set SO101_PORT, e.g. /dev/ttyACM0"

    robot_type = os.environ.get("SO101_ROBOT_TYPE", "so101_follower")
    robot_id = os.environ.get("SO101_ID", "so101")

    calibration_path = os.environ.get("SO101_CALIBRATION_PATH", None)
    urdf_path = os.environ.get("SO101_URDF_PATH", None)

    # region agent log
    _dlog(
        "H2",
        "test_so101_controller.py:main:env",
        "Loaded env vars",
        {
            "SO101_PORT_set": bool(port),
            "SO101_CALIBRATION_PATH_set": bool(calibration_path),
            "SO101_URDF_PATH_set": bool(urdf_path),
            "robot_type": robot_type,
            "robot_id": robot_id,
        },
    )
    _dlog2(
        "H2",
        "test_so101_controller.py:main:env",
        "Loaded env vars",
        {
            "SO101_PORT_set": bool(port),
            "SO101_CALIBRATION_PATH_set": bool(calibration_path),
            "SO101_URDF_PATH_set": bool(urdf_path),
            "robot_type": robot_type,
            "robot_id": robot_id,
            "SO101_CALIBRATION_PATH": calibration_path,
        },
    )
    # endregion

    if calibration_path:
        # region agent log
        _dlog(
            "H3",
            "test_so101_controller.py:main:calib_check",
            "Checking provided calibration_path",
        )
        _dlog2(
            "H3",
            "test_so101_controller.py:main:calib_check",
            "Checking provided calibration_path",
            {"calibration_path": calibration_path},
        )
        # endregion
        if not Path(calibration_path).expanduser().is_file():
            raise FileNotFoundError(
                "SO101_CALIBRATION_PATH 已设置，但指向的文件不存在。\n"
                f"  SO101_CALIBRATION_PATH={calibration_path}\n"
                "请修正该路径；或取消设置 SO101_CALIBRATION_PATH，让脚本自动搜索/提示校准流程。"
            )
    else:
        candidates = _discover_so101_calibration_files(
            robot_type=robot_type, robot_id=robot_id
        )
        if len(candidates) > 0:
            calibration_path = _select_calibration_path(candidates)
        else:
            expected = _default_lerobot_calibration_path(
                robot_type=robot_type, robot_id=robot_id
            )
            print("未发现任何 SO101 校准文件。")
            print("你可以运行 LeRobot 校准程序来生成校准文件：")
            print(
                f"  lerobot-calibrate --robot.type={robot_type} --robot.port={port} --robot.id={robot_id}"
            )
            print(f"常见生成路径：{expected}")
            if _prompt_yes_no("现在自动运行 `lerobot-calibrate`？", default=True):
                expected.parent.mkdir(parents=True, exist_ok=True)
                _run_lerobot_calibrate(robot_type=robot_type, port=port, robot_id=robot_id)
                if expected.is_file():
                    calibration_path = str(expected)
                    print(f"校准完成，使用校准文件：{calibration_path}")
                else:
                    # Re-scan in case LeRobot wrote elsewhere.
                    candidates2 = _discover_so101_calibration_files(
                        robot_type=robot_type, robot_id=robot_id
                    )
                    if len(candidates2) > 0:
                        calibration_path = _select_calibration_path(candidates2)
                    else:
                        raise FileNotFoundError(
                            "校准程序已运行，但未找到校准文件。"
                            "请确认 LeRobot 输出路径，并手动设置 SO101_CALIBRATION_PATH。"
                        )
            else:
                raise SystemExit(
                    "缺少校准文件，已退出。请先生成校准文件并设置 SO101_CALIBRATION_PATH。"
                )

    controller = SO101Controller.launch_controller(port=port, calibration_path=calibration_path)
    controller.connect().wait()

    # Kinematics
    if urdf_path is None:
        # Default to embedded minimal URDF
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

    try:
        # RealSense camera check (enumerate + optional serial validation)
        # Default policy:
        # - interactive: require camera unless user disables
        # - run_sequence: do NOT require camera unless user enables (Phase1 focuses on serial closed-loop)
        default_require_camera = "1" if args.mode == "interactive" else "0"
        require_camera = _is_truthy_env("SO101_REQUIRE_CAMERA", default=default_require_camera)
        expected_camera_serials = _parse_camera_serials_env()
        # region agent log
        _dlog(
            "H2_CAM",
            "test_so101_controller.py:main:camera_check",
            "Validating RealSense cameras",
            {
                "mode": args.mode,
                "require_camera": require_camera,
                "expected_camera_serials": expected_camera_serials,
            },
        )
        _dlog2(
            "H2_CAM",
            "test_so101_controller.py:main:camera_check",
            "Validating RealSense cameras",
            {
                "mode": args.mode,
                "require_camera": require_camera,
                "expected_camera_serials": expected_camera_serials,
            },
        )
        # endregion
        _validate_realsense_cameras(
            expected_serials=expected_camera_serials or None, require=require_camera
        )

        if args.mode == "run_sequence":
            _run_sequence(
                controller=controller,
                kin=kin,
                repeat=args.repeat,
                step_sleep_s=args.step_sleep_s,
                max_delta_deg=args.max_delta_deg,
                error_threshold_deg=args.error_threshold_deg,
                do_gripper=args.do_gripper,
            )
            return

        # interactive
        while True:
            try:
                cmd_str = input("Please input cmd (getpos/getpos_euler/q): ")
                if cmd_str == "q":
                    break
                elif cmd_str == "getpos":
                    joints = controller.get_joint_positions().wait()[0]
                    q = _joints_dict_to_q5(joints)
                    T = kin.forward_kinematics(q)
                    pos = T[:3, 3]
                    quat = R.from_matrix(T[:3, :3].copy()).as_quat()
                    print(np.concatenate([pos, quat]))
                elif cmd_str == "getpos_euler":
                    joints = controller.get_joint_positions().wait()[0]
                    q = _joints_dict_to_q5(joints)
                    T = kin.forward_kinematics(q)
                    pos = T[:3, 3]
                    euler = R.from_matrix(T[:3, :3].copy()).as_euler("xyz")
                    print(np.concatenate([pos, euler]))
                else:
                    print(f"Unknown cmd: {cmd_str}")
            except KeyboardInterrupt:
                break
            time.sleep(0.5)
    finally:
        # Always try to disconnect to leave the bus in a clean state.
        try:
            controller.disconnect().wait()
        except Exception:
            pass


if __name__ == "__main__":
    main()

