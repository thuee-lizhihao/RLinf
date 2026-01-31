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

    port = os.environ.get("SO101_PORT", None)
    assert port is not None, "Please set SO101_PORT, e.g. /dev/ttyACM0"

    # Calibration is REQUIRED by default.
    os.environ.setdefault("SO101_REQUIRE_CALIBRATION", "1")

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

    controller = SO101Controller.launch_controller(
        port=port, calibration_path=calibration_path
    )
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

    # RealSense camera check (enumerate + optional serial validation)
    # Put it here so the output is close to the interactive prompt and won't be drowned by logs.
    require_camera = _is_truthy_env("SO101_REQUIRE_CAMERA", default="1")
    expected_camera_serials = _parse_camera_serials_env()
    # region agent log
    _dlog(
        "H2_CAM",
        "test_so101_controller.py:main:camera_check",
        "Validating RealSense cameras",
        {
            "require_camera": require_camera,
            "expected_camera_serials": expected_camera_serials,
        },
    )
    _dlog2(
        "H2_CAM",
        "test_so101_controller.py:main:camera_check",
        "Validating RealSense cameras",
        {
            "require_camera": require_camera,
            "expected_camera_serials": expected_camera_serials,
        },
    )
    # endregion
    _validate_realsense_cameras(
        expected_serials=expected_camera_serials or None, require=require_camera
    )

    while True:
        try:
            cmd_str = input("Please input cmd (getpos/getpos_euler/q): ")
            if cmd_str == "q":
                break
            elif cmd_str == "getpos":
                joints = controller.get_joint_positions().wait()[0]
                q = np.array(
                    [
                        joints["shoulder_pan"],
                        joints["shoulder_lift"],
                        joints["elbow_flex"],
                        joints["wrist_flex"],
                        joints["wrist_roll"],
                    ],
                    dtype=float,
                )
                T = kin.forward_kinematics(q)
                pos = T[:3, 3]
                quat = R.from_matrix(T[:3, :3].copy()).as_quat()
                print(np.concatenate([pos, quat]))
            elif cmd_str == "getpos_euler":
                joints = controller.get_joint_positions().wait()[0]
                q = np.array(
                    [
                        joints["shoulder_pan"],
                        joints["shoulder_lift"],
                        joints["elbow_flex"],
                        joints["wrist_flex"],
                        joints["wrist_roll"],
                    ],
                    dtype=float,
                )
                T = kin.forward_kinematics(q)
                pos = T[:3, 3]
                euler = R.from_matrix(T[:3, :3].copy()).as_euler("xyz")
                print(np.concatenate([pos, euler]))
            else:
                print(f"Unknown cmd: {cmd_str}")
        except KeyboardInterrupt:
            break
        time.sleep(0.5)


if __name__ == "__main__":
    main()

