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

"""SO101 setup smoke test: camera + controller.

Intentionally minimal:
- Optional RealSense check (env-driven)
- Launch controller and connect once
"""

import os

from rlinf.envs.realworld.so101.so101_controller import SO101Controller


def _truthy_env(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in {"1", "true", "t", "yes", "y", "on"}


def check_realsense(*, require: bool) -> list[str]:
    try:
        import pyrealsense2 as rs  # type: ignore
    except ModuleNotFoundError:
        if require:
            raise
        print("[setup] pyrealsense2 not installed; skip camera check.")
        return []

    serials = [dev.get_info(rs.camera_info.serial_number) for dev in rs.context().devices]
    print(f"[setup] RealSense count={len(serials)} serials={serials}")
    if require and not serials:
        raise RuntimeError("SO101_REQUIRE_CAMERA=1 but no RealSense camera detected.")
    return serials


def connect_so101_controller() -> SO101Controller:
    port = os.environ.get("SO101_PORT")
    assert port, "Please set SO101_PORT, e.g. /dev/ttyACM0"

    calibration_path = os.environ.get("SO101_CALIBRATION_PATH")
    os.environ["SO101_REQUIRE_CALIBRATION"] = "1"

    if calibration_path:
        if not os.path.isfile(calibration_path):
            raise FileNotFoundError(f"SO101_CALIBRATION_PATH does not exist: {calibration_path}")
        print(f"[setup] using calibration: {calibration_path}")
    else:
        print("[setup] SO101_CALIBRATION_PATH not set; will auto-discover calibration JSON.")
        os.environ.setdefault("SO101_AUTO_CALIBRATE", "1")

    controller = SO101Controller.launch_controller(port=port, calibration_path=calibration_path)
    controller.connect().wait()
    return controller


def main():
    require_camera = _truthy_env("SO101_REQUIRE_CAMERA", default="1")
    check_realsense(require=require_camera)

    controller = connect_so101_controller()
    try:
        joints = controller.get_joint_positions().wait()[0]
        print(f"[setup] controller connected, joints={joints}")
    finally:
        try:
            controller.disconnect().wait()
        except Exception:
            pass


if __name__ == "__main__":
    main()

