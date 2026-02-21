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

"""SO101 interactive controller check aligned with Franka script style."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from test_so101_setup import connect_so101_controller


def main():
    controller = connect_so101_controller()
    try:
        start_time = time.time()
        while not controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                print(
                    f"Waited {time.time() - start_time} seconds for SO101 robot to be ready."
                )

        while True:
            try:
                cmd_str = input("Please input cmd(q: quit, pos, pos_euler):")
                if cmd_str == "q":
                    break
                elif cmd_str == "pos":
                    state = controller.get_state(refresh=True).wait()[0]
                    print(state.tcp_pose)
                elif cmd_str == "pos_euler":
                    tcp_pose = controller.get_state(refresh=True).wait()[0].tcp_pose
                    r = R.from_quat(tcp_pose[3:].copy())
                    euler = r.as_euler("xyz")
                    print(np.concatenate([tcp_pose[:3], euler]))
                else:
                    print(f"Unknown cmd: {cmd_str}")
            except KeyboardInterrupt:
                break
            time.sleep(1.0)
    finally:
        try:
            controller.disconnect().wait()
        except Exception:
            pass


if __name__ == "__main__":
    main()

