# Copyright 2026 The RLinf Authors.
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

"""Smoke-test the shared-bus GELLO joint reader without actuator motion."""

from __future__ import annotations

import argparse
import time

import numpy as np

from rlinf.envs.realworld.common.gello import (
    GelloDynamixelBus,
    GelloJointExpert,
    GelloJointMapper,
)


def _parse_vector(text: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(
            f"{name} must contain exactly 7 comma-separated values, got {len(values)}"
        )
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Read GELLO joints through GelloDynamixelBus and GelloJointExpert. "
            "This only reads the serial bus; it does not command actuators."
        )
    )
    parser.add_argument(
        "--port",
        required=True,
        help="GELLO serial port, preferably a stable /dev/serial/by-id path.",
    )
    parser.add_argument(
        "--signs",
        required=True,
        type=lambda text: _parse_vector(text, "signs"),
        help='Calibration signs from gello_calibrate.py, e.g. "1,-1,1,1,-1,1,1".',
    )
    parser.add_argument(
        "--offsets",
        required=True,
        type=lambda text: _parse_vector(text, "offsets"),
        help='Calibration offsets from gello_calibrate.py, e.g. "0,0,0,0,0,0,0".',
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=float(np.pi),
        help="Warn when the max joint delta between samples exceeds this value.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the first GELLO frame.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the smoke test until interrupted."""
    args = parse_args()

    bus = GelloDynamixelBus(port=args.port)
    mapper = GelloJointMapper(signs=args.signs, offsets=args.offsets)
    expert = GelloJointExpert(bus=bus, mapper=mapper)

    try:
        deadline = time.monotonic() + args.ready_timeout
        print("Waiting for GELLO frames...", flush=True)
        while not expert.ready and time.monotonic() < deadline:
            time.sleep(0.05)

        if not expert.ready:
            raise RuntimeError(
                "GELLO reader did not become ready; check the port, permissions, "
                "and whether another process owns the serial device."
            )

        previous_joints = None
        with np.printoptions(precision=3, suppress=True):
            while True:
                joints, gripper = expert.get_action()
                marker = ""
                if previous_joints is not None:
                    max_jump = float(np.max(np.abs(joints - previous_joints)))
                    if max_jump > args.jump_threshold:
                        marker = f"  JUMP? max_delta={max_jump:.3f}"
                print(f"joints={joints} gripper={gripper}{marker}", end="\r")
                previous_joints = joints
                time.sleep(0.1)
    finally:
        expert.close()
        bus.close()
        print("\nClosed GELLO reader cleanly.")


if __name__ == "__main__":
    main()
