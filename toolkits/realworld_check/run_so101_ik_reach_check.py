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
import os
from typing import Any

import gymnasium as gym
import numpy as np

# Import for side-effect task registration.
import rlinf.envs.realworld.so101.tasks  # noqa: F401
from rlinf.scheduler.hardware import SO101Config, SO101HWInfo

# Minimal task config (edit these values directly when needed).
TARGET_EE_POSE = [0.20, 0.00, 0.10, 0.00, 0.00, 0.00]  # [x, y, z, rx, ry, rz]
RESET_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0]  # 5 arm joints in degrees
SUCCESS_POSITION_ERROR_M = 0.04
SUCCESS_ORIENTATION_ERROR_DEG = 30.0
CHECK_ORIENTATION = False
NUM_EPISODES = 2
STEPS_PER_EPISODE = 100
PRINT_EVERY_N_STEPS = 10
REQUIRE_ALL_SUCCESS = False


def _camera_serials_from_env() -> list[str]:
    text = os.environ.get("SO101_CAMERA_SERIALS")
    if not text:
        raise RuntimeError(
            "SO101_CAMERA_SERIALS is not set, e.g. export SO101_CAMERA_SERIALS=233622077247"
        )
    serials = [x.strip() for x in text.split(",") if x.strip()]
    if not serials:
        raise ValueError("SO101_CAMERA_SERIALS is empty. Please provide at least one serial.")
    return serials


def _step_result(
    episode_id: int,
    step_id: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
):
    payload = {
        "episode_id": episode_id,
        "step_id": step_id,
        "task_success": bool(info.get("task_success", False)),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "task_pos_error_m": float(info.get("task_pos_error_m", np.nan)),
        "task_rot_error_deg": float(info.get("task_rot_error_deg", np.nan)),
        "ik_fallback_level": str(info.get("ik_fallback_level", "unknown")),
        "ik_converged": bool(info.get("ik_converged", False)),
        "move_success": bool(info.get("move_success", False)),
        "move_timed_out": bool(info.get("move_timed_out", False)),
    }
    return payload


def main():
    port = os.environ.get("SO101_PORT")
    if not port:
        raise RuntimeError("SO101_PORT is not set, e.g. export SO101_PORT=/dev/ttyACM0")
    camera_serials = _camera_serials_from_env()
    calibration_path = os.environ.get("SO101_CALIBRATION_PATH")

    override_cfg = {
        "port": port,
        "camera_serials": camera_serials,
        "enable_camera_player": False,
        "calibration_path": calibration_path,
        "target_ee_pose": TARGET_EE_POSE,
        "success_position_error_m": float(SUCCESS_POSITION_ERROR_M),
        "success_orientation_error_deg": float(SUCCESS_ORIENTATION_ERROR_DEG),
        "reset_position": RESET_POSITION,
        "check_orientation": bool(CHECK_ORIENTATION),
        "max_num_steps": max(1, int(STEPS_PER_EPISODE)),
        "step_frequency": 2.0,
        "max_joint_step_delta_deg": 20.0,
        "reset_max_step_delta_deg": 8.0,
        "reset_max_iters": 40,
        "reset_timeout_s": 2.0,
    }

    hardware_info = SO101HWInfo(
        type="SO101",
        model="SO101",
        config=SO101Config(
            port=port,
            camera_serials=camera_serials,
            node_rank=0,
            disable_validate=True,
        ),
    )

    env = gym.make(
        id="SO101IKReachCheckEnv-v1",
        override_cfg=override_cfg,
        worker_info=None,
        hardware_info=hardware_info,
        env_idx=0,
    )
    step_results: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []
    try:
        num_episodes = max(1, int(NUM_EPISODES))
        steps_per_episode = max(1, int(STEPS_PER_EPISODE))
        print_every_n_steps = max(1, int(PRINT_EVERY_N_STEPS))
        for episode_id in range(num_episodes):
            _, reset_info = env.reset()
            if reset_info:
                print(json.dumps({"episode_id": episode_id, "reset_info": reset_info}, sort_keys=True))

            episode_success = False
            terminated = False
            truncated = False
            steps_executed = 0
            for step_id in range(steps_per_episode):
                _, reward, terminated, truncated, info = env.step(np.zeros((7,), dtype=np.float32))
                payload = _step_result(
                    episode_id=episode_id,
                    step_id=step_id,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                step_results.append(payload)
                steps_executed += 1
                episode_success = episode_success or bool(payload["task_success"])
                should_print_step = ((step_id + 1) % print_every_n_steps == 0) or terminated or truncated
                if should_print_step:
                    print(json.dumps(payload, sort_keys=True))
                if terminated or truncated:
                    break

            episode_payload = {
                "episode_id": episode_id,
                "steps_executed": steps_executed,
                "episode_success": episode_success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            episode_results.append(episode_payload)
            print(json.dumps({"episode_summary": episode_payload}, sort_keys=True))
    finally:
        env.close()

    success_count = int(sum(1 for r in episode_results if bool(r["episode_success"])))
    summary = {
        "num_episodes": len(episode_results),
        "steps_per_episode": max(1, int(STEPS_PER_EPISODE)),
        "num_steps_total": len(step_results),
        "success_count": success_count,
        "success_rate": (
            float(success_count) / float(len(episode_results)) if episode_results else 0.0
        ),
        "position_error_m_mean": float(np.nanmean([r["task_pos_error_m"] for r in step_results])),
        "rotation_error_deg_mean": float(np.nanmean([r["task_rot_error_deg"] for r in step_results])),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))

    if REQUIRE_ALL_SUCCESS and success_count != len(episode_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
