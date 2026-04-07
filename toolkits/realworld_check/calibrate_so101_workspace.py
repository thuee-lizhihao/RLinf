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

"""SO101 robot workspace calibration: controller joint readback + FK sampling."""

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.so101.kinematics import SO101Kinematics
from test_so101_setup import connect_so101_controller

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _default_urdf_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "rlinf" / "envs" / "realworld" / "so101" / "assets" / "so101_minimal.urdf")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _read_urdf_limits_deg(urdf_path: str) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf_path).getroot()
    out: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if joint.get("type", "") != "revolute" or name not in JOINTS:
            continue
        lim = joint.find("limit")
        if lim is None or lim.get("lower") is None or lim.get("upper") is None:
            continue
        lo = float(lim.get("lower", "nan"))
        hi = float(lim.get("upper", "nan"))
        if np.isfinite(lo) and np.isfinite(hi):
            out[name] = (float(np.rad2deg(lo)), float(np.rad2deg(hi)))
    miss = [j for j in JOINTS if j not in out]
    if miss:
        raise RuntimeError(f"URDF missing joint limits: {miss}")
    return out


def _intersect_limits(
    urdf_limits: dict[str, tuple[float, float]],
    calib_limits: dict[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for j in JOINTS:
        u_lo, u_hi = urdf_limits[j]
        if calib_limits is not None and j in calib_limits:
            c_lo, c_hi = calib_limits[j]
            lo, hi = max(u_lo, c_lo), min(u_hi, c_hi)
        else:
            lo, hi = u_lo, u_hi
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise RuntimeError(f"Joint {j} intersection limits invalid: [{lo}, {hi}]")
        out[j] = (float(lo), float(hi))
    return out


def _q5_from_dict(j: dict[str, float]) -> np.ndarray:
    return np.array([j[k] for k in JOINTS], dtype=float)


def _fk_xyz_euler(kin: SO101Kinematics, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = kin.forward_kinematics(np.asarray(q_deg, dtype=float).reshape(5))
    return T[:3, 3].copy(), R.from_matrix(T[:3, :3].copy()).as_euler("xyz")


def _stats_minmax(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.min(x, axis=0), np.max(x, axis=0)


def _expand_minmax(x_min: np.ndarray, x_max: np.ndarray, margin_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if margin_ratio <= 0:
        return x_min, x_max
    pad = (x_max - x_min) * float(margin_ratio)
    return x_min - pad, x_max + pad


def main() -> None:
    p = argparse.ArgumentParser(description="SO101 robot calibration (passive only)")
    p.add_argument("--urdf_path", default=os.environ.get("SO101_URDF_PATH", "") or _default_urdf_path())
    p.add_argument("--target_frame_name", default=os.environ.get("SO101_TARGET_FRAME", "gripper_frame_link"))
    p.add_argument("--margin_ratio", type=float, default=float(os.environ.get("SO101_WORKSPACE_MARGIN_RATIO", "0.05")))
    p.add_argument("--passive_hz", type=float, default=float(os.environ.get("SO101_PASSIVE_HZ", "10.0")))
    p.add_argument("--passive_seconds", type=float, default=float(os.environ.get("SO101_PASSIVE_SECONDS", "30.0")))
    p.add_argument("--out", default=str(Path.cwd() / "so101_workspace_robot.json"))
    args = p.parse_args()

    urdf_limits = _read_urdf_limits_deg(args.urdf_path)
    kin = SO101Kinematics(urdf_path=args.urdf_path, target_frame_name=args.target_frame_name, joint_names=JOINTS)
    controller = connect_so101_controller()

    try:
        try:
            calib_limits = controller.get_joint_limits_deg().wait()[0]
        except Exception:
            calib_limits = None
        sample_limits = _intersect_limits(urdf_limits, calib_limits)
        q_list: list[np.ndarray] = []
        xyz_list: list[np.ndarray] = []
        euler_list: list[np.ndarray] = []

        try:
            controller.set_torque_enabled(False).wait()
        except Exception as e:
            print(f"[robot][passive] warning: torque disable failed: {e!r}")
        input(f"[robot][passive] move robot by hand and press Enter to sample ({args.passive_seconds:.1f}s)...")
        dt = 1.0 / float(args.passive_hz)
        t_end = time.time() + float(args.passive_seconds)
        while time.time() < t_end:
            q = _q5_from_dict(controller.get_joint_positions().wait()[0])
            xyz, euler = _fk_xyz_euler(kin, q)
            if np.all(np.isfinite(xyz)) and np.all(np.isfinite(euler)):
                q_list.append(q)
                xyz_list.append(xyz)
                euler_list.append(euler)
            time.sleep(dt)

        if not xyz_list:
            raise RuntimeError("No valid robot samples.")

        q_np = np.stack(q_list, axis=0)
        xyz = np.stack(xyz_list, axis=0)
        euler = np.stack(euler_list, axis=0)

        q_min, q_max = _stats_minmax(q_np)
        xyz_min, xyz_max = _stats_minmax(xyz)
        euler_min, euler_max = _stats_minmax(euler)
        xyz_min2, xyz_max2 = _expand_minmax(xyz_min, xyz_max, args.margin_ratio)
        euler_min2, euler_max2 = _expand_minmax(euler_min, euler_max, args.margin_ratio)
        euler_min2 = np.clip(euler_min2, -np.pi, np.pi)
        euler_max2 = np.clip(euler_max2, -np.pi, np.pi)

        result = {
            "schema_version": 2,
            "kind": "robot_passive",
            "created_at": _now_iso(),
            "urdf_path": args.urdf_path,
            "target_frame_name": args.target_frame_name,
            "mode": "passive",
            "num_samples_requested": int(round(args.passive_hz * args.passive_seconds)),
            "num_samples_valid": int(xyz.shape[0]),
            "urdf_joint_limits_deg": {k: [v[0], v[1]] for k, v in urdf_limits.items()},
            "calibration_joint_limits_deg": None if calib_limits is None else {k: [float(v[0]), float(v[1])] for k, v in calib_limits.items()},
            "sampling_joint_limits_deg": {k: [v[0], v[1]] for k, v in sample_limits.items()},
            "q_min_deg": q_min.tolist(),
            "q_max_deg": q_max.tolist(),
            "xyz_min_m": xyz_min.tolist(),
            "xyz_max_m": xyz_max.tolist(),
            "euler_min_rad": euler_min.tolist(),
            "euler_max_rad": euler_max.tolist(),
            "margin_ratio_applied": float(args.margin_ratio),
            "recommended_ee_pose_limit_min": np.concatenate([xyz_min2, euler_min2]).tolist(),
            "recommended_ee_pose_limit_max": np.concatenate([xyz_max2, euler_max2]).tolist(),
        }

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[robot] wrote: {args.out}")
        print(f"[robot] sampling_joint_limits_deg={result['sampling_joint_limits_deg']}")
        print(f"[robot] recommended_ee_pose_limit_min={result['recommended_ee_pose_limit_min']}")
        print(f"[robot] recommended_ee_pose_limit_max={result['recommended_ee_pose_limit_max']}")
    finally:
        try:
            controller.disconnect().wait()
        except Exception:
            pass


if __name__ == "__main__":
    main()

