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

"""SO101 offline workspace calibration: URDF + (optional) calibration limits + FK sampling."""

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.so101.kinematics import SO101Kinematics

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


def _parse_deg_pair(v: Any) -> tuple[float, float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    if isinstance(v, dict):
        keys = [
            ("min", "max"),
            ("lo", "hi"),
            ("lower", "upper"),
            ("min_deg", "max_deg"),
        ]
        for k1, k2 in keys:
            if k1 in v and k2 in v:
                return float(v[k1]), float(v[k2])
    return None


def _read_calibration_limits_deg(calibration_path: str, default_resolution: float) -> dict[str, tuple[float, float]]:
    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}

    # Prefer structures with degree min/max limits (e.g. JSON output by this tool)
    for top in ("calibration_joint_limits_deg", "joint_limits_deg", "command_joint_limits_deg"):
        if isinstance(payload.get(top), dict):
            for j in JOINTS:
                pair = _parse_deg_pair(payload[top].get(j))
                if pair is not None:
                    out[j] = pair

    # If calibration only has range_min/range_max(raw), approximate to symmetric degree limits like controller logic
    if len(out) < len(JOINTS):
        containers = [payload]
        for k in ("motors", "calibration", "joints"):
            if isinstance(payload.get(k), dict):
                containers.append(payload[k])
        for c in containers:
            for j in JOINTS:
                if j in out or not isinstance(c.get(j), dict):
                    continue
                m = c[j]
                if "range_min" in m and "range_max" in m:
                    span_raw = float(m["range_max"]) - float(m["range_min"])
                    max_res = float(m.get("max_resolution", default_resolution))
                    max_abs_deg = (span_raw * 180.0) / max_res
                    out[j] = (-max_abs_deg, +max_abs_deg)
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


def _sample_uniform(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, n: int) -> np.ndarray:
    return rng.random(size=(n, lo.shape[0])) * (hi - lo) + lo


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
    p = argparse.ArgumentParser(description="SO101 offline calibration (supports calibration limit sampling)")
    p.add_argument("--urdf_path", default=os.environ.get("SO101_URDF_PATH", "") or _default_urdf_path())
    p.add_argument("--target_frame_name", default=os.environ.get("SO101_TARGET_FRAME", "gripper_frame_link"))
    p.add_argument("--calibration_path", default=os.environ.get("SO101_CALIBRATION_PATH", ""))
    p.add_argument("--calibration_default_resolution", type=float, default=4095.0)
    p.add_argument("--num_samples", type=int, default=int(os.environ.get("SO101_WORKSPACE_SAMPLES", "5000")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SO101_WORKSPACE_SEED", "0")))
    p.add_argument("--margin_ratio", type=float, default=float(os.environ.get("SO101_WORKSPACE_MARGIN_RATIO", "0")))
    p.add_argument("--out", default=str(Path.cwd() / "so101_workspace_offline.json"))
    args = p.parse_args()

    urdf_limits = _read_urdf_limits_deg(args.urdf_path)
    calib_limits = None
    if args.calibration_path:
        calib_limits = _read_calibration_limits_deg(args.calibration_path, args.calibration_default_resolution)
    sample_limits = _intersect_limits(urdf_limits, calib_limits)

    lo = np.array([sample_limits[j][0] for j in JOINTS], dtype=float)
    hi = np.array([sample_limits[j][1] for j in JOINTS], dtype=float)
    rng = np.random.default_rng(args.seed)
    q_all = _sample_uniform(rng, lo, hi, int(args.num_samples))

    kin = SO101Kinematics(urdf_path=args.urdf_path, target_frame_name=args.target_frame_name, joint_names=JOINTS)
    xyz_list: list[np.ndarray] = []
    euler_list: list[np.ndarray] = []
    q_valid: list[np.ndarray] = []
    for q in q_all:
        xyz, euler = _fk_xyz_euler(kin, q)
        if np.all(np.isfinite(xyz)) and np.all(np.isfinite(euler)):
            q_valid.append(q)
            xyz_list.append(xyz)
            euler_list.append(euler)
    if not xyz_list:
        raise RuntimeError("No valid FK samples.")

    xyz = np.stack(xyz_list, axis=0)
    euler = np.stack(euler_list, axis=0)
    q_np = np.stack(q_valid, axis=0)

    xyz_min, xyz_max = _stats_minmax(xyz)
    euler_min, euler_max = _stats_minmax(euler)
    q_min, q_max = _stats_minmax(q_np)

    xyz_min2, xyz_max2 = _expand_minmax(xyz_min, xyz_max, args.margin_ratio)
    euler_min2, euler_max2 = _expand_minmax(euler_min, euler_max, args.margin_ratio)
    euler_min2 = np.clip(euler_min2, -np.pi, np.pi)
    euler_max2 = np.clip(euler_max2, -np.pi, np.pi)

    result = {
        "schema_version": 2,
        "kind": "offline",
        "created_at": _now_iso(),
        "urdf_path": args.urdf_path,
        "calibration_path": args.calibration_path or None,
        "target_frame_name": args.target_frame_name,
        "num_samples_requested": int(args.num_samples),
        "num_samples_valid": int(xyz.shape[0]),
        "seed": int(args.seed),
        "urdf_joint_limits_deg": {k: [v[0], v[1]] for k, v in urdf_limits.items()},
        "calibration_joint_limits_deg": None if calib_limits is None else {k: [v[0], v[1]] for k, v in calib_limits.items()},
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
    print(f"[offline] wrote: {args.out}")
    print(f"[offline] sampling_joint_limits_deg={result['sampling_joint_limits_deg']}")
    print(f"[offline] recommended_ee_pose_limit_min={result['recommended_ee_pose_limit_min']}")
    print(f"[offline] recommended_ee_pose_limit_max={result['recommended_ee_pose_limit_max']}")


if __name__ == "__main__":
    main()
