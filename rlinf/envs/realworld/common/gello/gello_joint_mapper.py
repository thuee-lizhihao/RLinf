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

"""Calibration mapping between raw GELLO Dynamixel and robot joints."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class GelloJointMapper:
    """Map GELLO raw motor positions to Franka-shaped joint coordinates.

    The mapper applies the calibrated per-joint affine transform used by the
    GELLO calibration tools:

    ``q_joint = sign * (q_raw - offset)``

    Args:
        signs: Per-joint sign values, normally ``+1`` or ``-1``.
        offsets: Per-joint raw-position offsets in radians.
    """

    def __init__(
        self,
        signs: Sequence[float] | np.ndarray | None = None,
        offsets: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if signs is None and offsets is None:
            signs_arr = np.ones(7, dtype=np.float64)
            offsets_arr = np.zeros(7, dtype=np.float64)
        elif signs is None or offsets is None:
            raise ValueError("signs and offsets must be provided together")
        else:
            signs_arr = np.asarray(signs, dtype=np.float64)
            offsets_arr = np.asarray(offsets, dtype=np.float64)

        if signs_arr.ndim != 1 or offsets_arr.ndim != 1:
            raise ValueError("signs and offsets must be 1-D arrays")
        if signs_arr.shape != offsets_arr.shape:
            raise ValueError(
                "signs and offsets must have the same shape, got "
                f"{signs_arr.shape} and {offsets_arr.shape}"
            )
        if not np.all(np.isfinite(signs_arr)) or not np.all(np.isfinite(offsets_arr)):
            raise ValueError("signs and offsets must contain only finite values")
        if not np.all(np.isin(signs_arr, (-1.0, 1.0))):
            raise ValueError("signs must contain only +1 or -1")

        self.signs = signs_arr.copy()
        self.offsets = offsets_arr.copy()

    @property
    def num_joints(self) -> int:
        """Number of joints handled by this mapper."""
        return int(self.signs.shape[0])

    def raw_to_joint(self, raw: Sequence[float] | np.ndarray) -> np.ndarray:
        """Convert raw Dynamixel positions to calibrated joint positions.

        Args:
            raw: Raw motor positions in radians.

        Returns:
            Calibrated joint positions in radians.
        """
        raw_arr = self._as_vector(raw, "raw")
        return self.signs * (raw_arr - self.offsets)

    def joint_to_raw(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Convert calibrated joint positions to raw Dynamixel positions.

        Args:
            q: Joint positions in radians.

        Returns:
            Raw motor target positions in radians.
        """
        q_arr = self._as_vector(q, "q")
        return self.signs * q_arr + self.offsets

    def unwrap(
        self,
        q: Sequence[float] | np.ndarray,
        reference: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return the equivalent joint vector nearest to ``reference``.

        Args:
            q: Joint positions in radians.
            reference: Reference joint positions used to choose the nearest
                ``2π`` equivalent.

        Returns:
            Unwrapped joint positions, continuous with ``reference``.
        """
        q_arr = self._as_vector(q, "q")
        ref_arr = self._as_vector(reference, "reference")
        return ref_arr + (q_arr - ref_arr + np.pi) % (2.0 * np.pi) - np.pi

    def _as_vector(self, values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != self.signs.shape:
            raise ValueError(
                f"{name} must have shape {self.signs.shape}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        return arr
