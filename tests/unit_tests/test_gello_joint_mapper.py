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

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.realworld.common.gello import GelloJointMapper


def test_raw_joint_round_trip() -> None:
    """Calibrated joint values round-trip through raw Dynamixel space."""
    mapper = GelloJointMapper(
        signs=np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
        offsets=np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4]),
    )
    joints = np.array([0.0, -0.5, 0.7, -1.2, 1.4, -2.0, 2.5])

    raw = mapper.joint_to_raw(joints)
    recovered = mapper.raw_to_joint(raw)

    np.testing.assert_allclose(recovered, joints)


def test_unwrap_chooses_nearest_equivalent() -> None:
    """Unwrap maps positions to the nearest 2π-equivalent reference."""
    mapper = GelloJointMapper()
    reference = np.array([0.1, -2.9, 2.8, 0.0, 1.0, -1.0, 3.0])
    q = reference + np.array(
        [2.0 * np.pi + 0.05, -2.0 * np.pi - 0.1, 0.2, -0.3, 0.0, 0.4, -0.2]
    )

    unwrapped = mapper.unwrap(q, reference)

    np.testing.assert_allclose(
        unwrapped,
        reference + np.array([0.05, -0.1, 0.2, -0.3, 0.0, 0.4, -0.2]),
    )


def test_mapper_rejects_invalid_shapes() -> None:
    """Shape mismatches fail before hardware code sees bad vectors."""
    mapper = GelloJointMapper(signs=[1.0, -1.0], offsets=[0.0, 0.5])

    with pytest.raises(ValueError, match="raw must have shape"):
        mapper.raw_to_joint([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        GelloJointMapper(signs=[1.0], offsets=[0.0, 1.0])

    with pytest.raises(ValueError, match="only \\+1 or -1"):
        GelloJointMapper(signs=[2.0, -1.0], offsets=[0.0, 0.0])
