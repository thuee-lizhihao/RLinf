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
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class SO101HWInfo(HardwareInfo):
    """Hardware information for a SO101 robotic system."""

    config: "SO101Config"


@Hardware.register()
class SO101Robot(Hardware):
    """Hardware policy for SO101 robot arm (serial + RealSense)."""

    HW_TYPE = "SO101"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["SO101Config"]] = None
    ) -> Optional[HardwareResource]:
        assert configs is not None, (
            "Robot hardware requires explicit configurations for SO101 serial port and camera serials."
        )

        robot_configs: list["SO101Config"] = []
        for config in configs:
            if isinstance(config, SO101Config) and config.node_rank == node_rank:
                robot_configs.append(config)

        if not robot_configs:
            return None

        so101_infos: list[SO101HWInfo] = []
        cameras = cls.enumerate_cameras()

        for config in robot_configs:
            # Use auto detected cameras if not specified
            if config.camera_serials is None:
                config.camera_serials = list(cameras)

            so101_infos.append(
                SO101HWInfo(
                    type=cls.HW_TYPE,
                    model=cls.HW_TYPE,
                    config=config,
                )
            )

            if config.disable_validate:
                continue

            # Validate port exists (best-effort)
            if not os.path.exists(config.port):
                raise FileNotFoundError(
                    f"SO101 port '{config.port}' does not exist on node rank {node_rank}."
                )

            # Validate camera serials (RealSense)
            try:
                importlib.import_module("pyrealsense2")
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"pyrealsense2 is required for SO101 RealSense serials check, but it is not installed on the node with rank {node_rank}."
                )
            if not cameras:
                raise ValueError(
                    f"No RealSense cameras are connected to node rank {node_rank} while SO101 requires at least one camera."
                )
            for serial in config.camera_serials:
                if serial not in cameras:
                    raise ValueError(
                        f"Camera with serial {serial} for SO101 is not connected to node rank {node_rank}. "
                        f"Available cameras are: {cameras}."
                    )

        return HardwareResource(type=cls.HW_TYPE, infos=so101_infos)

    @classmethod
    def enumerate_cameras(cls) -> set[str]:
        """Enumerate connected RealSense camera serial numbers."""
        cameras: set[str] = set()
        try:
            import pyrealsense2 as rs
        except ImportError:
            return cameras
        for device in rs.context().devices:
            cameras.add(device.get_info(rs.camera_info.serial_number))
        return cameras


@NodeHardwareConfig.register_hardware_config(SO101Robot.HW_TYPE)
@dataclass
class SO101Config(HardwareConfig):
    """Configuration for SO101 robot arm controller node."""

    port: str
    """Serial port of the Feetech bus adapter, e.g. /dev/ttyACM0"""

    camera_serials: Optional[list[str]] = None
    """List of RealSense camera serial numbers associated with the robot."""

    disable_validate: bool = False
    """Whether to disable validation of port existence and camera serials."""

    def __post_init__(self):
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in so101 config must be an integer. But got {type(self.node_rank)}."
        )
        if not isinstance(self.port, str) or not self.port:
            raise ValueError(
                f"'port' in so101 config must be a non-empty string. But got {self.port!r}."
            )
        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)

