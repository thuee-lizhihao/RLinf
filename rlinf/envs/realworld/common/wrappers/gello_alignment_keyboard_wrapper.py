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

"""Keyboard controls for GELLO leader alignment state transitions."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class GelloAlignmentKeyboardWrapper(gym.Wrapper):
    """Map keyboard presses to GELLO alignment state-machine methods."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.listener = KeyboardListener()

    @property
    def config(self):
        """Forward realworld env config access through this wrapper."""
        return self.env.config

    def reset(self, *, seed=None, options=None):
        self.listener.pop_pressed_keys()
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        event: str | None = None
        for key in self.listener.pop_pressed_keys():
            if key == "t":
                event = "align_requested"
                request_align = getattr(self.env, "request_align", None)
                if callable(request_align) and not request_align():
                    event = "align_rejected"
            elif key == "y":
                event = "teleop_confirmed"
                confirm_teleop = getattr(self.env, "confirm_teleop", None)
                if callable(confirm_teleop) and not confirm_teleop():
                    event = "teleop_rejected"
            elif key == "p":
                event = "policy_cancel"
                cancel_to_policy = getattr(self.env, "cancel_to_policy", None)
                if callable(cancel_to_policy):
                    cancel_to_policy()

        obs, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}
        info["gello_keyboard_event"] = event
        return obs, reward, terminated, truncated, info
