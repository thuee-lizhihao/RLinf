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

"""Phase2 观测链路 smoke：RealWorldEnv + 低风险动作序列，验证 states/main_images 稳定输出。"""

import os
from pathlib import Path

# 直接运行本脚本时未经过 .sh，需设置 EMBODIED_PATH 供 config 中 hydra.searchpath 解析
if "EMBODIED_PATH" not in os.environ:
    os.environ["EMBODIED_PATH"] = str(Path(__file__).resolve().parent)

import numpy as np
import torch

import hydra
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


def _check_obs(obs, step: int):
    """检查 obs 含 states 与 main_images，shape/dtype/范围正常，无 NaN/Inf。"""
    assert "states" in obs, f"step {step}: obs missing 'states'"
    assert "main_images" in obs, f"step {step}: obs missing 'main_images'"
    states = obs["states"]
    main_images = obs["main_images"]
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    if isinstance(main_images, torch.Tensor):
        main_images = main_images.detach().cpu().numpy()
    assert states.ndim >= 1 and states.size > 0, f"step {step}: states shape {getattr(obs['states'], 'shape', '?')}"
    assert main_images.ndim >= 3, f"step {step}: main_images.ndim >= 3, got {main_images.ndim}"
    # 至少 [1, 128, 128, 3]
    assert main_images.shape[-3] >= 128 and main_images.shape[-2] >= 128 and main_images.shape[-1] >= 3, (
        f"step {step}: main_images shape at least [..., 128, 128, 3], got {main_images.shape}"
    )
    # 对 uint8 图像，np.isnan/np.isinf 可能不支持；统一用 isfinite 更稳
    assert np.isfinite(states).all(), f"step {step}: states contains NaN or Inf"
    assert np.isfinite(main_images).all(), f"step {step}: main_images contains NaN or Inf"
    img_min, img_max = float(main_images.min()), float(main_images.max())
    return states.shape, main_images.shape, main_images.dtype, img_min, img_max


class SmokeObsWorker(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

    def run(self):
        num_steps = getattr(self.cfg, "smoke_num_steps", 200)
        obs, _ = self.env.reset()

        # 先检查 reset 后的 obs
        st_shape, img_shape, img_dtype, img_min, img_max = _check_obs(obs, -1)
        self.log_info(
            f"[reset] states.shape={st_shape}, main_images.shape={img_shape} dtype={img_dtype} "
            f"pixel_range=[{img_min}, {img_max}]"
        )

        # 低风险动作：全 0（可改为对称往返 +dx -> 0 -> -dx -> 0）
        action = np.zeros((1, 6), dtype=np.float32)
        for step in range(num_steps):
            obs, reward, term, trunc, info = self.env.step(action)
            st_shape, img_shape, img_dtype, img_min, img_max = _check_obs(obs, step)
            if step % 50 == 0:
                self.log_info(
                    f"step {step}: states.shape={st_shape}, main_images.shape={img_shape} "
                    f"pixel_range=[{img_min:.2f}, {img_max:.2f}]"
                )

        self.log_info(f"Smoke OK: {num_steps} steps completed, obs layout valid.")
        if getattr(self.env.video_cfg, "save_video", False):
            self.env.flush_video(video_sub_dir="smoke_obs")
        self.env.close()


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_smoke_obs_so101",
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    worker = SmokeObsWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    worker.run().wait()


if __name__ == "__main__":
    main()
