# SO101 机械臂接入 RLinf 分阶段规划

本文档描述将 SO101 机械臂接入 RLinf 的理想能力清单与从易到难的实现阶段，供后续开发参考。整体思路参考 Franka 接入 RLinf 的架构（Scheduler 硬件发现 → RealWorldEnv 统一观测/包装 → Controller Worker 控硬件 → Task Env 定义奖励/终止），并结合 SO101 自身特性（串口 Feetech、5DoF+夹爪、通常无力矩/力传感、相机可挂 RealSense）。

---

## 一、理想情况下接入后可支持的能力

### 1.1 基础硬件能力

| 能力 | 说明 |
|------|------|
| **串口连接/断开、健康检查** | 端口存在、bus 是否连接、超时/重连 |
| **关节位置读写** | 读 5 个关节 + 夹爪位置；写关节目标位置（position mode） |
| **夹爪控制** | 至少二值开/合；理想情况下支持连续开合（0~100） |
| **标定管理** | 能加载 LeRobot calibration；缺失时有清晰报错或自动引导生成 |
| **相机输入** | 1~N 个 RealSense，输出 128×128 RGB（对齐 RLinf 真实环境格式） |

### 1.2 RLinf RealWorld 统一接口能力（对齐 Franka）

| 能力 | 说明 |
|------|------|
| **观测格式对齐** | `{"state": {...}, "frames": {...}}` → 由 RealWorldEnv 包装为 `states/main_images/...` |
| **通用 wrappers 兼容** | 相对坐标、欧拉角观测、空间鼠标干预等（架构可接，按需开启） |
| **录视频/日志指标** | episode return、success_once、video dump |

### 1.3 动作空间（从易到难）

| 层级 | 说明 |
|------|------|
| **Joint-space policy** | 直接输出 5 关节 + 夹爪（最稳、最易落地） |
| **EE delta 控制** | 7 维 [Δx, Δy, Δz, Δr, Δp, Δy, gripper]，内部 IK → 关节目标 |
| **可选高级控制** | 轨迹插值、速度限制、碰撞/自碰避免（SO101 精度/刚性不足时尤其重要） |

### 1.4 任务能力（从易到难）

| 任务 | 说明 |
|------|------|
| **Reach** | 末端到达目标位姿（类似 Franka 的 peg-insertion 对位部分） |
| **Pick/Place** | 桌面抓取放置（对 SO101 很现实，门槛低于插入类） |
| **插入类（Peg/Charger）** | 对精度、顺应性、夹爪稳定性要求高；SO101 需更强容错与更慢控制频率 |

> **重要差异**：Franka 依赖 ROS + 阻抗控制，天然“顺应”；SO101 常为串口位置控制，更需要软件侧的安全限制、动作平滑、低速控制与任务容错。

---

## 二、从易到难的实现阶段规划

按「先打通闭环 → 再提高可用性 → 再做复杂任务」的顺序划分阶段，每阶段给出目标、主要工作与验收标准。

---

### Phase 0：依赖与节点分工（最先做）

**目标**：控制节点能稳定跑 SO101 控制与相机；训练节点能跑 RL。

**主要工作**：
- 控制节点：LeRobot(Feetech) + 串口权限 + RealSense 驱动
- 训练/rollout 节点：RLinf 训练依赖
- 集群配置里增加 `node_group: so101`（类似 Franka 的 `franka`）

**验收标准**：
- 在控制节点上能稳定枚举到串口、枚举到相机序列号
- 环境能启动（可先开 dummy 模式）

---

### Phase 1：真机串口闭环 + 关节空间动作序列（不再依赖 dummy，先把“能控”做稳）

**目标**：最小可控：读 joint → 发 joint goal → 再读 joint，形成闭环；并能执行一段**安全的固定关节序列**用于反复回归测试。

**主要工作**：
- `SO101Controller.connect()/get_joint_positions()/command_joints()/open/close_gripper()`
- 标定文件策略（缺失时报错清晰；可选自动校准）
- 安全策略（建议最少一层）：关节限位/限幅、低速/低频（例如 5–10 Hz 起步）、必要时提供急停/停机手段
- 增加一个**非交互**的 motion smoke test（自动跑固定关节序列并读回校验），避免调试依赖手工输入

**验收标准**：
- 执行固定关节序列，机械臂动作一致、无明显抖动/失步
- 能在 RLinf worker/进程里长期运行不掉线（重复运行 N 次不崩溃/不失联）

---

### Phase 2：观测链路（真实相机 + RealWorldEnv 包装）（在“能控”稳定后再打通）

**目标**：把 RealWorldEnv 的「图像 + 状态」整条链跑通：`SO101Env` 产出 obs → `RealWorldEnv` 包装得到 `states/main_images`（以及可选 `extra_view_images`）。

**主要工作**：
- `SO101Env.reset()/step()` 在 `is_dummy=False`（真机）下能返回完整 obs（state + frames）
- RealWorldEnv 包装后能得到 `states/main_images`
- （可选）开启 `video_cfg.save_video` 验证视频输出与基础日志指标

**验收标准**：
- 能跑一小段 rollout/step，观测维度与图像输出正常、无崩溃

---

### Phase 3：末端位姿状态估计（FK）+ 基础 EE 安全框（对 RL 观测很关键）

**目标**：有可信的 `tcp_pose(7)`、`tcp_vel(6)`，并能把动作限制在安全范围。

**主要工作**：
- 基于 URDF 的 FK（以及粗略的速度估计）
- 增加 workspace clip、姿态 clip、关节限位（至少一层）

**验收标准**：
- `tcp_pose` 连续稳定；速度估计不炸；越界动作会被 clip 而不是撞

---

### Phase 4：EE delta 控制（IK）与 Reach 任务（第一个「可学任务」）

**目标**：对齐 Franka 的动作范式，先做最简单的稀疏/稠密 reward 任务。

**主要工作**：
- IK 求解失败时的降级策略（保持原位/只做 position）
- `SO101ReachEnv`（success 判定、dense reward 可选）

**验收标准**：
- 脚本式策略能完成 reach；随机策略也能稳定运行不崩

---

### Phase 5：接入 RL 训练流程（SAC/RLPD/CNN）（把「能控」变成「能训」）

**目标**：复用 RLinf 的真实世界训练结构：actor/rollout 在 GPU，env 在 so101 控制节点。

**主要工作**：
- 编写 `examples/embodiment/config/realworld_so101_*.yaml`（对标 `realworld_peginsertion_rlpd_cnn_async.yaml`）
- 明确 action_dim/state_dim（SO101 state 更少；动作若用 EE delta 则 7 维）

**验收标准**：
- 能跑起来训练 loop；buffer 增长；reward 曲线有响应

---

### Phase 6：数据采集 + 人类干预（提高数据效率与安全性）

**目标**：像 Franka 一样支持「先收 demo → RLPD/混合训练」。

**主要工作**：
- 空间鼠标/键鼠干预（wrappers 级别接入）
- 采集脚本与数据格式对齐 replay buffer

**验收标准**：
- 能收集 N 个 episode 并复现回放；训练能利用这些数据加速

---

### Phase 7：更复杂任务（Pick/Place → 插入类）

**目标**：逐步增加任务难度，并补齐 SO101 相对薄弱的控制能力。

**主要工作（建议顺序）**：
- Pick/Place（容错高）→ 再挑战插入/对孔
- 动作平滑（插值/低通）、速度限制、失败恢复、自动重置姿态

**验收标准**：
- 任务成功率达到可用水平；长时间运行稳定

---

## 三、注意事项（SO101 相对 Franka 易踩坑点）

| 项 | 建议 |
|----|------|
| **控制频率** | 串口 + 电机响应决定上限，建议先从 5–10 Hz 做稳，别贪高 |
| **策略形态** | 先做 joint-space policy 再做 IK；SO101 的 IK/几何精度通常比 Franka 更脆 |
| **插入类任务** | 无阻抗控制时，需靠低速、动作限幅、接触容错设计做「软件顺应」 |

---

## 四、阶段依赖关系简图

```
Phase 0 (依赖/节点)
    ↓
Phase 1 (串口闭环 + 关节序列)
    ↓
Phase 2 (观测链路)
    ↓
Phase 3 (FK + 安全框)
    ↓
Phase 4 (IK + Reach)
    ↓
Phase 5 (RL 训练接入)
    ↓
Phase 6 (采集 + 干预) ──→ Phase 7 (Pick/Place → 插入)
```

---

*文档版本：初稿*  
*最后更新：按规划讨论整理*
