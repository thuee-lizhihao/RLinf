# SO101×RLinf 开发规范

> 本文件由 Claude Code 自动加载。记录 SO101 机械臂接入 RLinf 项目的开发规范与工作流，所有 AI 辅助开发均须遵循。

## 项目定位

- **RLinf**：分布式具身强化学习框架，使用 Ray 管理进程，Hydra 管理配置
- **本工作**：将 SO101 开源机械臂作为新真机载体接入 RLinf，与已有 Franka 实现平行，目标以 PR 形式贡献回 RLinf/RLinf 主仓库
- **开发语言**：Python 3.10-3.11
- **虚拟环境**：`source /home/zhihao/SO101_arm/venvs/so101/bin/activate`
- **SO101 核心代码位置**：`rlinf/envs/realworld/so101/`

## 分支与 Remote 配置

```
origin   → git@github.com:thuee-lizhihao/RLinf.git  （你的 fork，往这里推送）
upstream → https://github.com/RLinf/RLinf.git       （官方仓库，从这里拉取更新）
当前开发分支：so101_dev
```

## 每日开发启动 Checklist

每次打开电脑准备开发，按顺序执行：

```bash
# 1. 激活虚拟环境
source /home/zhihao/SO101_arm/venvs/so101/bin/activate

# 2. 进入仓库目录
cd /home/zhihao/SO101_arm/RLinf

# 3. 查看官方仓库是否有新的更新
git fetch upstream

# 4. 查看你的分支落后官方多少（有输出说明官方有更新）
git log HEAD..upstream/main --oneline

# 5. 如果官方有更新，把更新合并到你的分支（rebase = 把你的改动接在最新代码后面）
git rebase upstream/main

# 6. 把更新后的分支也同步到你的 fork
git push origin so101_dev
```

## Commit 规范（每次提交都要遵守）

### 格式
```
<类型>(<范围>): <简短描述>

<可选的详细说明>

Signed-off-by: 你的名字 <你的邮箱>
```

### 提交命令（`-s` 自动加 Signed-off-by）
```bash
git commit -s -m "feat(so101): add workspace calibration tool"
```

### 类型说明
| 类型 | 含义 | 示例场景 |
|------|------|---------|
| `feat` | 新功能 | 新增IK控制支持 |
| `fix` | 修bug | 修复关节限位判断 |
| `refactor` | 重构（不改功能） | 重写控制器结构 |
| `test` | 添加/修改测试 | 新增reach任务验收测试 |
| `docs` | 文档 | 更新注释或README |
| `chore` | 杂项（构建/配置） | 更新依赖、修改.gitignore |
| `style` | 格式（不改逻辑） | 统一缩进 |

### 范围
通常用 `so101`，也可以更具体：`so101/controller`、`so101/kinematics`、`so101/tasks`

### 好的 commit 示例
```
feat(so101): implement incremental IK solving for reach task

Use joint interpolation to bridge q_curr and q_target, limiting
per-step joint delta to avoid unreachable single-step jumps.

Signed-off-by: Zhihao Li <your@email.com>
```

### 不好的 commit 示例（不要这样写）
```
底层重构，test脚本简化
phase1 completed
修了个bug
```

## 代码规范

### 基本要求（来自 RLinf CONTRIBUTING.md）

1. **代码风格**：[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
2. **所有 public 函数/类**：必须有 docstring（Google 风格）
3. **所有函数参数和返回值**：必须有类型注解（type hints）
4. **日志**：用 `get_logger()` 而不是 `print()`
5. **断言**：必须附带有意义的错误信息，不能写空 assert

### 示例（规范写法）

```python
from rlinf.utils.logging import get_logger

logger = get_logger()

def compute_ik(
    target_pose: np.ndarray,
    q_current: np.ndarray,
    max_iter: int = 100,
) -> tuple[np.ndarray, bool]:
    """Compute inverse kinematics from current joint state to target pose.

    Args:
        target_pose: Target end-effector pose as [x, y, z, rx, ry, rz].
        q_current: Current joint positions in radians, shape (6,).
        max_iter: Maximum number of IK iterations.

    Returns:
        Tuple of (joint_positions, success_flag).
    """
    assert q_current.shape == (6,), f"q_current must have shape (6,), got {q_current.shape}"
    logger.info(f"Computing IK for target: {target_pose}")
    ...
```

### 配置 YAML 规范
- 只能有静态值，不能在 YAML 里做计算
- 参考 `examples/embodiment/config/env/realworld_so101_reach.yaml`

## 提交前 Checklist

每次 `git commit` 之前确认：

```bash
# 运行代码风格检查（pre-commit 已安装，git commit 时会自动跑，也可以手动提前跑）
pre-commit run --all-files

# 如果有 lint 报错，pre-commit 通常会自动修复，修复后重新 add
git add -u && git commit -s -m "..."
```

- [ ] 所有新增函数都有 docstring 和 type hints
- [ ] 没有 `print()` 语句（用 `logger.info()` 代替）
- [ ] 新功能有对应的测试文件（`tests/unit_tests/test_so101_*.py`）
- [ ] YAML 配置只有静态值

## PR 提交 Checklist

准备向官方仓库提 PR 时：

```bash
# 1. 确认分支是基于最新的官方 main
git fetch upstream
git rebase upstream/main

# 2. 整理 commit 历史（把 wip 类的临时 commit 合并整理）
# （具体操作届时再说）

# 3. 推送到你的 fork
git push origin so101_dev --force-with-lease
```

- [ ] 代码通过 `pre-commit run --all-files`
- [ ] 有单元测试或验收测试
- [ ] PR 标题格式：`feat(so101): ...`
- [ ] 填写 PR 模板的 Description 和 Checklist 部分
- [ ] 在 Motivation 部分说明 SO101 的社区价值（LeRobot 生态开源臂）
- [ ] 如果涉及性能，附上测试结果

## SO101 开发进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 依赖与节点分工 | ✅ 完成 |
| Phase 1 | 串口闭环 + 关节控制 | ✅ 完成 |
| Phase 2 | 相机 + 观测链路 | ✅ 完成 |
| Phase 3 | FK + 安全框架 | ✅ 完成 |
| Phase 4 | IK + Reach 任务 | 🔧 进行中（IK卡点：单步无法跨越大关节差） |
| Phase 5 | RL 训练接入 | ⬜ 未开始 |
| Phase 6 | 数据采集 + 人类干预 | ⬜ 未开始 |
| Phase 7 | Pick/Place、插入类任务 | ⬜ 未开始 |

**当前卡点**：IK 求解 `q_target` 时不考虑当前姿态 `q_curr`，导致单步关节差过大无法执行。

## 参考文件

- 开发规划：`SO101_INTEGRATION_PLAN.md`（仓库根目录）
- 贡献规范：`CONTRIBUTING.md`（仓库根目录）
- Franka 参考实现：`rlinf/envs/realworld/franka/`
- SO101 核心代码：`rlinf/envs/realworld/so101/`
- 示例配置：`examples/embodiment/config/`
